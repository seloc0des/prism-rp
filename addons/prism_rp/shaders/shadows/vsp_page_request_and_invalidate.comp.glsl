#version 460
#extension GL_GOOGLE_include_directive : require

#include "vsp_common.glsl"

// ============================================================================
// Prism RP — Virtual Shadow Pages invalidation & screen-space page allocator
//
// Mode 0 (VSP_PASS_INVALIDATE): dispatch (128 / 8, 128 / 8, PRISM_VSP_CLIPMAP_LEVELS), one thread
//   per virtual page.
//   - Ages allocated pages; frees those unused for evict_after_frames frames to the free stack.
//   - Invalidates pages scrolled into view, touched by moved dynamic AABBs, or all of them when
//     the sun moved (force_invalidate_all).
//   - Clears last frame's REQUESTED / DIRTY bits.
//
// Mode 1 (VSP_PASS_REQUEST_AND_ALLOC): dispatch (ceil(width / 8), ceil(height / 8), 1).
//   - Reconstructs each pixel's world position from the reverse-Z depth buffer.
//   - Requests every page vsp_sample_shadows can read for it: the pages its filter footprint
//     covers at its level, plus the coarsest level's page (the sampler's last fallback).
//   - Allocates missing pages from the free stack and queues pages without valid depth for
//     rendering. A page is marked VALID only once it is queued.
// ============================================================================

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std140) uniform VspUniformBlock {
    VspUniformUBO u_vsp;
};

// Reverse-Z primary depth buffer (D32_SFLOAT)
layout(set = 0, binding = 1) uniform sampler2D u_depth_buffer;

// Page table: 2D array [128 x 128 x 8 levels] of R32_UINT entries
layout(set = 0, binding = 2, r32ui) uniform uimage2DArray u_page_table;

// Dynamic dirty AABB list (moving / animated shadow casters)
layout(set = 0, binding = 3, std430) readonly buffer DynamicDirtyAabbBuffer {
    VspDynamicDirtyAabb b_dirty_aabbs[];
};

// Free physical page stack (PRISM_VSP_MAX_PHYSICAL_PAGES entries; size in b_state.free_page_count)
layout(set = 0, binding = 4, std430) buffer FreePageStackBuffer {
    uint b_free_pages[];
};

layout(set = 0, binding = 5, std430) buffer VspFrameStateBuffer {
    VspFrameState b_state;
};

// Pages to render this frame (viewport_and_counts.w entries)
layout(set = 0, binding = 6, std430) writeonly buffer DirtyPageQueueBuffer {
    VspPageWorkItem b_dirty_pages[];
};

bool aabb_overlaps_page_xy(vec3 aabb_min, vec3 aabb_max, VspClipmapLevel lvl, ivec2 linear_page) {
    vec2 uv_min = vec2( 1e30);
    vec2 uv_max = vec2(-1e30);
    for (int i = 0; i < 8; ++i) {
        vec3 corner = vec3(
            (i & 1) != 0 ? aabb_max.x : aabb_min.x,
            (i & 2) != 0 ? aabb_max.y : aabb_min.y,
            (i & 4) != 0 ? aabb_max.z : aabb_min.z
        );
        vec4 clip = lvl.light_view_proj * vec4(corner, 1.0);
        vec2 uv   = (clip.xy / clip.w) * 0.5 + 0.5;
        uv_min    = min(uv_min, uv);
        uv_max    = max(uv_max, uv);
    }
    vec2 page_uv_min = vec2(linear_page) / float(PRISM_VSP_VIRTUAL_PAGES_DIM);
    vec2 page_uv_max = vec2(linear_page + ivec2(1)) / float(PRISM_VSP_VIRTUAL_PAGES_DIM);
    return all(lessThanEqual(uv_min, page_uv_max)) && all(greaterThanEqual(uv_max, page_uv_min));
}

void run_invalidation_pass() {
    ivec3 coord = ivec3(gl_GlobalInvocationID.xyz);
    if (any(greaterThanEqual(coord, ivec3(ivec2(PRISM_VSP_VIRTUAL_PAGES_DIM), int(PRISM_VSP_CLIPMAP_LEVELS))))) {
        return;
    }

    uint pte = imageLoad(u_page_table, coord).r;
    if ((pte & VSP_PTE_ALLOCATED) == 0u) {
        if (pte != 0u) {
            imageStore(u_page_table, coord, uvec4(0u)); // drop a failed claim from last frame
        }
        return;
    }

    // Age; free pages nobody has asked for in a while
    bool requested = (pte & VSP_PTE_REQUESTED) != 0u;
    uint age = requested ? 0u : min(((pte & VSP_PTE_AGE_MASK) >> VSP_PTE_AGE_SHIFT) + 1u, 255u);
    if (!requested && age >= max(u_vsp.evict_after_frames, 1u)) {
        uint slot = atomicAdd(b_state.free_page_count, 1u);
        b_free_pages[slot] = pte & VSP_PTE_PAGE_MASK;
        imageStore(u_page_table, coord, uvec4(0u));
        return;
    }

    VspClipmapLevel lvl = u_vsp.levels[coord.z];
    ivec2 scroll_delta  = lvl.page_offset_and_pad.zw;
    ivec2 linear_page   = toroidal_to_linear_vpage(coord.xy, lvl.page_offset_and_pad.xy);
    int dim = int(PRISM_VSP_VIRTUAL_PAGES_DIM);

    // Pages that just wrapped across the toroidal edge now cover new ground
    bool scrolled_in = (scroll_delta.x > 0 && linear_page.x >= dim - scroll_delta.x) ||
                       (scroll_delta.x < 0 && linear_page.x < -scroll_delta.x) ||
                       (scroll_delta.y > 0 && linear_page.y >= dim - scroll_delta.y) ||
                       (scroll_delta.y < 0 && linear_page.y < -scroll_delta.y);

    bool invalidate = (u_vsp.force_invalidate_all != 0u) || scrolled_in;
    for (uint i = 0u; !invalidate && i < u_vsp.viewport_and_counts.z; ++i) {
        invalidate = aabb_overlaps_page_xy(b_dirty_aabbs[i].min_ws_and_pad.xyz, b_dirty_aabbs[i].max_ws_and_pad.xyz, lvl, linear_page);
    }

    uint next_pte = (pte & ~(VSP_PTE_REQUESTED | VSP_PTE_DIRTY | VSP_PTE_AGE_MASK)) | (age << VSP_PTE_AGE_SHIFT);
    if (invalidate) {
        next_pte &= ~VSP_PTE_VALID;
    }
    imageStore(u_page_table, coord, uvec4(next_pte, 0u, 0u, 0u));
}

void request_page(uint level, ivec2 linear_page) {
    ivec2 vpage = linear_to_toroidal_vpage(linear_page, u_vsp.levels[level].page_offset_and_pad.xy);
    ivec3 pte_coord = ivec3(vpage, int(level));

    // Fast path: cached and valid
    uint prev_pte = imageAtomicOr(u_page_table, pte_coord, VSP_PTE_REQUESTED);
    if ((prev_pte & VSP_PTE_VALID) != 0u) {
        return;
    }

    // Slow path: claim the page so exactly one thread allocates / queues it this frame
    uint claimed_pte = imageAtomicOr(u_page_table, pte_coord, VSP_PTE_DIRTY);
    if ((claimed_pte & VSP_PTE_DIRTY) != 0u) {
        return;
    }

    uint phys_page = claimed_pte & VSP_PTE_PAGE_MASK;
    if ((claimed_pte & VSP_PTE_ALLOCATED) == 0u) {
        uint top = atomicAdd(b_state.free_page_count, 0xFFFFFFFFu); // pop
        if (top == 0u || top > PRISM_VSP_MAX_PHYSICAL_PAGES) {
            // Pool exhausted: give the count back. The page stays claimed (no retries this frame)
            // and unallocated; the sampler falls back to a coarser level.
            atomicAdd(b_state.free_page_count, 1u);
            return;
        }
        phys_page = b_free_pages[top - 1u] & VSP_PTE_PAGE_MASK;
    }

    uint q_slot = atomicAdd(b_state.dirty_page_count, 1u);
    if (q_slot >= u_vsp.viewport_and_counts.w) {
        // Render queue full: keep the page but leave it invalid, so nothing reads stale depth.
        // It is requested again next frame.
        imageStore(u_page_table, pte_coord, uvec4(VSP_PTE_DIRTY | VSP_PTE_ALLOCATED | VSP_PTE_REQUESTED | phys_page, 0u, 0u, 0u));
        return;
    }

    b_dirty_pages[q_slot] = VspPageWorkItem(vsp_pack_level_vpage(level, vpage), phys_page);
    imageStore(u_page_table, pte_coord, uvec4(VSP_PTE_VALID | VSP_PTE_DIRTY | VSP_PTE_ALLOCATED | VSP_PTE_REQUESTED | phys_page, 0u, 0u, 0u));

    atomicMax(b_state.clear_dispatch.x, q_slot + 1u);
    atomicMin(b_state.dirty_page_rect[level].x, uint(linear_page.x));
    atomicMin(b_state.dirty_page_rect[level].y, uint(linear_page.y));
    atomicMax(b_state.dirty_page_rect[level].z, uint(linear_page.x));
    atomicMax(b_state.dirty_page_rect[level].w, uint(linear_page.y));
}

void run_request_and_alloc_pass() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(pixel, ivec2(u_vsp.viewport_and_counts.xy)))) {
        return;
    }

    float rev_z = texelFetch(u_depth_buffer, pixel, 0).r;
    if (rev_z <= 0.0) {
        return; // sky / far plane: no shadow needed
    }

    vec2 ndc       = (vec2(pixel) + 0.5) / vec2(u_vsp.viewport_and_counts.xy) * 2.0 - 1.0;
    vec4 world_h   = u_vsp.inv_view_proj * vec4(ndc, rev_z, 1.0);
    vec3 world_pos = world_h.xyz / world_h.w;
    float view_dist = max(-(u_vsp.view * vec4(world_pos, 1.0)).z, u_vsp.camera_pos_and_near.w);

    // Same level choice as vsp_sample_shadows: the selected level, or the first coarser one
    // that contains the point.
    uint level = select_vsp_clipmap_level(view_dist, u_vsp.levels[0].world_origin_scale.w, u_vsp.pcss_params.w);
    vec2 clip_uv;
    float light_z;
    while (!project_world_to_vsp_level(world_pos, u_vsp.levels[level], clip_uv, light_z)) {
        if (++level >= PRISM_VSP_CLIPMAP_LEVELS) {
            return; // beyond every level
        }
    }

    // Every page the sampler's taps can reach: blocker search / filter radius (max filter texels),
    // plus a few texels for the normal-offset bias.
    float reach_uv = (u_vsp.pcss_params.z + 4.0) / float(PRISM_VSP_VIRTUAL_RES);
    ivec2 page_min = clip_uv_to_linear_vpage(clip_uv - reach_uv);
    ivec2 page_max = min(clip_uv_to_linear_vpage(clip_uv + reach_uv), page_min + 2); // at most 3x3
    for (int py = page_min.y; py <= page_max.y; ++py) {
        for (int px = page_min.x; px <= page_max.x; ++px) {
            request_page(level, ivec2(px, py));
        }
    }

    // The coarsest level backs up any page that couldn't be allocated or rendered
    uint coarsest = PRISM_VSP_CLIPMAP_LEVELS - 1u;
    if (level != coarsest && project_world_to_vsp_level(world_pos, u_vsp.levels[coarsest], clip_uv, light_z)) {
        request_page(coarsest, clip_uv_to_linear_vpage(clip_uv));
    }
}

void main() {
    if (u_vsp.pass_mode == VSP_PASS_INVALIDATE) {
        run_invalidation_pass();
    } else {
        run_request_and_alloc_pass();
    }
}
