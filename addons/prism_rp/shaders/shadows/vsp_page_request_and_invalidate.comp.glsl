#version 460

#include "../include/vsp_common.glsl"

// ============================================================================
// Prism RP — Virtual Shadow Pages Invalidation & Screen-Space Page Allocator
//
// Mode 0 (VSP_PASS_INVALIDATE):
//   Dispatched over all virtual pages: (16, 16, PRISM_VSP_CLIPMAP_LEVELS).
//   - Clears VSP_PTE_REQUESTED from last frame.
//   - Invalidates pages scrolled out of the toroidal window or intersecting
//     moved dynamic AABBs (or all pages if sun direction shifted).
//
// Mode 1 (VSP_PASS_REQUEST_AND_ALLOC):
//   Dispatched over screen pixels: (ceil(width / 8), ceil(height / 8), 1).
//   - Reconstructs world position from Reverse-Z D32_SFLOAT depth.
//   - Marks covering virtual page VSP_PTE_REQUESTED.
//   - If page is unallocated, pops a physical page index from FreePageStack.
//   - If page is not VSP_PTE_VALID (newly allocated or invalidated), queues it
//     once into VspDirtyPageQueue for rasterization.
// ============================================================================

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std140) uniform VspUniformBlock {
    VspUniformUBO u_vsp;
};

// Reverse-Z primary depth buffer (D32_SFLOAT)
layout(set = 0, binding = 1) uniform sampler2D u_depth_buffer;

// 2D Texture Array [16 x 16 x 8 layers] of R32_UINT Page Table Entries
layout(set = 0, binding = 2, r32ui) uniform uimage2DArray u_page_table;

// Dynamic dirty AABB list (for moving/animated objects)
layout(set = 0, binding = 3, std430) readonly buffer DynamicDirtyAabbBuffer {
    VspDynamicDirtyAabb b_dirty_aabbs[];
};

// Free physical page pool stack (size = 4096)
layout(set = 0, binding = 4, std430) buffer FreePageStackBuffer {
    uint b_free_pages[];
};

// Atomic counters:
//   b_vsp_counters[0] = free_page_stack_top (initialized to 4096 at startup)
//   b_vsp_counters[1] = dirty_page_queue_count (reset to 0 before Mode 1)
layout(set = 0, binding = 5, std430) buffer VspCountersBuffer {
    uint b_vsp_counters[];
};

// Output queue of dirty pages to rasterize this frame
layout(set = 0, binding = 6, std430) writeonly buffer DirtyPageQueueBuffer {
    VspPageWorkItem b_dirty_pages[];
};

bool aabb_overlaps_page_xy(
    vec3 aabb_min,
    vec3 aabb_max,
    VspClipmapLevel lvl,
    ivec2 linear_page
) {
    // Project 8 corners of world AABB into light clip UV [0, 1]
    vec2 uv_min = vec2( 1e9);
    vec2 uv_max = vec2(-1e9);
    for (int i = 0; i < 8; ++i) {
        vec3 corner = vec3(
            (i & 1) != 0 ? aabb_max.x : aabb_min.x,
            (i & 2) != 0 ? aabb_max.y : aabb_min.y,
            (i & 4) != 0 ? aabb_max.z : aabb_min.z
        );
        vec4 clip = lvl.light_view_proj * vec4(corner, 1.0);
        vec2 uv   = (clip.xy / max(clip.w, 1e-6)) * 0.5 + 0.5;
        uv_min    = min(uv_min, uv);
        uv_max    = max(uv_max, uv);
    }

    vec2 page_uv_min = vec2(linear_page) / float(PRISM_VSP_VIRTUAL_PAGES_DIM);
    vec2 page_uv_max = vec2(linear_page + ivec2(1)) / float(PRISM_VSP_VIRTUAL_PAGES_DIM);

    return all(lessThanEqual(uv_min, page_uv_max)) &&
           all(greaterThanEqual(uv_max, page_uv_min));
}

void run_invalidation_pass() {
    ivec3 coord = ivec3(gl_GlobalInvocationID.xyz);
    if (coord.x >= int(PRISM_VSP_VIRTUAL_PAGES_DIM) ||
        coord.y >= int(PRISM_VSP_VIRTUAL_PAGES_DIM) ||
        coord.z >= int(PRISM_VSP_CLIPMAP_LEVELS)) {
        return;
    }

    uint pte = imageLoad(u_page_table, coord).r;
    if ((pte & VSP_PTE_ALLOCATED) == 0u) {
        return;
    }

    VspClipmapLevel lvl = u_vsp.levels[coord.z];
    ivec2 scroll_delta  = lvl.page_offset_and_pad.zw;

    // Recover linear page index [0..15] from toroidal coordinate
    ivec2 linear_page = (coord.xy - lvl.page_offset_and_pad.xy) & int(PRISM_VSP_VIRTUAL_PAGES_DIM - 1u);

    // Check if this page just wrapped across the toroidal boundary due to camera scroll
    bool scrolled_out = false;
    if (scroll_delta.x > 0 && linear_page.x >= int(PRISM_VSP_VIRTUAL_PAGES_DIM) - scroll_delta.x) scrolled_out = true;
    if (scroll_delta.x < 0 && linear_page.x < -scroll_delta.x) scrolled_out = true;
    if (scroll_delta.y > 0 && linear_page.y >= int(PRISM_VSP_VIRTUAL_PAGES_DIM) - scroll_delta.y) scrolled_out = true;
    if (scroll_delta.y < 0 && linear_page.y < -scroll_delta.y) scrolled_out = true;

    bool invalidate = (u_vsp.force_invalidate_all != 0u) || scrolled_out;

    // Check intersection with dynamic moving object AABBs
    if (!invalidate) {
        uint aabb_count = u_vsp.viewport_and_counts.z;
        for (uint i = 0u; i < aabb_count; ++i) {
            VspDynamicDirtyAabb box = b_dirty_aabbs[i];
            if (aabb_overlaps_page_xy(box.min_ws_and_pad.xyz, box.max_ws_and_pad.xyz, lvl, linear_page)) {
                invalidate = true;
                break;
            }
        }
    }

    // Clear per-frame REQUESTED & DIRTY flags; strip VALID if invalidated
    uint next_pte = pte & ~(VSP_PTE_REQUESTED | VSP_PTE_DIRTY);
    if (invalidate) {
        next_pte &= ~VSP_PTE_VALID;
    }

    imageStore(u_page_table, coord, uvec4(next_pte, 0u, 0u, 0u));
}

void run_request_and_alloc_pass() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= int(u_vsp.viewport_and_counts.x) ||
        pixel.y >= int(u_vsp.viewport_and_counts.y)) {
        return;
    }

    float rev_z = texelFetch(u_depth_buffer, pixel, 0).r;
    if (rev_z <= 0.0) {
        return; // Sky / far plane -> no shadow page needed
    }

    vec2 uv  = (vec2(pixel) + 0.5) / vec2(u_vsp.viewport_and_counts.xy);
    vec2 ndc = uv * 2.0 - 1.0;

    vec4 world_h   = u_vsp.inv_view_proj * vec4(ndc, rev_z, 1.0);
    vec3 world_pos = world_h.xyz / max(world_h.w, 1e-6);

    vec3 view_pos   = (u_vsp.view * vec4(world_pos, 1.0)).xyz;
    float view_dist = max(-view_pos.z, u_vsp.camera_pos_and_near.w);

    uint level = select_vsp_clipmap_level(
        view_dist,
        u_vsp.levels[0].world_origin_scale.w,
        u_vsp.pcss_params.w
    );

    VspClipmapLevel lvl = u_vsp.levels[level];
    vec2 clip_uv;
    float light_z;
    if (!project_world_to_vsp_level(world_pos, lvl, clip_uv, light_z)) {
        return;
    }

    ivec2 vpage = clip_uv_to_toroidal_vpage(clip_uv, lvl.page_offset_and_pad.xy);
    ivec3 pte_coord = ivec3(vpage.x, vpage.y, int(level));

    // Atomically mark page as requested this frame
    uint prev_pte = imageAtomicOr(u_page_table, pte_coord, VSP_PTE_REQUESTED);

    // Fast path: already valid and cached -> zero allocation or rasterization needed
    if ((prev_pte & VSP_PTE_VALID) != 0u) {
        return;
    }

    // Slow path: page is either unallocated or invalidated.
    // Use bit VSP_PTE_DIRTY as an atomic spin-claim so only ONE thread allocates/enqueues it.
    uint claimed_pte = imageAtomicOr(u_page_table, pte_coord, VSP_PTE_DIRTY);
    if ((claimed_pte & VSP_PTE_DIRTY) != 0u) {
        return; // Another pixel thread already claimed this page this frame
    }

    uint phys_page = claimed_pte & VSP_PTE_PAGE_MASK;
    if ((claimed_pte & VSP_PTE_ALLOCATED) == 0u) {
        // Pop a free physical page from the 4,096-entry pool
        uint stack_idx = atomicAdd(b_vsp_counters[0], 0xFFFFFFFFu); // atomic decrement
        if (stack_idx == 0u || stack_idx > PRISM_VSP_MAX_PHYSICAL_PAGES) {
            // Pool exhausted -> restore counter & release dirty claim
            atomicAdd(b_vsp_counters[0], 1u);
            imageAtomicAnd(u_page_table, pte_coord, ~VSP_PTE_DIRTY);
            return;
        }
        phys_page = b_free_pages[stack_idx - 1u] & VSP_PTE_PAGE_MASK;
    }

    // Commit allocated physical page + VALID + DIRTY + REQUESTED bits
    uint final_pte = VSP_PTE_VALID | VSP_PTE_DIRTY | VSP_PTE_ALLOCATED | VSP_PTE_REQUESTED | phys_page;
    imageStore(u_page_table, pte_coord, uvec4(final_pte, 0u, 0u, 0u));

    // Enqueue dirty page into VspDirtyPageQueue for shadow rasterization
    uint q_slot = atomicAdd(b_vsp_counters[1], 1u);
    if (q_slot < u_vsp.viewport_and_counts.w) {
        VspPageWorkItem item;
        item.level_and_vpage = ((level & 0xFFu) << 24u)
                             | ((uint(vpage.y) & 0xFFu) << 8u)
                             | (uint(vpage.x) & 0xFFu);
        item.physical_page   = phys_page;
        b_dirty_pages[q_slot] = item;
    }
}

void main() {
    if (u_vsp.pass_mode == VSP_PASS_INVALIDATE) {
        run_invalidation_pass();
    } else {
        run_request_and_alloc_pass();
    }
}
