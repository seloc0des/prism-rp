#version 460
#extension GL_GOOGLE_include_directive : require

#include "visibility_common.glsl"
#include "vsp_common.glsl"

// ============================================================================
// Prism RP — Virtual Shadow Pages meshlet culling
// Dispatch (ceil(total_meshlet_candidates / 64), 1, 1), one thread per candidate meshlet.
// For every clipmap level, finds the pages queued this frame that the meshlet's bounding sphere
// overlaps and appends one (meshlet, page) raster item for each, growing
// VspFrameState.raster_dispatch. Rasterization then costs only what actually overlaps, instead of
// every page scanning the whole scene. The search is limited to each level's dirty-page rect.
// If the raster queue is full, the page loses VALID so nothing samples it half-drawn (it is
// re-rendered next frame; the sampler falls back to a coarser level meanwhile).
// ============================================================================

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std140) uniform VspUniformBlock {
    VspUniformUBO u_vsp;
};

layout(set = 0, binding = 1, std430) readonly buffer MeshletBuffer {
    GpuMeshlet b_meshlets[];
};

layout(set = 0, binding = 2, std430) readonly buffer InstanceBuffer {
    GpuInstance b_instances[];
};

layout(set = 0, binding = 3, std430) readonly buffer CandidateBuffer {
    uvec2 b_candidates[]; // (instance_id, meshlet_local_id)
};

layout(set = 0, binding = 4, r32ui) uniform uimage2DArray u_page_table;

layout(set = 0, binding = 5, std430) writeonly buffer RasterItemBuffer {
    VspRasterItem b_raster_items[];
};

layout(set = 0, binding = 6, std430) buffer VspFrameStateBuffer {
    VspFrameState b_state;
};

void main() {
    uint c_idx = gl_GlobalInvocationID.x;
    if (c_idx >= u_vsp.total_meshlet_candidates) {
        return;
    }

    uvec2 item = b_candidates[c_idx];
    GpuInstance inst = b_instances[item.x];
    if ((inst.flags & PRISM_INSTANCE_FLAG_VISIBLE) == 0u || (inst.flags & PRISM_INSTANCE_FLAG_CAST_SHADOW) == 0u ||
            item.y >= inst.meshlet_count) {
        return;
    }
    GpuMeshlet m = b_meshlets[inst.meshlet_offset + item.y];
    if (m.triangle_count == 0u) {
        return;
    }

    mat4 M = inst.world_from_local;
    vec3 world_center  = (M * vec4(m.bound_sphere.xyz, 1.0)).xyz;
    float world_radius = m.bound_sphere.w * max(length(M[0].xyz), max(length(M[1].xyz), length(M[2].xyz)));

    for (uint level = 0u; level < PRISM_VSP_CLIPMAP_LEVELS; ++level) {
        uvec4 dirty = b_state.dirty_page_rect[level];
        if (dirty.x > dirty.z) {
            continue; // nothing queued at this level
        }

        // Sphere footprint in this level's UV (the light projection is orthographic: no depth scaling)
        VspClipmapLevel lvl = u_vsp.levels[level];
        vec2 uv_c  = (lvl.light_view_proj * vec4(world_center, 1.0)).xy * 0.5 + 0.5;
        float uv_r = world_radius / max(lvl.world_origin_scale.w, 1e-6);
        if (any(lessThan(uv_c + uv_r, vec2(0.0))) || any(greaterThan(uv_c - uv_r, vec2(1.0)))) {
            continue; // outside the level
        }
        ivec2 page_min = max(clip_uv_to_linear_vpage(uv_c - uv_r), ivec2(dirty.xy));
        ivec2 page_max = min(clip_uv_to_linear_vpage(uv_c + uv_r), ivec2(dirty.zw));

        for (int py = page_min.y; py <= page_max.y; ++py) {
            for (int px = page_min.x; px <= page_max.x; ++px) {
                ivec2 vpage = linear_to_toroidal_vpage(ivec2(px, py), lvl.page_offset_and_pad.xy);
                ivec3 pte_coord = ivec3(vpage, int(level));
                uint pte = imageLoad(u_page_table, pte_coord).r;
                if ((pte & (VSP_PTE_VALID | VSP_PTE_DIRTY)) != (VSP_PTE_VALID | VSP_PTE_DIRTY)) {
                    continue; // not queued for rendering this frame
                }
                uint slot = atomicAdd(b_state.raster_item_count, 1u);
                if (slot >= u_vsp.max_raster_items) {
                    imageAtomicAnd(u_page_table, pte_coord, ~VSP_PTE_VALID);
                    continue;
                }
                b_raster_items[slot] = VspRasterItem(item.x, item.y, vsp_pack_level_vpage(level, vpage), pte & VSP_PTE_PAGE_MASK);
                atomicMax(b_state.raster_dispatch.x, slot + 1u);
            }
        }
    }
}
