#version 460

#include "../include/visibility_common.glsl"
#include "../include/vsp_common.glsl"

// ============================================================================
// Prism RP — Virtual Shadow Pages Compute Meshlet Rasterizer
// Dispatched with workgroups = (dirty_page_count, 1, 1), local_size = (8, 8, 1).
// Each workgroup rasterizes one dirty 128x128 physical shadow page:
//   1. Clears its 128x128 tile in the physical atlas to 1.0 (far depth).
//   2. Streams candidate scene meshlets, culls against the page's light-space
//      2D bounding box, and software-rasterizes triangles directly into the
//      128x128 physical page using atomicMin on ordered uint depth bits.
// ============================================================================

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

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

layout(set = 0, binding = 4, std430) readonly buffer GlobalVertexPosBuffer {
    vec4 b_positions[];
};

layout(set = 0, binding = 5, std430) readonly buffer MeshletVertexIndicesBuffer {
    uint b_meshlet_vertices[];
};

layout(set = 0, binding = 6, std430) readonly buffer MeshletTrianglesBuffer {
    uint b_meshlet_triangles[];
};

layout(set = 0, binding = 7, std430) readonly buffer DirtyPageQueueBuffer {
    VspPageWorkItem b_dirty_pages[];
};

// Physical 8192x8192 shadow atlas bound as R32_UINT alias for atomicMin depth writes
layout(set = 0, binding = 8, r32ui) uniform uimage2D u_physical_shadow_atlas;

// Shared-memory cache for transformed meshlet vertex positions in page-pixel space
// xy = pixel coords in [0..128], z = linear light depth in [0..1]
shared vec3 s_page_verts[PRISM_MESHLET_MAX_VERTICES];
shared uint s_meshlet_accepted;

float edge_func(vec2 a, vec2 b, vec2 c) {
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

void main() {
    uint page_work_idx = gl_WorkGroupID.x;
    uint local_tid     = gl_LocalInvocationIndex; // 0..63

    VspPageWorkItem work = b_dirty_pages[page_work_idx];
    uint level      = (work.level_and_vpage >> 24u) & 0xFFu;
    ivec2 vpage     = ivec2(
        int(work.level_and_vpage & 0xFFu),
        int((work.level_and_vpage >> 8u) & 0xFFu)
    );
    uint phys_page  = work.physical_page & VSP_PTE_PAGE_MASK;

    VspClipmapLevel lvl = u_vsp.levels[level];

    // Convert toroidal vpage to linear clipmap page [0..15]
    ivec2 linear_page = (vpage - lvl.page_offset_and_pad.xy) & int(PRISM_VSP_VIRTUAL_PAGES_DIM - 1u);
    vec2 page_uv_min  = vec2(linear_page) / float(PRISM_VSP_VIRTUAL_PAGES_DIM);
    vec2 page_uv_max  = vec2(linear_page + ivec2(1)) / float(PRISM_VSP_VIRTUAL_PAGES_DIM);

    // Physical atlas pixel origin for this 128x128 page
    ivec2 phys_origin = ivec2(
        int(phys_page & (PRISM_VSP_ATLAS_PAGES_DIM - 1u)) * int(PRISM_VSP_PAGE_SIZE),
        int(phys_page / PRISM_VSP_ATLAS_PAGES_DIM) * int(PRISM_VSP_PAGE_SIZE)
    );

    // 1. Clear 128x128 physical page to far depth (1.0 -> 0x3F800000u)
    uint far_depth_uint = float_depth_to_ordered_uint(1.0);
    for (uint py = gl_LocalInvocationID.y; py < PRISM_VSP_PAGE_SIZE; py += 8u) {
        for (uint px = gl_LocalInvocationID.x; px < PRISM_VSP_PAGE_SIZE; px += 8u) {
            imageStore(
                u_physical_shadow_atlas,
                phys_origin + ivec2(int(px), int(py)),
                uvec4(far_depth_uint, 0u, 0u, 0u)
            );
        }
    }

    barrier();

    // 2. Iterate candidate meshlets and rasterize intersecting triangles into this 128x128 page
    uint total_candidates = u_vsp.total_meshlet_candidates;
    for (uint c_idx = 0u; c_idx < total_candidates; ++c_idx) {
        uvec2 item       = b_candidates[c_idx];
        uint instance_id = item.x;
        uint meshlet_id  = item.y;

        GpuInstance inst = b_instances[instance_id];
        if ((inst.flags & PRISM_INSTANCE_FLAG_VISIBLE) == 0u ||
            (inst.flags & PRISM_INSTANCE_FLAG_CAST_SHADOW) == 0u) {
            continue;
        }

        GpuMeshlet m = b_meshlets[inst.meshlet_offset + meshlet_id];
        if (m.triangle_count == 0u) {
            continue;
        }

        // Lane 0 tests meshlet bounding sphere against this virtual page's clipmap UV bounds
        if (local_tid == 0u) {
            mat4 M = inst.world_from_local;
            vec3 world_center = (M * vec4(m.bound_sphere.xyz, 1.0)).xyz;
            float max_scale = max(
                length(M[0].xyz),
                max(length(M[1].xyz), length(M[2].xyz))
            );
            float world_radius = m.bound_sphere.w * max_scale;

            vec4 clip_c = lvl.light_view_proj * vec4(world_center, 1.0);
            vec2 uv_c   = (clip_c.xy / max(clip_c.w, 1e-6)) * 0.5 + 0.5;
            float uv_r  = world_radius / max(lvl.world_origin_scale.w, 1e-4);

            bool overlaps = all(lessThanEqual(uv_c - vec2(uv_r), page_uv_max)) &&
                            all(greaterThanEqual(uv_c + vec2(uv_r), page_uv_min));
            s_meshlet_accepted = overlaps ? 1u : 0u;
        }

        barrier();

        if (s_meshlet_accepted == 0u) {
            continue;
        }

        // Transform up to 64 unique meshlet vertices into [0..128] page-pixel coordinates
        uint vert_count = min(m.vertex_count, PRISM_MESHLET_MAX_VERTICES);
        if (local_tid < vert_count) {
            uint global_vtx = inst.vertex_base + b_meshlet_vertices[m.vertex_offset + local_tid];
            vec4 local_pos  = vec4(b_positions[global_vtx].xyz, 1.0);
            vec4 world_pos  = inst.world_from_local * local_pos;
            vec4 light_clip = lvl.light_view_proj * world_pos;

            vec3 light_ndc = light_clip.xyz / max(light_clip.w, 1e-6);
            vec2 clip_uv   = light_ndc.xy * 0.5 + 0.5;
            vec2 page_uv   = (clip_uv - page_uv_min) * float(PRISM_VSP_VIRTUAL_PAGES_DIM);

            s_page_verts[local_tid] = vec3(
                page_uv * float(PRISM_VSP_PAGE_SIZE),
                clamp(light_ndc.z, 0.0, 1.0)
            );
        }

        barrier();

        // Each of the 64 threads rasterizes up to 2 micro-triangles (64 * 2 = 128 >= 124)
        uint prim_count = min(m.triangle_count, PRISM_MESHLET_MAX_TRIANGLES);
        for (uint tri_idx = local_tid; tri_idx < prim_count; tri_idx += 64u) {
            uint packed_tri = b_meshlet_triangles[m.triangle_offset + tri_idx];
            uint i0 = (packed_tri >>  0u) & 0xFFu;
            uint i1 = (packed_tri >>  8u) & 0xFFu;
            uint i2 = (packed_tri >> 16u) & 0xFFu;

            vec3 v0 = s_page_verts[i0];
            vec3 v1 = s_page_verts[i1];
            vec3 v2 = s_page_verts[i2];

            float area = edge_func(v0.xy, v1.xy, v2.xy);
            if (abs(area) < 1e-5) {
                continue;
            }
            float inv_area = 1.0 / area;

            ivec2 min_px = max(ivec2(floor(min(min(v0.xy, v1.xy), v2.xy))), ivec2(0));
            ivec2 max_px = min(ivec2(ceil(max(max(v0.xy, v1.xy), v2.xy))), ivec2(int(PRISM_VSP_PAGE_SIZE) - 1));

            if (min_px.x > max_px.x || min_px.y > max_px.y) {
                continue;
            }

            for (int y = min_px.y; y <= max_px.y; ++y) {
                for (int x = min_px.x; x <= max_px.x; ++x) {
                    vec2 p = vec2(float(x) + 0.5, float(y) + 0.5);
                    float w0 = edge_func(v1.xy, v2.xy, p) * inv_area;
                    float w1 = edge_func(v2.xy, v0.xy, p) * inv_area;
                    float w2 = edge_func(v0.xy, v1.xy, p) * inv_area;

                    if (w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0) {
                        float z = v0.z * w0 + v1.z * w1 + v2.z * w2;
                        uint depth_u = float_depth_to_ordered_uint(z);
                        imageAtomicMin(
                            u_physical_shadow_atlas,
                            phys_origin + ivec2(x, y),
                            depth_u
                        );
                    }
                }
            }
        }

        barrier();
    }
}
