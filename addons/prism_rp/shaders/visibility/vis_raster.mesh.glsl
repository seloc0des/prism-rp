#version 460
#extension GL_EXT_mesh_shader : require

#include "../include/visibility_common.glsl"

// ============================================================================
// Prism RP — Visibility Buffer Mesh Shader (VK_EXT_mesh_shader)
// One workgroup (128 threads) processes one VisibleMeshlet (<=64v / <=124t).
// Emits clip-space positions + flat per-primitive IDs with zero attribute bloat.
// ============================================================================

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
layout(triangles) out;
layout(max_vertices = 64, max_primitives = 124) out;

layout(set = 0, binding = 0, std140) uniform CullUniformBlock {
    VisibilityCullUBO u_cull;
};

layout(set = 0, binding = 1, std430) readonly buffer MeshletBuffer {
    GpuMeshlet b_meshlets[];
};

layout(set = 0, binding = 2, std430) readonly buffer InstanceBuffer {
    GpuInstance b_instances[];
};

layout(set = 0, binding = 3, std430) readonly buffer GlobalVertexPosBuffer {
    vec4 b_positions[];
};

layout(set = 0, binding = 4, std430) readonly buffer MeshletVertexIndicesBuffer {
    uint b_meshlet_vertices[];
};

layout(set = 0, binding = 5, std430) readonly buffer MeshletTrianglesBuffer {
    uint b_meshlet_triangles[];
};

layout(set = 0, binding = 6, std430) readonly buffer VisibleMeshletBuffer {
    VisibleMeshlet b_visible_meshlets[];
};

// Per-primitive flat outputs consumed by vis_raster.frag.glsl
layout(location = 0) perprimitiveEXT flat out uvec4 out_prim_meta[];

shared vec4 s_clip_pos[PRISM_MESHLET_MAX_VERTICES];

void main() {
    uint vis_slot = gl_WorkGroupID.x;
    uint tid      = gl_LocalInvocationID.x;

    VisibleMeshlet vis = b_visible_meshlets[vis_slot];
    GpuInstance inst   = b_instances[vis.instance_id];
    GpuMeshlet m       = b_meshlets[inst.meshlet_offset + vis.meshlet_local_id];

    uint vert_count = min(m.vertex_count, PRISM_MESHLET_MAX_VERTICES);
    uint prim_count = min(m.triangle_count, PRISM_MESHLET_MAX_TRIANGLES);

    SetMeshOutputsEXT(vert_count, prim_count);

    // 1. Transform up to 64 unique meshlet vertices into clip space
    if (tid < vert_count) {
        uint global_vert_idx = inst.vertex_base + b_meshlet_vertices[m.vertex_offset + tid];
        vec4 local_pos = vec4(b_positions[global_vert_idx].xyz, 1.0);
        vec4 world_pos = inst.world_from_local * local_pos;
        vec4 clip_pos  = u_cull.view_proj * world_pos;

        s_clip_pos[tid] = clip_pos;
        gl_MeshVerticesEXT[tid].gl_Position = clip_pos;
    }

    barrier();

    // 2. Emit up to 124 micro-triangles + degenerate/sub-pixel primitive culling
    if (tid < prim_count) {
        uint packed_tri = b_meshlet_triangles[m.triangle_offset + tid];
        uint i0 = (packed_tri >>  0u) & 0xFFu;
        uint i1 = (packed_tri >>  8u) & 0xFFu;
        uint i2 = (packed_tri >> 16u) & 0xFFu;

        gl_PrimitiveTriangleIndicesEXT[tid] = uvec3(i0, i1, i2);

        // Quick 2D screen-space backface + zero-area cull when all 3 verts are in front of near plane
        vec4 p0 = s_clip_pos[i0];
        vec4 p1 = s_clip_pos[i1];
        vec4 p2 = s_clip_pos[i2];

        bool culled = false;
        if (p0.w > 0.0 && p1.w > 0.0 && p2.w > 0.0) {
            vec2 ndc0 = p0.xy / p0.w;
            vec2 ndc1 = p1.xy / p1.w;
            vec2 ndc2 = p2.xy / p2.w;

            // Godot clockwise front-face winding in Vulkan top-down NDC has negative signed area
            float twice_area = (ndc1.x - ndc0.x) * (ndc2.y - ndc0.y)
                             - (ndc1.y - ndc0.y) * (ndc2.x - ndc0.x);

            bool two_sided = (inst.flags & PRISM_INSTANCE_FLAG_TWO_SIDED) != 0u;
            if ((!two_sided && twice_area >= 0.0) || abs(twice_area) < 1e-11) {
                culled = true;
            }
        }

        gl_MeshPrimitivesEXT[tid].gl_CullPrimitiveEXT = culled;

        out_prim_meta[tid] = uvec4(
            vis.instance_id,
            vis.meshlet_local_id,
            tid,
            inst.flags
        );
    }
}
