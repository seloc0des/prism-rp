#version 460
#extension GL_GOOGLE_include_directive : require

#include "visibility_common.glsl"

// ============================================================================
// Prism RP — Visibility Buffer Programmable Vertex-Pulling Shader (Fallback)
// Used when VK_EXT_mesh_shader is unavailable (e.g. macOS / Tier 3 GPUs).
// Driven by VkDrawIndexedIndirectCommand emitted by vis_cull_meshlets.comp.glsl:
//   - gl_InstanceIndex = slot in b_visible_meshlets[]
//   - gl_VertexIndex   = [0 .. triangle_count * 3 - 1]
// ============================================================================

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

// Matches vis_raster.mesh.glsl flat output interface so both share vis_raster.frag.glsl
layout(location = 0) flat out uvec4 out_prim_meta;

void main() {
    uint vis_slot = uint(gl_InstanceIndex);
    uint vert_id  = uint(gl_VertexIndex);

    uint prim_id = vert_id / 3u;
    uint corner  = vert_id % 3u;

    VisibleMeshlet vis = b_visible_meshlets[vis_slot];
    GpuInstance inst   = b_instances[vis.instance_id];
    GpuMeshlet m       = b_meshlets[inst.meshlet_offset + vis.meshlet_local_id];

    // Unpack local micro-vertex index [0..63] from packed u8x3 triangle word
    uint packed_tri    = b_meshlet_triangles[m.triangle_offset + prim_id];
    uint local_vtx_idx = (packed_tri >> (corner * 8u)) & 0xFFu;

    uint global_vert_idx = inst.vertex_base + b_meshlet_vertices[m.vertex_offset + local_vtx_idx];
    vec4 local_pos = vec4(b_positions[global_vert_idx].xyz, 1.0);
    vec4 world_pos = inst.world_from_local * local_pos;

    gl_Position = u_cull.view_proj * world_pos;

    // Provoking vertex writes flat metadata for the triangle
    out_prim_meta = uvec4(
        vis.instance_id,
        vis.meshlet_local_id,
        prim_id,
        inst.flags
    );
}
