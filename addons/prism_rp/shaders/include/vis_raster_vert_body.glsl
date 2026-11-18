// Body of vis_raster.vert.glsl (opaque list) and vis_raster_masked.vert.glsl (PRISM_VIS_MASKED:
// alpha-tested list, also outputs UVs). The entry file declares #version 460 and
// GL_GOOGLE_include_directive.
//
// Vertex-pulling alternative to the mesh shader. Draw the list's VkDrawIndirectCommands from
// vis_cull_meshlets non-indexed (vkCmdDrawIndirectCount), with no vertex buffers:
//   gl_InstanceIndex = firstInstance = the list entry's index in b_visible_meshlets
//   gl_VertexIndex   = 0 .. 3 * triangle_count - 1 (each triangle gets its own 3 vertices)
// Pipeline state: cull back faces with front face = CLOCKWISE (Godot's front faces are clockwise
// on screen with its Y-flipped projection). Mirrored instances (negative determinant) need the
// opposite front face: draw them in a separate pass, or mark them two-sided and disable culling.
// Pair with vis_raster.frag.glsl / vis_raster_masked.frag.glsl.

#include "visibility_common.glsl"
#ifdef PRISM_VIS_MASKED
#include "vis_shading_common.glsl"
#endif

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

#ifdef PRISM_VIS_MASKED
layout(set = 0, binding = 7, std430) readonly buffer GlobalVertexAttrBuffer {
    GpuVertexAttributes b_attributes[];
};
#endif

// (instance_id, meshlet_local_id, primitive_id), constant over the triangle
layout(location = 0) flat out uvec3 v_prim_meta;
#ifdef PRISM_VIS_MASKED
layout(location = 1) out vec2 v_uv;
#endif

void main() {
    uint vis_index = uint(gl_InstanceIndex);
    uint vert_id   = uint(gl_VertexIndex);
    uint prim_id   = vert_id / 3u;
    uint corner    = vert_id % 3u;

    VisibleMeshlet vis = b_visible_meshlets[vis_index];
    GpuInstance inst   = b_instances[vis.instance_id];
    GpuMeshlet m       = b_meshlets[inst.meshlet_offset + vis.meshlet_local_id];

    uint packed_tri      = b_meshlet_triangles[m.triangle_offset + prim_id];
    uint local_vtx_idx   = (packed_tri >> (corner * 8u)) & 0xFFu;
    uint global_vert_idx = inst.vertex_base + b_meshlet_vertices[m.vertex_offset + local_vtx_idx];

    gl_Position = u_cull.view_proj * (inst.world_from_local * vec4(b_positions[global_vert_idx].xyz, 1.0));
    v_prim_meta = uvec3(vis.instance_id, vis.meshlet_local_id, prim_id);
#ifdef PRISM_VIS_MASKED
    v_uv = b_attributes[global_vert_idx].uv.xy;
#endif
}
