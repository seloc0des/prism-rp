// Body of vis_raster.mesh.glsl (opaque list) and vis_raster_masked.mesh.glsl (PRISM_VIS_MASKED:
// alpha-tested list, also outputs UVs). The entry file declares #version 460,
// GL_GOOGLE_include_directive and GL_EXT_mesh_shader.
//
// One workgroup (128 threads) per visible meshlet (<= 64 vertices / 124 triangles). Dispatch with
// vkCmdDrawMeshTasksIndirectEXT on the list's command from vis_cull_meshlets. Pair with
// vis_raster_ms.frag.glsl / vis_raster_ms_masked.frag.glsl.

#include "visibility_common.glsl"
#ifdef PRISM_VIS_MASKED
#include "vis_shading_common.glsl"
#endif

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

#ifdef PRISM_VIS_MASKED
layout(set = 0, binding = 7, std430) readonly buffer GlobalVertexAttrBuffer {
    GpuVertexAttributes b_attributes[];
};
#endif

// Per-primitive (instance_id, meshlet_local_id, primitive_id) for the fragment shader
layout(location = 0) perprimitiveEXT flat out uvec3 v_prim_meta[];
#ifdef PRISM_VIS_MASKED
layout(location = 1) out vec2 v_uv[];
#endif

shared vec4 s_clip_pos[PRISM_MESHLET_MAX_VERTICES];

void main() {
#ifdef PRISM_VIS_MASKED
    uint vis_index = u_cull.max_visible_meshlets + gl_WorkGroupID.x;
#else
    uint vis_index = gl_WorkGroupID.x;
#endif
    uint tid = gl_LocalInvocationID.x;

    VisibleMeshlet vis = b_visible_meshlets[vis_index];
    GpuInstance inst   = b_instances[vis.instance_id];
    GpuMeshlet m       = b_meshlets[inst.meshlet_offset + vis.meshlet_local_id];

    uint vert_count = min(m.vertex_count, PRISM_MESHLET_MAX_VERTICES);
    uint prim_count = min(m.triangle_count, PRISM_MESHLET_MAX_TRIANGLES);
    SetMeshOutputsEXT(vert_count, prim_count);

    // 1. Transform the meshlet's unique vertices to clip space
    if (tid < vert_count) {
        uint global_vert_idx = inst.vertex_base + b_meshlet_vertices[m.vertex_offset + tid];
        vec4 clip_pos = u_cull.view_proj * (inst.world_from_local * vec4(b_positions[global_vert_idx].xyz, 1.0));
        s_clip_pos[tid] = clip_pos;
        gl_MeshVerticesEXT[tid].gl_Position = clip_pos;
#ifdef PRISM_VIS_MASKED
        v_uv[tid] = b_attributes[global_vert_idx].uv.xy;
#endif
    }

    barrier();

    // 2. Emit triangles; drop back faces and zero-area ones early
    if (tid < prim_count) {
        uint packed_tri = b_meshlet_triangles[m.triangle_offset + tid];
        uvec3 idx = uvec3(packed_tri & 0xFFu, (packed_tri >> 8u) & 0xFFu, (packed_tri >> 16u) & 0xFFu);
        gl_PrimitiveTriangleIndicesEXT[tid] = idx;

        vec4 p0 = s_clip_pos[idx.x];
        vec4 p1 = s_clip_pos[idx.y];
        vec4 p2 = s_clip_pos[idx.z];

        // Only when all corners are in front of the camera; otherwise the hardware clips it.
        bool culled = false;
        if (p0.w > 0.0 && p1.w > 0.0 && p2.w > 0.0) {
            bool two_sided = (inst.flags & PRISM_INSTANCE_FLAG_TWO_SIDED) != 0u;
            bool mirrored  = prism_det3(mat3(inst.world_from_local)) < 0.0;
            culled = vis_triangle_culled(p0.xy / p0.w, p1.xy / p1.w, p2.xy / p2.w, two_sided, mirrored);
        }
        gl_MeshPrimitivesEXT[tid].gl_CullPrimitiveEXT = culled;
        v_prim_meta[tid] = uvec3(vis.instance_id, vis.meshlet_local_id, tid);
    }
}
