#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require

#include "vis_shading_common.glsl"
#include "vsp_common.glsl"

// ============================================================================
// Prism RP — Virtual Shadow Pages compute meshlet rasterizer
// One workgroup (64 threads) per raster item from vsp_cull_meshlets:
// vkCmdDispatchIndirect on VspFrameState.raster_dispatch. Transforms the meshlet's vertices into
// the item's 128x128 page and rasterizes its triangles (both faces) with imageAtomicMin of the
// light depth. vsp_clear_pages has already cleared the page. Alpha-tested instances test the
// material's albedo alpha per texel.
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

layout(set = 0, binding = 3, std430) readonly buffer GlobalVertexPosBuffer {
    vec4 b_positions[];
};

layout(set = 0, binding = 4, std430) readonly buffer MeshletVertexIndicesBuffer {
    uint b_meshlet_vertices[];
};

layout(set = 0, binding = 5, std430) readonly buffer MeshletTrianglesBuffer {
    uint b_meshlet_triangles[];
};

layout(set = 0, binding = 6, std430) readonly buffer RasterItemBuffer {
    VspRasterItem b_raster_items[];
};

layout(set = 0, binding = 7, std430) readonly buffer VspFrameStateBuffer {
    VspFrameState b_state;
};

// Physical atlas, 4096x4096 R32_UINT
layout(set = 0, binding = 8, r32ui) uniform uimage2D u_physical_shadow_atlas;

// Alpha test inputs (only read for PRISM_INSTANCE_FLAG_ALPHA_TEST instances)
layout(set = 0, binding = 9, std430) readonly buffer GlobalVertexAttrBuffer {
    GpuVertexAttributes b_attributes[];
};

layout(set = 0, binding = 10, std430) readonly buffer MaterialBuffer {
    GpuPbrMaterial b_materials[];
};

layout(set = 0, binding = 11) uniform sampler2D u_textures[];

// Meshlet vertices in page texel coordinates (xy in [0, 128] inside the page) + light depth (z)
shared vec3 s_page_verts[PRISM_MESHLET_MAX_VERTICES];
shared vec2 s_uvs[PRISM_MESHLET_MAX_VERTICES];

float edge_func(vec2 a, vec2 b, vec2 c) {
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

void main() {
    uint item_idx = gl_WorkGroupID.x;
    if (item_idx >= min(b_state.raster_item_count, u_vsp.max_raster_items)) {
        return;
    }
    uint local_tid = gl_LocalInvocationIndex;

    VspRasterItem item = b_raster_items[item_idx];
    uint level;
    ivec2 vpage;
    vsp_unpack_level_vpage(item.level_and_vpage, level, vpage);
    VspClipmapLevel lvl = u_vsp.levels[level];
    vec2 page_uv_min = vec2(toroidal_to_linear_vpage(vpage, lvl.page_offset_and_pad.xy)) / float(PRISM_VSP_VIRTUAL_PAGES_DIM);
    ivec2 phys_origin = vsp_physical_page_origin(item.physical_page & VSP_PTE_PAGE_MASK);

    GpuInstance inst = b_instances[item.instance_id];
    GpuMeshlet m     = b_meshlets[inst.meshlet_offset + item.meshlet_local_id];
    bool masked      = (inst.flags & PRISM_INSTANCE_FLAG_ALPHA_TEST) != 0u;

    // 1. Vertices -> page texel coordinates
    uint vert_count = min(m.vertex_count, PRISM_MESHLET_MAX_VERTICES);
    if (local_tid < vert_count) {
        uint global_vtx = inst.vertex_base + b_meshlet_vertices[m.vertex_offset + local_tid];
        vec4 light_clip = lvl.light_view_proj * (inst.world_from_local * vec4(b_positions[global_vtx].xyz, 1.0));
        vec2 clip_uv    = light_clip.xy / light_clip.w * 0.5 + 0.5;
        s_page_verts[local_tid] = vec3((clip_uv - page_uv_min) * float(PRISM_VSP_VIRTUAL_RES),
                                       clamp(light_clip.z / light_clip.w, 0.0, 1.0));
        if (masked) {
            s_uvs[local_tid] = b_attributes[global_vtx].uv.xy;
        }
    }
    barrier();

    GpuPbrMaterial mat;
    bool alpha_textured = false;
    if (masked) {
        mat = b_materials[inst.material_id];
        alpha_textured = mat.tex_indices.x != PRISM_INVALID_TEX_INDEX;
    }

    // 2. Each thread rasterizes up to 2 of the <= 124 triangles
    uint prim_count = min(m.triangle_count, PRISM_MESHLET_MAX_TRIANGLES);
    for (uint tri_idx = local_tid; tri_idx < prim_count; tri_idx += 64u) {
        uint packed_tri = b_meshlet_triangles[m.triangle_offset + tri_idx];
        uvec3 idx = uvec3(packed_tri & 0xFFu, (packed_tri >> 8u) & 0xFFu, (packed_tri >> 16u) & 0xFFu);
        vec3 v0 = s_page_verts[idx.x];
        vec3 v1 = s_page_verts[idx.y];
        vec3 v2 = s_page_verts[idx.z];

        float area = edge_func(v0.xy, v1.xy, v2.xy);
        if (abs(area) < 1e-5) {
            continue;
        }
        float inv_area = 1.0 / area;

        ivec2 min_px = max(ivec2(floor(min(min(v0.xy, v1.xy), v2.xy))), ivec2(0));
        ivec2 max_px = min(ivec2(ceil(max(max(v0.xy, v1.xy), v2.xy))), ivec2(int(PRISM_VSP_PAGE_SIZE) - 1));
        if (any(greaterThan(min_px, max_px))) {
            continue;
        }

        // Alpha test setup: the light projection is orthographic, so UVs interpolate linearly in
        // page texels and their gradients are constant over the triangle.
        vec2 uv0 = vec2(0.0), uv1 = vec2(0.0), uv2 = vec2(0.0), duv_dx = vec2(0.0), duv_dy = vec2(0.0);
        if (alpha_textured) {
            uv0 = s_uvs[idx.x];
            uv1 = s_uvs[idx.y];
            uv2 = s_uvs[idx.z];
            vec3 dw_dx = vec3(v2.y - v1.y, v0.y - v2.y, v1.y - v0.y) * inv_area;
            vec3 dw_dy = vec3(v1.x - v2.x, v2.x - v0.x, v0.x - v1.x) * inv_area;
            duv_dx = uv0 * dw_dx.x + uv1 * dw_dx.y + uv2 * dw_dx.z;
            duv_dy = uv0 * dw_dy.x + uv1 * dw_dy.y + uv2 * dw_dy.z;
        }

        for (int y = min_px.y; y <= max_px.y; ++y) {
            for (int x = min_px.x; x <= max_px.x; ++x) {
                vec2 p = vec2(float(x) + 0.5, float(y) + 0.5);
                vec3 w = vec3(edge_func(v1.xy, v2.xy, p), edge_func(v2.xy, v0.xy, p), edge_func(v0.xy, v1.xy, p)) * inv_area;
                if (any(lessThan(w, vec3(0.0)))) {
                    continue;
                }
                if (masked) {
                    float alpha = mat.base_color_factor.a;
                    if (alpha_textured) {
                        vec2 uv = uv0 * w.x + uv1 * w.y + uv2 * w.z;
                        alpha *= textureGrad(u_textures[nonuniformEXT(mat.tex_indices.x)], uv, duv_dx, duv_dy).a;
                    }
                    if (alpha < mat.metallic_ao_alpha.z) {
                        continue;
                    }
                }
                float z = v0.z * w.x + v1.z * w.y + v2.z * w.z;
                imageAtomicMin(u_physical_shadow_atlas, phys_origin + ivec2(x, y), float_depth_to_ordered_uint(z));
            }
        }
    }
}
