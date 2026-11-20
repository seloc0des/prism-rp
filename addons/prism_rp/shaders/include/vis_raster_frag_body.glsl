// Body of the four visibility-buffer fragment shaders. The entry file declares #version 460 and
// GL_GOOGLE_include_directive, plus:
//   PRISM_VIS_MESH_PATH  inputs come from vis_raster(_masked).mesh.glsl as per-primitive values
//                        (needs GL_EXT_mesh_shader); otherwise from vis_raster(_masked).vert.glsl.
//   PRISM_VIS_MASKED     alpha test against the material's albedo alpha and cutoff
//                        (needs GL_EXT_nonuniform_qualifier).
//
// Writes the 64-bit payload (visibility_common.glsl) to an R32G32_UINT target cleared to
// PRISM_VIS_PAYLOAD_EMPTY. No barycentrics: vis_shade_materials rebuilds them.

#include "visibility_common.glsl"
#ifdef PRISM_VIS_MASKED
#include "vis_shading_common.glsl"

layout(set = 0, binding = 2, std430) readonly buffer InstanceBuffer {
    GpuInstance b_instances[];
};

layout(set = 0, binding = 8, std430) readonly buffer MaterialBuffer {
    GpuPbrMaterial b_materials[];
};

layout(set = 0, binding = 9) uniform sampler2D u_textures[PRISM_MAX_MATERIAL_TEXTURES];
#endif

#ifdef PRISM_VIS_MESH_PATH
layout(location = 0) perprimitiveEXT flat in uvec3 v_prim_meta;
#else
layout(location = 0) flat in uvec3 v_prim_meta;
#endif
#ifdef PRISM_VIS_MASKED
layout(location = 1) in vec2 v_uv;
#endif

layout(location = 0) out uvec2 out_vis_payload;

#ifndef PRISM_VIS_MASKED
// Opaque: depth-test before the shader runs so hidden fragments never execute it. Not for the
// masked variant, where discard has to be able to stop the depth write.
layout(early_fragment_tests) in;
#endif

void main() {
#ifdef PRISM_VIS_MASKED
    GpuPbrMaterial mat = b_materials[b_instances[v_prim_meta.x].material_id];
    float alpha = mat.base_color_factor.a;
    if (mat.tex_indices.x != PRISM_INVALID_TEX_INDEX) {
        alpha *= texture(u_textures[nonuniformEXT(mat.tex_indices.x)], v_uv).a;
    }
    if (alpha < mat.metallic_ao_alpha.z) {
        discard;
    }
    const bool alpha_tested = true;
#else
    const bool alpha_tested = false;
#endif
    out_vis_payload = pack_visibility_payload(v_prim_meta.x, v_prim_meta.y, v_prim_meta.z, alpha_tested);
}
