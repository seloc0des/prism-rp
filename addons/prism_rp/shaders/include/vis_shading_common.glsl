#ifndef PRISM_VIS_SHADING_COMMON_GLSL
#define PRISM_VIS_SHADING_COMMON_GLSL

#include "visibility_common.glsl"

// ============================================================================
// Prism RP — Phase 4 Deferred Visibility Shading & 8x8 Tile Classification
// Target: GLSL 4.60 / Vulkan 1.3
// ============================================================================

const uint PRISM_VIS_TILE_SIZE      = 8u;
const uint PRISM_MAX_TILE_MATERIALS = 256u; // Supported material IDs per classification pass
const uint PRISM_INVALID_TEX_INDEX  = 0xFFFFFFFFu;

// Material flags
const uint PRISM_MAT_FLAG_DOUBLE_SIDED   = 1u << 0;
const uint PRISM_MAT_FLAG_ALPHA_TEST     = 1u << 1;
const uint PRISM_MAT_FLAG_UNSHADED       = 1u << 2;
const uint PRISM_MAT_FLAG_USE_ORM_PACKED = 1u << 3; // R=Occlusion, G=Roughness, B=Metallic

// ----------------------------------------------------------------------------
// Per-vertex shading attributes (48 bytes, std430)
// Indexed in parallel with GlobalVertexPosBuffer (b_positions[])
// ----------------------------------------------------------------------------
struct GpuVertexAttributes {
    vec4 normal_and_uv_x;   // xyz = object-space unit normal, w = uv.x
    vec4 tangent_and_uv_y;  // xyz = object-space unit tangent, w = sign (+1/-1)
    vec4 uv_and_color;      // xy = uv0, zw = packed/unpacked vertex color rg (or uv1)
};

// ----------------------------------------------------------------------------
// PBR Material Descriptor (80 bytes, std430)
// ----------------------------------------------------------------------------
struct GpuPbrMaterial {
    vec4  base_color_factor;      // rgba
    vec4  emissive_and_roughness; // rgb = linear ACEScg emissive, a = roughness_factor
    vec4  metallic_ao_alpha;      // x = metallic_factor, y = ao_strength, z = alpha_cutoff, w = normal_scale
    uvec4 tex_indices;            // x = albedo, y = normal, z = orm/roughness-metallic, w = emissive
    uint  flags;
    uint  _pad0;
    uint  _pad1;
    uint  _pad2;
};

// ----------------------------------------------------------------------------
// Compact 8x8 Tile + Material Work Item emitted by vis_tile_classify.comp.glsl
// ----------------------------------------------------------------------------
struct MaterialTileWorkItem {
    uint packed_tile_xy; // [15:0] = tile_x, [31:16] = tile_y
    uint material_id;    // Material ID active in this 8x8 tile
};

// ----------------------------------------------------------------------------
// Standard Vulkan Compute Indirect Dispatch Command
// ----------------------------------------------------------------------------
struct VkDispatchIndirectCommand {
    uint x;
    uint y;
    uint z;
};

// ----------------------------------------------------------------------------
// Deferred Visibility Shading Uniforms (std140)
// ----------------------------------------------------------------------------
struct VisibilityShadeUBO {
    mat4  view_proj;
    mat4  prev_view_proj;
    mat4  inv_view_proj;
    vec4  camera_pos_and_exposure; // xyz = world camera pos, w = exposure
    vec4  sun_dir_and_intensity;   // xyz = unit vector toward sun, w = illuminance
    vec4  sun_color_and_ambient;   // rgb = ACEScg sun color, a = sky ambient fallback
    uvec4 viewport_and_limits;     // xy = render resolution, z = tile_count_x, w = max_work_items
};

// ----------------------------------------------------------------------------
// Barycentric & Screen-Space Ray-Differential UV Gradient Reconstruction
// Computes exact screen-pixel derivatives (ddx_uv, ddy_uv) without 2x2 quad
// helper lanes so textureGrad() selects crisp anisotropic mips across edges.
// ----------------------------------------------------------------------------
struct ShadedVertexSample {
    vec3  world_pos;
    vec3  prev_world_pos;
    vec3  world_normal;
    vec4  world_tangent; // xyz = tangent, w = bitangent sign
    vec2  uv;
    vec2  ddx_uv;
    vec2  ddy_uv;
    vec3  ddx_world_pos;
    vec3  ddy_world_pos;
};

void compute_barycentric_gradients(
    vec4 clip0,
    vec4 clip1,
    vec4 clip2,
    vec2 pixel_ndc,
    vec2 inv_viewport,
    out vec3 bary,
    out vec3 ddx_bary,
    out vec3 ddy_bary
) {
    // Inverse clip W for perspective-correct interpolation
    vec3 inv_w = 1.0 / max(vec3(clip0.w, clip1.w, clip2.w), vec3(1e-6));
    vec2 ndc0  = clip0.xy * inv_w.x;
    vec2 ndc1  = clip1.xy * inv_w.y;
    vec2 ndc2  = clip2.xy * inv_w.z;

    // 2D edge functions in NDC space
    float det = (ndc1.x - ndc0.x) * (ndc2.y - ndc0.y)
              - (ndc1.y - ndc0.y) * (ndc2.x - ndc0.x);
    float inv_det = (abs(det) > 1e-12) ? (1.0 / det) : 0.0;

    // Screen-space derivatives of unnormalized NDC barycentrics (1 pixel = 2 * inv_viewport in NDC)
    vec2 ndc_step = 2.0 * inv_viewport;

    vec3 ddx_a = vec3(
        (ndc1.y - ndc2.y) * inv_det * ndc_step.x,
        (ndc2.y - ndc0.y) * inv_det * ndc_step.x,
        (ndc0.y - ndc1.y) * inv_det * ndc_step.x
    );
    vec3 ddy_a = vec3(
        (ndc2.x - ndc1.x) * inv_det * ndc_step.y,
        (ndc0.x - ndc2.x) * inv_det * ndc_step.y,
        (ndc1.x - ndc0.x) * inv_det * ndc_step.y
    );

    // Evaluate unnormalized barycentrics at pixel_ndc
    vec3 a = vec3(
        ((ndc1.x - pixel_ndc.x) * (ndc2.y - pixel_ndc.y) - (ndc1.y - pixel_ndc.y) * (ndc2.x - pixel_ndc.x)) * inv_det,
        ((ndc2.x - pixel_ndc.x) * (ndc0.y - pixel_ndc.y) - (ndc2.y - pixel_ndc.y) * (ndc0.x - pixel_ndc.x)) * inv_det,
        ((ndc0.x - pixel_ndc.x) * (ndc1.y - pixel_ndc.y) - (ndc0.y - pixel_ndc.y) * (ndc1.x - pixel_ndc.x)) * inv_det
    );

    // Perspective-correct weights w_i = a_i * inv_w_i
    vec3 w     = a * inv_w;
    vec3 ddx_w = ddx_a * inv_w;
    vec3 ddy_w = ddy_a * inv_w;

    float sum_w     = max(w.x + w.y + w.z, 1e-8);
    float inv_sum_w = 1.0 / sum_w;
    float ddx_sum_w = ddx_w.x + ddx_w.y + ddx_w.z;
    float ddy_sum_w = ddy_w.x + ddy_w.y + ddy_w.z;

    bary     = w * inv_sum_w;
    ddx_bary = (ddx_w * sum_w - w * ddx_sum_w) * (inv_sum_w * inv_sum_w);
    ddy_bary = (ddy_w * sum_w - w * ddy_sum_w) * (inv_sum_w * inv_sum_w);
}

// Octahedral unit-vector encoding (matches Lumina/Chronos G-buffer normal convention)
vec2 encode_octahedral_normal(vec3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z) + 1e-8);
    if (n.z < 0.0) {
        vec2 sign_not_zero = vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
        n.xy = (1.0 - abs(n.yx)) * sign_not_zero;
    }
    return n.xy * 0.5 + 0.5;
}

#endif // PRISM_VIS_SHADING_COMMON_GLSL
