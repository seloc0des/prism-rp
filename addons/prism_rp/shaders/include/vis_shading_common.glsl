#ifndef PRISM_VIS_SHADING_COMMON_GLSL
#define PRISM_VIS_SHADING_COMMON_GLSL

#include "visibility_common.glsl"

// ============================================================================
// Prism RP — Track B deferred visibility shading & 8x8 tile classification
// Target: GLSL 4.60 / Vulkan 1.3
//
// Colour: every input colour (material factors, emissive, sun) is linear Rec.709, albedo / emissive
// textures are sampled through sRGB views (so they read as linear Rec.709), and the HDR output is
// linear Rec.709: what Godot's buffers hold and what Chronos expects.
// ============================================================================

const uint PRISM_VIS_TILE_SIZE        = 8u;
const uint PRISM_MAX_TILE_MATERIALS   = 256u;  // classification bins; higher IDs share the last bin
const uint PRISM_INVALID_TEX_INDEX    = 0xFFFFFFFFu;
const uint PRISM_MAX_MATERIAL_TEXTURES = 256u; // size of u_textures[]; the host binds every slot, unused ones to a placeholder
const uint PRISM_SHADE_DISPATCH_WIDTH = 256u;  // shade workgroups per row of the 2D indirect dispatch

// Material flags
const uint PRISM_MAT_FLAG_DOUBLE_SIDED   = 1u << 0;
const uint PRISM_MAT_FLAG_ALPHA_TEST     = 1u << 1; // set PRISM_INSTANCE_FLAG_ALPHA_TEST on its instances
const uint PRISM_MAT_FLAG_UNSHADED       = 1u << 2;
const uint PRISM_MAT_FLAG_USE_ORM_PACKED = 1u << 3; // R=Occlusion, G=Roughness, B=Metallic

// ----------------------------------------------------------------------------
// Per-vertex shading attributes (48 bytes, std430), parallel to the position buffer
// ----------------------------------------------------------------------------
struct GpuVertexAttributes {
    vec4 normal;   // xyz = object-space unit normal, w = unused
    vec4 tangent;  // xyz = object-space unit tangent, w = bitangent sign (+1/-1), Godot's convention
    vec4 uv;       // xy = UV, zw = UV2
};

// ----------------------------------------------------------------------------
// PBR material descriptor (80 bytes, std430)
// ----------------------------------------------------------------------------
struct GpuPbrMaterial {
    vec4  base_color_factor;      // rgba, linear Rec.709
    vec4  emissive_and_roughness; // rgb = linear Rec.709 emissive, a = roughness_factor
    vec4  metallic_ao_alpha;      // x = metallic_factor, y = ao_strength, z = alpha_cutoff, w = normal_scale
    uvec4 tex_indices;            // x = albedo, y = normal, z = orm/roughness-metallic, w = emissive
    uint  flags;
    uint  _pad0;
    uint  _pad1;
    uint  _pad2;
};

// ----------------------------------------------------------------------------
// Compact 8x8 tile + material work item emitted by vis_tile_classify.comp.glsl
// ----------------------------------------------------------------------------
struct MaterialTileWorkItem {
    uint packed_tile_xy; // [15:0] = tile_x, [31:16] = tile_y
    uint material_id;    // classification bin (min(material_id, PRISM_MAX_TILE_MATERIALS - 1))
};

// ----------------------------------------------------------------------------
// Shade dispatch: VkDispatchIndirectCommand followed by the work-item count. The dispatch is
// (PRISM_SHADE_DISPATCH_WIDTH, rows, 1) so it never exceeds maxComputeWorkGroupCount[0].
// Reset each frame to {PRISM_SHADE_DISPATCH_WIDTH, 0, 1, 0}.
// ----------------------------------------------------------------------------
struct ShadeDispatch {
    uint group_count_x;
    uint group_count_y;
    uint group_count_z;
    uint work_item_count;   // may exceed the work-item buffer; clamp with viewport_and_limits.w
};

// ----------------------------------------------------------------------------
// Deferred visibility shading uniforms (std140)
// ----------------------------------------------------------------------------
struct VisibilityShadeUBO {
    mat4  view_proj;
    mat4  prev_view_proj;
    mat4  inv_view_proj;
    vec4  camera_pos_and_exposure; // xyz = world camera pos, w = exposure (unused by these passes)
    vec4  sun_dir_and_intensity;   // xyz = unit vector toward sun, w = illuminance
    vec4  sun_color_and_ambient;   // rgb = linear Rec.709 sun colour, a = sky ambient fallback
    uvec4 viewport_and_limits;     // xy = render resolution, z = tile_count_x, w = max_work_items
};

// ----------------------------------------------------------------------------
// Barycentrics & screen-space derivatives
// ----------------------------------------------------------------------------

// Perspective-correct barycentrics of the pixel at `pixel_ndc` and their per-pixel screen-space
// derivatives (for textureGrad), from the triangle's clip-space corners. Works in homogeneous
// coordinates (Olano & Greer, "Triangle Scan Conversion using 2D Homogeneous Coordinates"), so it
// stays exact when corners lie behind the camera (w <= 0), as large ground triangles often do.
// Returns false for a triangle seen edge-on.
bool compute_barycentric_gradients(
    vec4 clip0,
    vec4 clip1,
    vec4 clip2,
    vec2 pixel_ndc,
    vec2 inv_viewport,
    out vec3 bary,
    out vec3 ddx_bary,
    out vec3 ddy_bary
) {
    vec3 v0 = clip0.xyw;
    vec3 v1 = clip1.xyw;
    vec3 v2 = clip2.xyw;
    // Rows of the adjugate of [v0 v1 v2]: e_i = dot(c_i, (ndc, 1)) is proportional to barycentric i.
    vec3 c0 = cross(v1, v2);
    vec3 c1 = cross(v2, v0);
    vec3 c2 = cross(v0, v1);

    vec3 p = vec3(pixel_ndc, 1.0);
    vec3 e = vec3(dot(c0, p), dot(c1, p), dot(c2, p));
    float sum = e.x + e.y + e.z;
    // det [v0 v1 v2] is zero when the triangle's plane passes through the eye (edge-on); sum is
    // zero when this pixel's ray runs parallel to the plane.
    float det = dot(v0, c0);
    bool edge_on = !(abs(det) > 1e-6 * length(v0) * length(v1) * length(v2));
    if (edge_on || !(abs(sum) > 1e-4 * (abs(e.x) + abs(e.y) + abs(e.z)))) {
        bary = vec3(1.0 / 3.0);
        ddx_bary = vec3(0.0);
        ddy_bary = vec3(0.0);
        return false; // edge-on (also catches NaN)
    }

    // One pixel is 2 / size in NDC; +y in pixels is +y in NDC (Vulkan, Godot's Y-flipped projection).
    vec3 de_dx = vec3(c0.x, c1.x, c2.x) * (2.0 * inv_viewport.x);
    vec3 de_dy = vec3(c0.y, c1.y, c2.y) * (2.0 * inv_viewport.y);

    bary     = e / sum;
    ddx_bary = (de_dx - bary * (de_dx.x + de_dx.y + de_dx.z)) / sum;
    ddy_bary = (de_dy - bary * (de_dy.x + de_dy.y + de_dy.z)) / sum;
    return true;
}

// Octahedral unit-vector encoding into [0, 1]^2. Decode with lumina_oct_decode(e * 2.0 - 1.0).
// Note: vis_shade_materials writes WORLD-space normals; Lumina's Track A input is Godot's
// view-space normal_roughness buffer, so a Track B Lumina input needs its own read path.
vec2 encode_octahedral_normal(vec3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z) + 1e-8);
    if (n.z < 0.0) {
        vec2 sign_not_zero = vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
        n.xy = (1.0 - abs(n.yx)) * sign_not_zero;
    }
    return n.xy * 0.5 + 0.5;
}

#endif // PRISM_VIS_SHADING_COMMON_GLSL
