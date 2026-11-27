#ifndef LUMINA_COMMON_GLSL
#define LUMINA_COMMON_GLSL

// Shared helpers for Lumina-Cascades. View space is Godot's: +X right, +Y up, camera looks down -Z.

// Godot 4.4 Forward+ normal_roughness (RGBA8): view-space normal packed with "best fit normals"
// (scaled, not rotated), so normalize(rgb * 2 - 1) recovers it. See scene_forward_clustered.glsl.
vec3 lumina_decode_normal(vec4 normal_roughness) {
    return normalize(normal_roughness.xyz * 2.0 - 1.0);
}

// Octahedral direction mapping on [-1, 1]^2. A uniform grid subdivides cleanly: each texel of an
// NxN map owns exactly the 2x2 texels below it in the 2Nx2N map, which is what cascade merging needs.
vec2 lumina_sign_not_zero(vec2 v) {
    return vec2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
}

vec3 lumina_oct_decode(vec2 e) {
    vec3 v = vec3(e, 1.0 - abs(e.x) - abs(e.y));
    if (v.z < 0.0) {
        v.xy = (1.0 - abs(v.yx)) * lumina_sign_not_zero(v.xy);
    }
    return normalize(v);
}

// Direction for texel `d` of a `block` x `block` octahedral map.
vec3 lumina_block_direction(ivec2 d, int block) {
    return lumina_oct_decode((vec2(d) + 0.5) / float(block) * 2.0 - 1.0);
}

// proj = (P00, P11, P20, P21) of the depth-corrected projection (flip-Y, so +ndc.y is down, as UV).
// depth_params: see prism_linearize_depth. Returns view-space position; z is negative in front.
vec3 lumina_view_position(vec2 uv, float depth, vec4 proj, vec2 depth_params) {
    float linear_z = prism_linearize_depth(depth, depth_params);
    vec2 ndc = uv * 2.0 - 1.0;
    float view_z = -linear_z;
    return vec3(-view_z * (ndc.x + proj.z) / proj.x, -view_z * (ndc.y + proj.w) / proj.y, view_z);
}

// Inverse of lumina_view_position's projection. Only valid for p.z < 0.
vec2 lumina_project_to_uv(vec3 p, vec4 proj) {
    vec2 ndc = vec2(proj.x * p.x + proj.z * p.z, proj.y * p.y + proj.w * p.z) / (-p.z);
    return ndc * 0.5 + 0.5;
}

float lumina_luminance(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

bool lumina_is_sky(float depth) {
    return depth <= 1e-6;  // reverse-Z: the far plane / sky is 0
}

// How far a parent probe's light can stand in for a child probe's, from their linear depths (> 0): an
// exponential falloff in their relative depth difference. Shared by the merge, which weights parents by
// it, and the trace, which runs a probe's interval on by itself when no parent qualifies.
const float kLuminaParentDepthTolerance = 0.1;

float lumina_parent_depth_weight(float child_z, float parent_z) {
    return exp(-abs(parent_z - child_z) / (kLuminaParentDepthTolerance * child_z));
}

#endif // LUMINA_COMMON_GLSL
