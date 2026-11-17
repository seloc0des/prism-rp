#ifndef PRISM_COMMON_GLSL
#define PRISM_COMMON_GLSL

// Reconstruct linear view-space depth from Godot's reverse-Z infinite far plane depth buffer
float prism_linearize_depth_reverse_z(float depth, float near_plane) {
    return near_plane / max(depth, 1e-6);
}

// Convert linear Rec.709 / sRGB primaries to scene-referred ACEScg (AP1) working color space
vec3 prism_linear_srgb_to_acescg(vec3 c) {
    const mat3 srgb_to_ap1 = mat3(
        vec3(0.613097, 0.070194, 0.020616),
        vec3(0.339523, 0.916354, 0.109570),
        vec3(0.047379, 0.013452, 0.869815)
    );
    return srgb_to_ap1 * c;
}

#endif // PRISM_COMMON_GLSL
