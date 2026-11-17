#ifndef PRISM_COMMON_GLSL
#define PRISM_COMMON_GLSL

// Reconstruct linear view-space depth from Godot's reverse-Z infinite far plane depth buffer
float prism_linearize_depth_reverse_z(float depth, float near_plane) {
    return near_plane / max(depth, 1e-6);
}

// --- Primary Color Space Conversion Matrices (D65 Bradford-Adapted) ---

// Linear Rec.709 / sRGB -> ACEScg (AP1)
vec3 prism_linear_srgb_to_acescg(vec3 c) {
    const mat3 m = mat3(
        vec3(0.613097, 0.070194, 0.020616),
        vec3(0.339523, 0.916354, 0.109570),
        vec3(0.047379, 0.013452, 0.869815)
    );
    return m * c;
}

// ACEScg (AP1) -> Linear Rec.709 / sRGB
vec3 prism_acescg_to_linear_srgb(vec3 c) {
    const mat3 m = mat3(
        vec3( 1.705051, -0.130256, -0.024003),
        vec3(-0.621792,  1.140805, -0.128969),
        vec3(-0.083259, -0.010548,  1.152972)
    );
    return m * c;
}

// ACEScg (AP1) -> ACES2065-1 (AP0)
vec3 prism_acescg_to_aces2065(vec3 c) {
    const mat3 m = mat3(
        vec3(0.695452, 0.044795, -0.005526),
        vec3(0.140679, 0.859671,  0.004025),
        vec3(0.163869, 0.095534,  1.001501)
    );
    return m * c;
}

// ACES2065-1 (AP0) -> Linear Rec.709 / sRGB (D65)
vec3 prism_aces2065_to_rec709(vec3 c) {
    const mat3 m = mat3(
        vec3( 2.521686, -0.276480, -0.015375),
        vec3(-1.134131,  1.372719, -0.152975),
        vec3(-0.387555, -0.096239,  1.168350)
    );
    return m * c;
}

// ACES2065-1 (AP0) -> Linear Display P3 (D65)
vec3 prism_aces2065_to_display_p3(vec3 c) {
    const mat3 m = mat3(
        vec3( 2.041369, -0.205921, -0.012465),
        vec3(-0.564946,  1.264301, -0.133793),
        vec3(-0.476423, -0.058380,  1.146258)
    );
    return m * c;
}

// ACES2065-1 (AP0) -> Linear Rec.2020 (D65)
vec3 prism_aces2065_to_rec2020(vec3 c) {
    const mat3 m = mat3(
        vec3( 1.490409, -0.080168,  0.003223),
        vec3(-0.266171,  1.182167, -0.034761),
        vec3(-0.224238, -0.101999,  1.031538)
    );
    return m * c;
}

// ACEScg (AP1) relative luminance weights
float prism_acescg_luminance(vec3 ap1) {
    return dot(ap1, vec3(0.2722287, 0.6740818, 0.0536895));
}

// --- Display Opto-Electronic Transfer Functions (OETF) ---

vec3 prism_encode_srgb(vec3 x) {
    x = max(x, vec3(0.0));
    vec3 lo = x * 12.92;
    vec3 hi = 1.055 * pow(x, vec3(1.0 / 2.4)) - 0.055;
    return mix(lo, hi, greaterThan(x, vec3(0.0031308)));
}

vec3 prism_encode_pq_st2084(vec3 nits) {
    const float m1 = 2610.0 / 16384.0;
    const float m2 = (2523.0 / 4096.0) * 128.0;
    const float c1 = 3424.0 / 4096.0;
    const float c2 = (2413.0 / 4096.0) * 32.0;
    const float c3 = (2392.0 / 4096.0) * 32.0;

    vec3 y = clamp(nits / 10000.0, vec3(0.0), vec3(1.0));
    vec3 ym1 = pow(y, vec3(m1));
    return pow((c1 + c2 * ym1) / (1.0 + c3 * ym1), vec3(m2));
}

// Jimenez Interleaved Gradient Noise (IGN) for temporal dithering
float prism_interleaved_gradient_noise(vec2 pixel, uint frame_index) {
    vec2 p = pixel + float(frame_index & 63u) * vec2(5.588238, 5.588238);
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

// Linear RGB -> YCoCg (decorrelates luminance Y from chrominance Co/Cg for tight TAA clamping)
vec3 prism_rgb_to_ycocg(vec3 rgb) {
    float y  =  0.25 * rgb.r + 0.50 * rgb.g + 0.25 * rgb.b;
    float co =  0.50 * rgb.r                - 0.50 * rgb.b;
    float cg = -0.25 * rgb.r + 0.50 * rgb.g - 0.25 * rgb.b;
    return vec3(y, co, cg);
}

// YCoCg -> Linear RGB
vec3 prism_ycocg_to_rgb(vec3 ycocg) {
    float tmp = ycocg.x - ycocg.z;
    float r   = tmp + ycocg.y;
    float g   = ycocg.x + ycocg.z;
    float b   = tmp - ycocg.y;
    return max(vec3(r, g, b), vec3(0.0));
}

#endif // PRISM_COMMON_GLSL
