#version 460
#extension GL_GOOGLE_include_directive : require

#include "prism_common.glsl"
#include "lumina_common.glsl"

// SVGF edge-avoiding a-trous wavelet filter (one iteration; run with step 1, 2, 4, ...).
// 5x5 B3-spline kernel; depth, normal and variance-scaled luminance edge-stopping.
// rgb = filtered irradiance, a = filtered variance.

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D u_input_tex;   // rgb + variance
layout(set = 0, binding = 1) uniform sampler2D u_geometry_tex;  // xyz view normal, w linear depth (0 for sky)
layout(rgba16f, set = 0, binding = 2) uniform writeonly image2D u_output;

layout(push_constant, std430) uniform PushConstants {
    ivec2 internal_resolution;
    int step_px;
    float phi_color;
    float phi_normal;
    float phi_depth;
    vec2 _pad;
} pc;

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= pc.internal_resolution.x || pixel.y >= pc.internal_resolution.y) {
        return;
    }

    vec4 center = texelFetch(u_input_tex, pixel, 0);
    vec4 geo = texelFetch(u_geometry_tex, pixel, 0);
    float z = geo.w;
    if (z <= 0.0) {
        imageStore(u_output, pixel, center);
        return;
    }
    vec3 n = geo.xyz * inversesqrt(dot(geo.xyz, geo.xyz));   // fp16 storage leaves |n| off 1 by up to 4e-4, and pow(., 128) amplifies it
    float luma = lumina_luminance(center.rgb);
    float sigma_l = pc.phi_color * sqrt(max(center.a, 1e-8)) + 1e-4;

    const float kernel[3] = float[3](3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0);

    vec3 sum = vec3(0.0);
    float sum_var = 0.0;
    float sum_w = 0.0;
    ivec2 max_px = pc.internal_resolution - ivec2(1);

    for (int y = -2; y <= 2; ++y) {
        for (int x = -2; x <= 2; ++x) {
            ivec2 q = clamp(pixel + ivec2(x, y) * pc.step_px, ivec2(0), max_px);
            vec4 s = texelFetch(u_input_tex, q, 0);
            vec4 qg = texelFetch(u_geometry_tex, q, 0);
            float qz = qg.w;
            if (qz <= 0.0) {
                continue;
            }
            vec3 qn = qg.xyz * inversesqrt(dot(qg.xyz, qg.xyz));

            float w_depth = exp(-abs(qz - z) / (pc.phi_depth * 0.05 * z * float(max(abs(x), abs(y))) + 1e-4));
            float w_normal = pow(max(dot(n, qn), 0.0), pc.phi_normal);
            float w_luma = exp(-abs(lumina_luminance(s.rgb) - luma) / sigma_l);
            float h = kernel[abs(x)] * kernel[abs(y)];
            float w = h * w_depth * w_normal * w_luma;

            sum += s.rgb * w;
            sum_var += s.a * w * w;
            sum_w += w;
        }
    }

    vec4 result = sum_w > 1e-6 ? vec4(sum / sum_w, sum_var / (sum_w * sum_w)) : center;
    imageStore(u_output, pixel, result);
}
