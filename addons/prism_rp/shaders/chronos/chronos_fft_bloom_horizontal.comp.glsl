#version 460
#extension GL_GOOGLE_include_directive : require

#include "prism_common.glsl"

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D u_hdr_color_tex;
layout(rgba32f, set = 0, binding = 1) uniform writeonly image2D u_fft_rg_out;
layout(rgba32f, set = 0, binding = 2) uniform writeonly image2D u_fft_b_out;

layout(push_constant, std430) uniform PushConstants {
    float bloom_threshold;
    float bloom_soft_knee;
    float exposure_scale;
    float _pad;
} pc;

shared vec2 s_r_a[256];
shared vec2 s_g_a[256];
shared vec2 s_b_a[256];

shared vec2 s_r_b[256];
shared vec2 s_g_b[256];
shared vec2 s_b_b[256];

vec2 cmul(vec2 a, vec2 b) {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Each bloom texel covers a (source_size / 224)-pixel footprint. A 4x4 grid of bilinear taps
// spread across that footprint integrates it (each tap averages 2x2 texels), which keeps small
// highlights from popping in and out as they move between bloom texels.
vec3 prefilter_acescg_highlights(vec2 uv, float active_span) {
    vec2 src_size = vec2(textureSize(u_hdr_color_tex, 0));
    vec2 footprint_uv = 1.0 / vec2(active_span);
    vec2 min_step = 1.0 / src_size;

    vec3 color = vec3(0.0);
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            vec2 offset = (vec2(x, y) + 0.5) / 4.0 - 0.5;
            vec2 tap_uv = uv + offset * max(footprint_uv, 2.0 * min_step);
            vec3 tap = textureLod(u_hdr_color_tex, tap_uv, 0.0).rgb;
            color += prism_linear_srgb_to_acescg(max(tap, vec3(0.0)));
        }
    }
    color *= (1.0 / 16.0) * pc.exposure_scale;
    color = min(color, vec3(64.0));

    float luma = prism_acescg_luminance(color);
    float knee = pc.bloom_threshold * pc.bloom_soft_knee + 1e-5;
    float soft = clamp(luma - pc.bloom_threshold + knee, 0.0, 2.0 * knee);
    soft = (soft * soft) / (4.0 * knee);

    float contribution = max(soft, luma - pc.bloom_threshold) / max(luma, 1e-5);
    return color * max(contribution, 0.0);
}

void main() {
    uint t = gl_LocalInvocationID.x;
    uint row = gl_WorkGroupID.y;

    const float pad_margin = 16.0;
    const float active_span = 256.0 - 2.0 * pad_margin;

    uint x0 = t;
    uint x1 = t + 128u;

    vec3 rgb0 = vec3(0.0);
    vec3 rgb1 = vec3(0.0);

    vec2 coord0 = vec2(float(x0), float(row)) - vec2(pad_margin);
    vec2 coord1 = vec2(float(x1), float(row)) - vec2(pad_margin);

    if (all(greaterThanEqual(coord0, vec2(0.0))) && all(lessThan(coord0, vec2(active_span)))) {
        rgb0 = prefilter_acescg_highlights((coord0 + 0.5) / active_span, active_span);
    }
    if (all(greaterThanEqual(coord1, vec2(0.0))) && all(lessThan(coord1, vec2(active_span)))) {
        rgb1 = prefilter_acescg_highlights((coord1 + 0.5) / active_span, active_span);
    }

    s_r_a[x0] = vec2(rgb0.r, 0.0);
    s_g_a[x0] = vec2(rgb0.g, 0.0);
    s_b_a[x0] = vec2(rgb0.b, 0.0);

    s_r_a[x1] = vec2(rgb1.r, 0.0);
    s_g_a[x1] = vec2(rgb1.g, 0.0);
    s_b_a[x1] = vec2(rgb1.b, 0.0);

    barrier();

    for (uint stage = 0u; stage < 8u; ++stage) {
        uint m = 1u << stage;
        uint j = t & (m - 1u);
        uint k = t >> stage;

        uint idx0 = k * m + j;
        uint idx1 = idx0 + 128u;
        uint out0 = k * (2u * m) + j;
        uint out1 = out0 + m;

        float angle = -3.14159265359 * float(j) / float(m);
        vec2 w = vec2(cos(angle), sin(angle));

        if ((stage & 1u) == 0u) {
            vec2 tr = cmul(w, s_r_a[idx1]);
            vec2 tg = cmul(w, s_g_a[idx1]);
            vec2 tb = cmul(w, s_b_a[idx1]);

            s_r_b[out0] = s_r_a[idx0] + tr;
            s_r_b[out1] = s_r_a[idx0] - tr;
            s_g_b[out0] = s_g_a[idx0] + tg;
            s_g_b[out1] = s_g_a[idx0] - tg;
            s_b_b[out0] = s_b_a[idx0] + tb;
            s_b_b[out1] = s_b_a[idx0] - tb;
        } else {
            vec2 tr = cmul(w, s_r_b[idx1]);
            vec2 tg = cmul(w, s_g_b[idx1]);
            vec2 tb = cmul(w, s_b_b[idx1]);

            s_r_a[out0] = s_r_b[idx0] + tr;
            s_r_a[out1] = s_r_b[idx0] - tr;
            s_g_a[out0] = s_g_b[idx0] + tg;
            s_g_a[out1] = s_g_b[idx0] - tg;
            s_b_a[out0] = s_b_b[idx0] + tb;
            s_b_a[out1] = s_b_b[idx0] - tb;
        }

        barrier();
    }

    imageStore(u_fft_rg_out, ivec2(int(x0), int(row)), vec4(s_r_a[x0], s_g_a[x0]));
    imageStore(u_fft_b_out,  ivec2(int(x0), int(row)), vec4(s_b_a[x0], 0.0, 0.0));

    imageStore(u_fft_rg_out, ivec2(int(x1), int(row)), vec4(s_r_a[x1], s_g_a[x1]));
    imageStore(u_fft_b_out,  ivec2(int(x1), int(row)), vec4(s_b_a[x1], 0.0, 0.0));
}
