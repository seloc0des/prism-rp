#version 460
#extension GL_GOOGLE_include_directive : require

#include "prism_common.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D u_color_tex;
layout(set = 0, binding = 1) uniform sampler2D u_prev_color_tex;
layout(set = 0, binding = 2) uniform sampler2D u_depth_tex;
layout(set = 0, binding = 3) uniform sampler2D u_dilated_velocity_tex;
layout(r8, set = 0, binding = 4) uniform writeonly image2D u_reactive_mask_out;

layout(push_constant, std430) uniform PushConstants {
    ivec2 resolution;
    float reactive_sensitivity;
    float luma_spark_threshold;
    vec2 jitter_delta_uv;   // (current - previous) camera jitter, UV units
    vec2 depth_params;      // see prism_linearize_depth
} pc;

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= pc.resolution.x || pixel.y >= pc.resolution.y) {
        return;
    }

    vec2 inv_res = 1.0 / vec2(pc.resolution);
    vec2 uv = (vec2(pixel) + 0.5) * inv_res;

    vec3 curr_rgb = texelFetch(u_color_tex, pixel, 0).rgb;
    vec3 curr_ap1 = prism_linear_srgb_to_acescg(max(curr_rgb, vec3(0.0)));
    float curr_luma = prism_acescg_luminance(curr_ap1);

    float spark_reactivity = clamp(
        (curr_luma - pc.luma_spark_threshold) / max(pc.luma_spark_threshold * 2.0, 0.25),
        0.0,
        1.0
    );

    // Dilated velocity is jitter-free; both color frames here are raw (jittered),
    // so remove the jitter delta again to reproject raw-to-raw. prev = uv + velocity.
    vec2 velocity = texelFetch(u_dilated_velocity_tex, pixel, 0).rg;
    vec2 prev_uv = uv + velocity - pc.jitter_delta_uv;
    float shock_reactivity = 0.0;

    if (all(greaterThanEqual(prev_uv, vec2(0.0))) && all(lessThanEqual(prev_uv, vec2(1.0)))) {
        vec3 prev_rgb = textureLod(u_prev_color_tex, prev_uv, 0.0).rgb;
        vec3 prev_ap1 = prism_linear_srgb_to_acescg(max(prev_rgb, vec3(0.0)));
        float prev_luma = prism_acescg_luminance(prev_ap1);

        float luma_delta = abs(log2(curr_luma + 0.0625) - log2(prev_luma + 0.0625));
        float chroma_delta = length(curr_ap1 / (curr_luma + 0.25) - prev_ap1 / (prev_luma + 0.25));

        float static_boost = mix(1.35, 0.85, clamp(length(velocity) * 60.0, 0.0, 1.0));
        shock_reactivity = clamp((luma_delta * 0.55 + chroma_delta * 0.45) * static_boost - 0.12, 0.0, 1.0);
    } else {
        shock_reactivity = 1.0;
    }

    // Relative linear-depth discontinuity, so edges weigh the same near and far from the camera.
    float center_depth = prism_linearize_depth(texelFetch(u_depth_tex, pixel, 0).r, pc.depth_params);
    float max_depth_diff = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            if (x == 0 && y == 0) continue;
            ivec2 sc = clamp(pixel + ivec2(x, y), ivec2(0), pc.resolution - ivec2(1));
            float nd = prism_linearize_depth(texelFetch(u_depth_tex, sc, 0).r, pc.depth_params);
            max_depth_diff = max(max_depth_diff, abs(nd - center_depth) / max(min(nd, center_depth), 1e-4));
        }
    }
    float edge_reactivity = clamp((max_depth_diff - 0.05) * 3.0, 0.0, 0.65);

    float combined = max(max(spark_reactivity, shock_reactivity), edge_reactivity);
    float final_reactive = clamp(combined * pc.reactive_sensitivity, 0.0, 1.0);

    imageStore(u_reactive_mask_out, pixel, vec4(final_reactive, 0.0, 0.0, 0.0));
}
