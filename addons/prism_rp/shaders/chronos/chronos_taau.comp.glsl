#version 460
#extension GL_GOOGLE_include_directive : require

#include "prism_common.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D u_current_color_tex;
layout(set = 0, binding = 1) uniform sampler2D u_history_color_tex;
layout(set = 0, binding = 2) uniform sampler2D u_depth_tex;
layout(set = 0, binding = 3) uniform sampler2D u_dilated_velocity_tex;
layout(set = 0, binding = 4) uniform sampler2D u_reactive_mask_tex;
layout(rgba16f, set = 0, binding = 5) uniform writeonly image2D u_taau_output;

layout(push_constant, std430) uniform PushConstants {
    ivec2 internal_resolution;
    ivec2 target_resolution;
    vec2 jitter_offset;     // current camera jitter in internal pixels (+x right, +y down)
    float feedback_min;
    float feedback_max;
    float motion_scale;
    uint frame_index;
    uint flags;             // bit 0: history reset
    float _pad;
} pc;

vec4 sample_history_catmull_rom_5tap(sampler2D tex, vec2 uv, vec2 tex_size) {
    vec2 sample_pos = uv * tex_size;
    vec2 tc1 = floor(sample_pos - 0.5) + 0.5;
    vec2 f = sample_pos - tc1;
    vec2 f2 = f * f;
    vec2 f3 = f2 * f;

    vec2 w0 = f2 - 0.5 * (f3 + f);
    vec2 w1 = 1.5 * f3 - 2.5 * f2 + 1.0;
    vec2 w3 = 0.5 * (f3 - f2);
    vec2 w2 = 1.0 - w0 - w1 - w3;

    vec2 w12 = w1 + w2;
    vec2 offset12 = w2 / max(w12, vec2(1e-5));

    vec2 inv_size = 1.0 / tex_size;
    vec2 tc0  = (tc1 - 1.0) * inv_size;
    vec2 tc3  = (tc1 + 2.0) * inv_size;
    vec2 tc12 = (tc1 + offset12) * inv_size;

    float weight_top    = w12.x * w0.y;
    float weight_left   = w0.x  * w12.y;
    float weight_center = w12.x * w12.y;
    float weight_right  = w3.x  * w12.y;
    float weight_bottom = w12.x * w3.y;

    float total_weight = weight_top + weight_left + weight_center + weight_right + weight_bottom;

    vec4 result =
        textureLod(tex, vec2(tc12.x, tc0.y),  0.0) * weight_top +
        textureLod(tex, vec2(tc0.x,  tc12.y), 0.0) * weight_left +
        textureLod(tex, vec2(tc12.x, tc12.y), 0.0) * weight_center +
        textureLod(tex, vec2(tc3.x,  tc12.y), 0.0) * weight_right +
        textureLod(tex, vec2(tc12.x, tc3.y),  0.0) * weight_bottom;

    return result / max(total_weight, 1e-5);
}

vec3 clip_history_ycocg(vec3 history_ycocg, vec3 aabb_min, vec3 aabb_max) {
    vec3 extents = max(0.5 * (aabb_max - aabb_min), vec3(1e-5));
    vec3 box_center = 0.5 * (aabb_max + aabb_min);
    vec3 offset = history_ycocg - box_center;
    vec3 unit = abs(offset / extents);
    float max_unit = max(unit.x, max(unit.y, unit.z));

    if (max_unit > 1.0) {
        return box_center + offset / max_unit;
    }
    return history_ycocg;
}

float blackman_harris_weight(float dist) {
    float x = clamp(dist / 1.5, 0.0, 1.0);
    float t = 3.14159265359 * (1.0 - x);
    return 0.35875 - 0.48829 * cos(t) + 0.14128 * cos(2.0 * t) - 0.01168 * cos(3.0 * t);
}

void main() {
    ivec2 dst_pixel = ivec2(gl_GlobalInvocationID.xy);
    if (dst_pixel.x >= pc.target_resolution.x || dst_pixel.y >= pc.target_resolution.y) {
        return;
    }

    vec2 target_size_f = vec2(pc.target_resolution);
    vec2 internal_size_f = vec2(pc.internal_resolution);
    vec2 dst_uv = (vec2(dst_pixel) + 0.5) / target_size_f;

    vec2 src_pixel_pos = dst_uv * internal_size_f;
    ivec2 src_center_pixel = clamp(ivec2(floor(src_pixel_pos)), ivec2(0), pc.internal_resolution - ivec2(1));

    vec3 moment1 = vec3(0.0);
    vec3 moment2 = vec3(0.0);
    vec3 filtered_ycocg = vec3(0.0);
    float total_spatial_weight = 0.0;

    vec3 cross_min = vec3(1e6);
    vec3 cross_max = vec3(-1e6);

    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            ivec2 sample_coord = clamp(src_center_pixel + ivec2(x, y), ivec2(0), pc.internal_resolution - ivec2(1));
            vec3 tap_rgb = max(texelFetch(u_current_color_tex, sample_coord, 0).rgb, vec3(0.0));
            vec3 tap_ycocg = prism_rgb_to_ycocg(tap_rgb);

            moment1 += tap_ycocg;
            moment2 += tap_ycocg * tap_ycocg;

            bool is_cross = (abs(x) + abs(y)) <= 1;
            if (is_cross) {
                cross_min = min(cross_min, tap_ycocg);
                cross_max = max(cross_max, tap_ycocg);
            }

            // A jittered sample at pixel center c shows the scene point at c - jitter.
            vec2 delta = (vec2(sample_coord) + 0.5 - pc.jitter_offset) - src_pixel_pos;
            float w_spatial = blackman_harris_weight(length(delta));
            float w_karis = 1.0 / (1.0 + max(tap_ycocg.x, 0.0));
            float w = max(w_spatial * w_karis, 1e-5);

            filtered_ycocg += tap_ycocg * w;
            total_spatial_weight += w;
        }
    }

    filtered_ycocg /= max(total_spatial_weight, 1e-5);
    vec3 mean_ycocg = moment1 / 9.0;
    vec3 variance = max((moment2 / 9.0) - (mean_ycocg * mean_ycocg), vec3(0.0));
    vec3 stddev = sqrt(variance);

    vec2 velocity = texelFetch(u_dilated_velocity_tex, src_center_pixel, 0).rg;
    float reactive = textureLod(u_reactive_mask_tex, dst_uv, 0.0).r;

    float velocity_px = length(velocity * target_size_f);
    float gamma = mix(1.25, 0.75, clamp(max(reactive, velocity_px * 0.05), 0.0, 1.0));

    vec3 box_min = min(cross_min, mean_ycocg - gamma * stddev);
    vec3 box_max = max(cross_max, mean_ycocg + gamma * stddev);

    // Godot velocity = prev_uv - curr_uv (dilation already removed camera jitter).
    vec2 prev_uv = dst_uv + velocity;
    bool history_reset = (pc.flags & 1u) != 0u;
    bool in_bounds = !history_reset &&
        all(greaterThanEqual(prev_uv, vec2(0.0))) &&
        all(lessThanEqual(prev_uv, vec2(1.0)));

    vec3 current_rgb = prism_ycocg_to_rgb(filtered_ycocg);
    vec3 resolved_rgb = current_rgb;
    float history_confidence = 0.0;

    if (in_bounds) {
        vec4 history_sample = sample_history_catmull_rom_5tap(u_history_color_tex, prev_uv, target_size_f);
        vec3 history_rgb = max(history_sample.rgb, vec3(0.0));
        float prev_confidence = clamp(history_sample.a, 0.0, 1.0);

        vec3 history_ycocg = prism_rgb_to_ycocg(history_rgb);
        vec3 clipped_ycocg = clip_history_ycocg(history_ycocg, box_min, box_max);
        vec3 clipped_rgb = prism_ycocg_to_rgb(clipped_ycocg);

        float clip_dist = length(history_ycocg - clipped_ycocg) / max(mean_ycocg.x + 0.125, 0.125);
        float disocclusion_factor = clamp(clip_dist * 1.5, 0.0, 1.0);

        float motion_factor = clamp(velocity_px / max(pc.motion_scale, 1.0), 0.0, 1.0);
        float base_feedback = mix(pc.feedback_max, pc.feedback_min, max(motion_factor, disocclusion_factor));

        float confidence_scale = mix(0.65, 1.0, prev_confidence);
        float feedback = base_feedback * confidence_scale;
        feedback *= (1.0 - clamp(reactive * 0.94, 0.0, 1.0));

        float luma_curr = max(filtered_ycocg.x, 0.0);
        float luma_hist = max(clipped_ycocg.x, 0.0);
        float w_curr = (1.0 - feedback) / (1.0 + luma_curr);
        float w_hist = feedback / (1.0 + luma_hist);

        resolved_rgb = (current_rgb * w_curr + clipped_rgb * w_hist) / max(w_curr + w_hist, 1e-5);
        history_confidence = clamp(prev_confidence * (1.0 - disocclusion_factor) * (1.0 - reactive) + 0.125, 0.0, 1.0);
    } else {
        history_confidence = 0.125;
    }

    // No sharpening here: this output is next frame's history. Sharpening runs in the composite pass.
    imageStore(u_taau_output, dst_pixel, vec4(max(resolved_rgb, vec3(0.0)), history_confidence));
}
