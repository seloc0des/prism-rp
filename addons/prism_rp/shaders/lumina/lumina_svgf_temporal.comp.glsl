#version 460
#extension GL_GOOGLE_include_directive : require

#include "prism_common.glsl"
#include "lumina_common.glsl"

// SVGF temporal accumulation: reproject last frame's accumulated irradiance and luminance moments,
// reject history across depth/normal discontinuities, clamp it to this frame's neighbourhood, blend,
// and estimate per-pixel variance for the a-trous filter. History length (frames) is kept in history.a.
//
// The clamp keeps lighting changes from lingering: history may differ from this frame's 3x3 mean by
// kClampSigma standard deviations plus kClampRelative of the mean. Frame-to-frame noise stays inside
// that band, so history still averages it; a light switching on or off moves the mean outside it,
// and the history is pulled to the band's edge and shortened, so the result follows within a frame
// or two instead of over 1 / alpha_min frames (Gate 2: settled within 6).

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D u_irradiance_tex;     // this frame, raw
layout(set = 0, binding = 1) uniform sampler2D u_prev_history_tex;   // rgb + history length
layout(set = 0, binding = 2) uniform sampler2D u_prev_moments_tex;   // luminance m1, m2
layout(set = 0, binding = 3) uniform sampler2D u_prev_geometry_tex;  // xyz view normal, w linear depth
layout(set = 0, binding = 4) uniform sampler2D u_velocity_tex;
layout(set = 0, binding = 5) uniform sampler2D u_depth_tex;
layout(set = 0, binding = 6) uniform sampler2D u_normal_roughness_tex;
layout(rgba16f, set = 0, binding = 7) uniform writeonly image2D u_history_out;
layout(rg16f, set = 0, binding = 8) uniform writeonly image2D u_moments_out;
layout(rgba16f, set = 0, binding = 9) uniform writeonly image2D u_geometry_out;
layout(rgba16f, set = 0, binding = 10) uniform writeonly image2D u_filter_in_out;  // rgb + variance

layout(push_constant, std430) uniform PushConstants {
    ivec2 internal_resolution;
    vec2 depth_params;
    uint flags;              // bit 0: reset history
    float alpha_min;         // minimum blend weight of the new sample (e.g. 0.1)
    float moments_alpha_min;
    float max_history;       // history length cap, in frames
} pc;

const float kClampSigma = 1.5;
const float kClampRelative = 0.15;
const float kClampFloor = 1e-3;   // absolute, so black neighbourhoods still allow a little drift

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= pc.internal_resolution.x || pixel.y >= pc.internal_resolution.y) {
        return;
    }

    vec2 res_f = vec2(pc.internal_resolution);
    vec2 uv = (vec2(pixel) + 0.5) / res_f;
    vec3 current = texelFetch(u_irradiance_tex, pixel, 0).rgb;
    float depth = texelFetch(u_depth_tex, pixel, 0).r;
    float z = lumina_is_sky(depth) ? 0.0 : prism_linearize_depth(depth, pc.depth_params);
    vec3 n = lumina_decode_normal(texelFetch(u_normal_roughness_tex, pixel, 0));

    float luma = lumina_luminance(current);
    vec2 moments = vec2(luma, luma * luma);
    vec3 color = current;
    float history_len = 1.0;

    // Godot velocity = prev_uv - curr_uv.
    vec2 prev_uv = uv + texelFetch(u_velocity_tex, pixel, 0).rg;
    bool in_bounds = all(greaterThanEqual(prev_uv, vec2(0.0))) && all(lessThanEqual(prev_uv, vec2(1.0)));

    if ((pc.flags & 1u) == 0u && in_bounds && z > 0.0) {
        vec4 prev_geo = textureLod(u_prev_geometry_tex, prev_uv, 0.0);
        bool same_surface =
            prev_geo.w > 0.0 &&
            abs(prev_geo.w - z) / z < 0.1 &&
            dot(normalize(prev_geo.xyz), n) > 0.9;
        if (same_surface) {
            vec4 hist = textureLod(u_prev_history_tex, prev_uv, 0.0);
            vec2 prev_moments = textureLod(u_prev_moments_tex, prev_uv, 0.0).rg;
            history_len = min(hist.a + 1.0, pc.max_history);

            vec3 m1 = vec3(0.0);
            vec3 m2 = vec3(0.0);
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    ivec2 q = clamp(pixel + ivec2(dx, dy), ivec2(0), pc.internal_resolution - 1);
                    vec3 c = texelFetch(u_irradiance_tex, q, 0).rgb;
                    m1 += c;
                    m2 += c * c;
                }
            }
            m1 *= 1.0 / 9.0;
            m2 *= 1.0 / 9.0;
            vec3 band = kClampSigma * sqrt(max(m2 - m1 * m1, vec3(0.0))) + kClampRelative * m1 + vec3(kClampFloor);
            vec3 clamped = clamp(hist.rgb, m1 - band, m1 + band);
            if (any(greaterThan(abs(clamped - hist.rgb), 0.5 * band))) {
                history_len = min(history_len, 2.0);   // a lighting change: converge again quickly
            }

            float alpha = max(1.0 / history_len, pc.alpha_min);
            float alpha_m = max(1.0 / history_len, pc.moments_alpha_min);
            color = mix(clamped, current, alpha);
            moments = mix(prev_moments, moments, alpha_m);
        }
    }

    float variance = max(moments.y - moments.x * moments.x, 0.0);
    // Too little history for a temporal estimate: assume high variance so a-trous filters harder.
    if (history_len < 4.0) {
        variance = max(variance, (4.0 - history_len) * 0.25 * max(luma * luma, 0.01));
    }

    imageStore(u_history_out, pixel, vec4(color, history_len));
    imageStore(u_moments_out, pixel, vec4(moments, 0.0, 0.0));
    imageStore(u_geometry_out, pixel, vec4(n, z));
    imageStore(u_filter_in_out, pixel, vec4(color, variance));
}
