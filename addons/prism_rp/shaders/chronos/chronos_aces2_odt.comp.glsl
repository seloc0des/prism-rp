#version 460
#extension GL_GOOGLE_include_directive : require

#include "prism_common.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(rgba16f, set = 0, binding = 0) uniform image2D u_color_io;
layout(set = 0, binding = 1) uniform sampler2D u_bloom_tex;
layout(set = 0, binding = 2) uniform sampler2D u_dilated_velocity_tex;
layout(set = 0, binding = 3) uniform sampler2D u_reactive_mask_tex;

layout(push_constant, std430) uniform PushConstants {
    ivec2 resolution;
    float exposure_scale;
    float peak_luminance;
    int display_mode;     // 0: SDR sRGB, 1: Display P3, 2: HDR10 Rec.2020 PQ
    int debug_view_mode;  // 0: Final, 1: Dilated Velocity, 2: Reactive Mask, 3: FFT Bloom, 4: TAAU Confidence
    uint frame_index;
    uint flags;           // bit 0: bloom_enabled, bit 1: aces_odt_enabled
} pc;

vec3 apply_aces2_output_transform(vec3 ap1, float peak_nits, int display_mode) {
    float achromatic = max(max(ap1.r, ap1.g), ap1.b);
    if (achromatic > 1e-5) {
        vec3 dist = (achromatic - ap1) / achromatic;
        const vec3 compress_thresh = vec3(0.815, 0.803, 0.880);
        const vec3 compress_limit  = vec3(1.147, 1.264, 1.312);
        vec3 scale = compress_limit - compress_thresh;
        vec3 compressed = compress_thresh + scale * (
            (dist - compress_thresh) / max(scale + (dist - compress_thresh), vec3(1e-5))
        );
        vec3 final_dist = mix(dist, compressed, greaterThan(dist, compress_thresh));
        ap1 = achromatic * (1.0 - final_dist);
    }

    float y = max(prism_acescg_luminance(ap1), 0.0);
    float white_scale = peak_nits / 100.0;

    const float toe_strength = 0.018;
    const float mid_gray = 0.18;
    float y_toe = (y * y) / (y + toe_strength);

    float n = 1.12;
    float k = pow(mid_gray, 1.0 - n) * (white_scale - mid_gray) / max(white_scale, 0.25);
    float y_mapped = white_scale * pow(y_toe, n) / (pow(y_toe, n) + k);

    float chroma_attenuation = 1.0 / (1.0 + 0.42 * (y_mapped / max(white_scale, 0.1)));
    vec3 chroma_dir = ap1 - vec3(y);
    float luma_gain = y_mapped / max(y, 1e-6);

    vec3 toned_ap1 = max(vec3(y_mapped) + chroma_dir * luma_gain * chroma_attenuation, vec3(0.0));
    vec3 ap0 = prism_acescg_to_aces2065(toned_ap1);

    if (display_mode == 2) {
        vec3 rec2020 = max(prism_aces2065_to_rec2020(ap0), vec3(0.0));
        return prism_encode_pq_st2084(rec2020 * 100.0);
    } else if (display_mode == 1) {
        vec3 p3 = clamp(prism_aces2065_to_display_p3(ap0), vec3(0.0), vec3(1.0));
        return prism_encode_srgb(p3);
    } else {
        vec3 rec709 = clamp(prism_aces2065_to_rec709(ap0), vec3(0.0), vec3(1.0));
        return prism_encode_srgb(rec709);
    }
}

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= pc.resolution.x || pixel.y >= pc.resolution.y) {
        return;
    }

    vec2 uv = (vec2(pixel) + 0.5) / vec2(pc.resolution);
    vec4 src = imageLoad(u_color_io, pixel);

    vec2 bloom_uv = (vec2(16.0) + uv * 224.0) / 256.0;
    vec3 bloom_ap1 = textureLod(u_bloom_tex, bloom_uv, 0.0).rgb;

    if (pc.debug_view_mode == 1) {
        vec2 vel = texelFetch(u_dilated_velocity_tex, pixel, 0).rg;
        vec3 vel_vis = vec3(abs(vel.x) * 32.0, abs(vel.y) * 32.0, length(vel) * 16.0);
        imageStore(u_color_io, pixel, vec4(vel_vis, src.a));
        return;
    } else if (pc.debug_view_mode == 2) {
        float reactive = texelFetch(u_reactive_mask_tex, pixel, 0).r;
        vec3 heat = mix(vec3(0.02, 0.05, 0.18), vec3(1.0, 0.22, 0.04), reactive);
        imageStore(u_color_io, pixel, vec4(heat, src.a));
        return;
    } else if (pc.debug_view_mode == 3) {
        vec3 bloom_vis = apply_aces2_output_transform(bloom_ap1, pc.peak_luminance, pc.display_mode);
        imageStore(u_color_io, pixel, vec4(bloom_vis, src.a));
        return;
    } else if (pc.debug_view_mode == 4) {
        vec3 conf_vis = mix(vec3(0.95, 0.15, 0.08), vec3(0.10, 0.85, 0.35), clamp(src.a, 0.0, 1.0));
        imageStore(u_color_io, pixel, vec4(conf_vis, 1.0));
        return;
    }

    vec3 scene_ap1 = prism_linear_srgb_to_acescg(max(src.rgb, vec3(0.0))) * pc.exposure_scale;

    if ((pc.flags & 1u) != 0u) {
        scene_ap1 += bloom_ap1;
    }

    vec3 out_rgb;
    if ((pc.flags & 2u) != 0u) {
        out_rgb = apply_aces2_output_transform(scene_ap1, pc.peak_luminance, pc.display_mode);
    } else {
        out_rgb = max(prism_acescg_to_linear_srgb(scene_ap1), vec3(0.0));
    }

    float dither = (prism_interleaved_gradient_noise(vec2(pixel), pc.frame_index) - 0.5) / 255.0;
    out_rgb = max(out_rgb + vec3(dither), vec3(0.0));

    imageStore(u_color_io, pixel, vec4(out_rgb, src.a));
}
