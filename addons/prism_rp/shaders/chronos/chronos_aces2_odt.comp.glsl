#version 460
#extension GL_GOOGLE_include_directive : require

#include "prism_common.glsl"

// Chronos composite: bloom add, post-history sharpening, and the optional SDR tone curve.
//
// Output contract: compositor effects run before Godot's own tonemap pass, which converts
// linear -> the display encoding. So this pass always writes LINEAR Rec.709:
//  - look disabled: scene-referred linear HDR; Godot's tonemapper handles display mapping.
//  - look enabled: display-referred linear in [0, output_max_linear]: [0, 1] on an SDR display;
//    with Godot 4.7's HDR output, up to the display's maximum linear value (1.0 = SDR white).
//    The Environment must use TONE_MAPPER_LINEAR with exposure 1.0 so Godot only encodes.

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D u_resolved_tex;        // linear Rec.709 HDR (alpha: TAA confidence)
layout(set = 0, binding = 1) uniform sampler2D u_bloom_tex;           // ACEScg bloom, 256x256 padded grid
layout(set = 0, binding = 2) uniform sampler2D u_dilated_velocity_tex;
layout(set = 0, binding = 3) uniform sampler2D u_reactive_mask_tex;
layout(rgba16f, set = 0, binding = 4) uniform writeonly image2D u_color_out;

layout(push_constant, std430) uniform PushConstants {
    ivec2 resolution;
    float exposure_scale;
    float sharpness;
    int debug_view_mode;  // 0: Final, 1: Dilated Velocity, 2: Reactive Mask, 3: FFT Bloom, 4: TAA Confidence, 5: Signed Velocity
    uint flags;           // bit 0: bloom, bit 1: SDR look, bit 2: sharpen, bit 3: alpha holds TAA confidence
    float output_max_linear; // display peak in linear units (SDR white = 1); <= 1 means an SDR display
    float _pad1;
} pc;

const uint FLAG_BLOOM = 1u;
const uint FLAG_SDR_LOOK = 2u;
const uint FLAG_SHARPEN = 4u;
const uint FLAG_HAS_CONFIDENCE = 8u;

// Debug view 5 writes rg = 0.5 + velocity * this: signed, so automated checks (tests/check_motion_vectors.py)
// can read the motion vectors Chronos uses, after jitter compensation and dilation.
const float SIGNED_VELOCITY_SCALE = 10.0;

// ACES 1.3 Reference Gamut Compression (power 1.2): maps distance `limit` onto the gamut
// boundary (1.0) and leaves distances below `threshold` untouched.
vec3 aces_reference_gamut_compress(vec3 ap1) {
    float achromatic = max(max(ap1.r, ap1.g), ap1.b);
    if (achromatic <= 1e-5) {
        return ap1;
    }
    const vec3 thr = vec3(0.815, 0.803, 0.880);
    const vec3 lim = vec3(1.147, 1.264, 1.312);
    const float p = 1.2;

    vec3 dist = (achromatic - ap1) / achromatic;
    vec3 scl = (lim - thr) / pow(pow((1.0 - thr) / (lim - thr), vec3(-p)) - 1.0, vec3(1.0 / p));
    vec3 nd = max(dist - thr, vec3(0.0)) / scl;
    vec3 compressed = thr + scl * nd / pow(1.0 + pow(nd, vec3(p)), vec3(1.0 / p));
    vec3 final_dist = mix(dist, compressed, greaterThan(dist, thr));
    return achromatic * (1.0 - final_dist);
}

// Highlight expansion for HDR displays: f * (1 + (peak - 1) * f^4) leaves shadows and midtones
// where the SDR curve puts them (mid grey moves < 0.3%), rises monotonically, and approaches
// `peak` as the SDR curve approaches white.
const float HDR_EXPANSION_POWER = 4.0;

// SDR look: ACES 1.3 gamut compression + a custom filmic tonescale. Not the ACES 2.0 ODT.
// Returns display-referred LINEAR Rec.709 in [0, 1], or [0, peak] when peak > 1 (HDR output).
vec3 apply_sdr_look(vec3 ap1, float peak) {
    ap1 = aces_reference_gamut_compress(ap1);

    float y = max(prism_acescg_luminance(ap1), 0.0);
    const float white_scale = 1.0;
    const float toe_strength = 0.018;
    const float mid_gray = 0.18;
    float y_toe = (y * y) / (y + toe_strength);

    const float n = 1.12;
    float k = pow(mid_gray, 1.0 - n) * (white_scale - mid_gray) / white_scale;
    float y_mapped = white_scale * pow(y_toe, n) / (pow(y_toe, n) + k);
    if (peak > 1.0) {
        y_mapped *= 1.0 + (peak - 1.0) * pow(y_mapped, HDR_EXPANSION_POWER);
    }

    float chroma_attenuation = 1.0 / (1.0 + 0.42 * y_mapped);
    vec3 chroma_dir = ap1 - vec3(y);
    float luma_gain = y_mapped / max(y, 1e-6);

    vec3 toned_ap1 = max(vec3(y_mapped) + chroma_dir * luma_gain * chroma_attenuation, vec3(0.0));
    return clamp(prism_aces2065_to_rec709(prism_acescg_to_aces2065(toned_ap1)), vec3(0.0), vec3(max(peak, 1.0)));
}

// Reversible tonemap so sharpening behaves the same on HDR highlights as on midtones.
vec3 rt_forward(vec3 c) { return c / (1.0 + max(c.r, max(c.g, c.b))); }
vec3 rt_inverse(vec3 t) { return t / max(1.0 - max(t.r, max(t.g, t.b)), 1e-4); }

vec3 fetch_rt(ivec2 p) {
    p = clamp(p, ivec2(0), pc.resolution - ivec2(1));
    return rt_forward(max(texelFetch(u_resolved_tex, p, 0).rgb, vec3(0.0)));
}

// Lightweight RCAS-style sharpen on the resolved image (post-history, so it never accumulates).
vec3 sharpen(ivec2 pixel, vec3 center_rgb) {
    vec3 c = rt_forward(center_rgb);
    vec3 n = fetch_rt(pixel + ivec2(0, -1));
    vec3 w = fetch_rt(pixel + ivec2(-1, 0));
    vec3 e = fetch_rt(pixel + ivec2(1, 0));
    vec3 s = fetch_rt(pixel + ivec2(0, 1));

    vec3 min_bound = min(min(min(n, s), min(w, e)), c);
    vec3 max_bound = max(max(max(n, s), max(w, e)), c);
    vec3 amp = clamp(min(min_bound, 1.0 - max_bound) / max(max_bound, vec3(1e-4)), vec3(0.0), vec3(1.0));
    vec3 lobe = -sqrt(amp) * (pc.sharpness * 0.25);

    vec3 sharpened = (c + (n + s + w + e) * lobe) / max(vec3(1.0) + 4.0 * lobe, vec3(0.25));
    return rt_inverse(clamp(sharpened, min_bound, max_bound));
}

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= pc.resolution.x || pixel.y >= pc.resolution.y) {
        return;
    }

    vec2 uv = (vec2(pixel) + 0.5) / vec2(pc.resolution);
    vec4 src = texelFetch(u_resolved_tex, pixel, 0);

    vec2 bloom_uv = (vec2(16.0) + uv * 224.0) / 256.0;
    vec3 bloom_ap1 = textureLod(u_bloom_tex, bloom_uv, 0.0).rgb;

    if (pc.debug_view_mode == 1) {
        vec2 vel = texelFetch(u_dilated_velocity_tex, pixel, 0).rg;
        vec3 vel_vis = vec3(abs(vel.x) * 32.0, abs(vel.y) * 32.0, length(vel) * 16.0);
        imageStore(u_color_out, pixel, vec4(vel_vis, 1.0));
        return;
    } else if (pc.debug_view_mode == 2) {
        float reactive = texelFetch(u_reactive_mask_tex, pixel, 0).r;
        vec3 heat = mix(vec3(0.02, 0.05, 0.18), vec3(1.0, 0.22, 0.04), reactive);
        imageStore(u_color_out, pixel, vec4(heat, 1.0));
        return;
    } else if (pc.debug_view_mode == 3) {
        imageStore(u_color_out, pixel, vec4(apply_sdr_look(bloom_ap1, pc.output_max_linear), 1.0));
        return;
    } else if (pc.debug_view_mode == 4) {
        float conf = ((pc.flags & FLAG_HAS_CONFIDENCE) != 0u) ? clamp(src.a, 0.0, 1.0) : 0.0;
        vec3 conf_vis = mix(vec3(0.95, 0.15, 0.08), vec3(0.10, 0.85, 0.35), conf);
        imageStore(u_color_out, pixel, vec4(conf_vis, 1.0));
        return;
    } else if (pc.debug_view_mode == 5) {
        vec2 vel = texelFetch(u_dilated_velocity_tex, pixel, 0).rg;
        imageStore(u_color_out, pixel, vec4(0.5 + vel * SIGNED_VELOCITY_SCALE, 0.5, 1.0));
        return;
    }

    vec3 scene_rgb = max(src.rgb, vec3(0.0));
    if ((pc.flags & FLAG_SHARPEN) != 0u) {
        scene_rgb = sharpen(pixel, scene_rgb);
    }

    vec3 scene_ap1 = prism_linear_srgb_to_acescg(scene_rgb) * pc.exposure_scale;
    if ((pc.flags & FLAG_BLOOM) != 0u) {
        scene_ap1 += bloom_ap1;
    }

    vec3 out_rgb;
    if ((pc.flags & FLAG_SDR_LOOK) != 0u) {
        out_rgb = apply_sdr_look(scene_ap1, pc.output_max_linear);
    } else {
        out_rgb = max(prism_acescg_to_linear_srgb(scene_ap1), vec3(0.0));
    }

    // Alpha: TAA confidence lives in the history, not the frame; keep the frame opaque.
    imageStore(u_color_out, pixel, vec4(out_rgb, 1.0));
}
