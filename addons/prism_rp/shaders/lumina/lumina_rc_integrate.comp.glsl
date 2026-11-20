#version 460
#extension GL_GOOGLE_include_directive : require

#include "prism_common.glsl"
#include "lumina_common.glsl"

// Per-pixel irradiance from the merged level-0 cascade: cosine-weighted sum over each probe's
// directions, blended over the 4 nearest probes with surface-aware bilinear weights (a probe must lie
// on the pixel's tangent plane and face the same way). Output is E / pi, i.e. outgoing diffuse
// radiance for albedo 1.

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D u_cascade0_tex;
layout(set = 0, binding = 1) uniform sampler2D u_depth_tex;
layout(set = 0, binding = 2) uniform sampler2D u_normal_roughness_tex;
layout(rgba16f, set = 0, binding = 3) uniform writeonly image2D u_irradiance_out;

layout(push_constant, std430) uniform PushConstants {
    ivec2 internal_resolution;
    ivec2 probe_count;
    int block;
    float probe_spacing_px;
    vec2 depth_params;
    vec4 proj;   // P00, P11, P20, P21 (see lumina_view_position)
} pc;

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= pc.internal_resolution.x || pixel.y >= pc.internal_resolution.y) {
        return;
    }

    float depth = texelFetch(u_depth_tex, pixel, 0).r;
    if (lumina_is_sky(depth)) {
        imageStore(u_irradiance_out, pixel, vec4(0.0));
        return;
    }
    vec2 res = vec2(pc.internal_resolution);
    vec3 pos = lumina_view_position((vec2(pixel) + 0.5) / res, depth, pc.proj, pc.depth_params);
    float z = -pos.z;
    vec3 n = lumina_decode_normal(texelFetch(u_normal_roughness_tex, pixel, 0));

    vec2 coord = (vec2(pixel) + 0.5) / pc.probe_spacing_px - 0.5;
    ivec2 p0 = ivec2(floor(coord));
    vec2 f = coord - vec2(p0);
    ivec2 max_probe = pc.probe_count - ivec2(1);

    ivec2 probes[4];
    float weights[4];
    float total_w = 0.0;
    for (int j = 0; j < 4; ++j) {
        weights[j] = 0.0;
        ivec2 o = ivec2(j & 1, j >> 1);
        ivec2 probe = clamp(p0 + o, ivec2(0), max_probe);
        float w = (o.x == 1 ? f.x : 1.0 - f.x) * (o.y == 1 ? f.y : 1.0 - f.y);

        ivec2 probe_px = clamp(ivec2((vec2(probe) + 0.5) * pc.probe_spacing_px), ivec2(0), pc.internal_resolution - ivec2(1));
        float probe_depth = texelFetch(u_depth_tex, probe_px, 0).r;
        if (lumina_is_sky(probe_depth)) {
            continue;
        }
        vec3 probe_pos = lumina_view_position((vec2(probe_px) + 0.5) / res, probe_depth, pc.proj, pc.depth_params);
        vec3 probe_n = lumina_decode_normal(texelFetch(u_normal_roughness_tex, probe_px, 0));
        // Distance from the pixel's tangent plane, not depth difference: on a surface seen at a grazing
        // angle depth changes fast across a probe spacing, and a depth test rejected the surface's own probes.
        w *= exp(-abs(dot(n, probe_pos - pos)) / max(0.05 * z, 1e-3));   // reject probes on other surfaces
        w *= pow(max(dot(probe_n, n), 0.0), 8.0);                        // and across creases
        if (w <= 1e-5) {
            continue;
        }
        probes[j] = probe;
        weights[j] = w;
        total_w += w;
    }

    // Deliberately no firefly clamp. A small bright spot rarely lands on one direction at full strength,
    // because the trace samples each hit at the mip matching its cone's footprint (lumina_rc_trace.comp.glsl).
    // A clamp would cut real light from any direction much brighter than the probe's mean, bright walls included.
    //
    // Each probe's irradiance is the normalised cosine-weighted average over its directions: E / pi =
    // sum(L cos) / sum(cos). Exact for uniform radiance, and it doesn't depend on per-texel solid angles
    // (octahedral texels aren't equal-area; assuming 4pi/N each made uniform light integrate ~18% dark at
    // 16 directions). The cosines depend on the pixel's normal only, so one pass over the directions
    // serves all four probes.
    vec3 total = vec3(0.0);
    float cos_sum = 0.0;
    for (int y = 0; y < pc.block; ++y) {
        for (int x = 0; x < pc.block; ++x) {
            float cos_theta = max(dot(lumina_block_direction(ivec2(x, y), pc.block), n), 0.0);
            cos_sum += cos_theta;
            for (int j = 0; j < 4; ++j) {
                if (weights[j] > 0.0) {
                    total += texelFetch(u_cascade0_tex, probes[j] * pc.block + ivec2(x, y), 0).rgb * (cos_theta * weights[j]);
                }
            }
        }
    }

    vec3 irradiance = (total_w > 1e-5 && cos_sum > 1e-5) ? total / (total_w * cos_sum) : vec3(0.0);
    imageStore(u_irradiance_out, pixel, vec4(irradiance, 1.0));
}
