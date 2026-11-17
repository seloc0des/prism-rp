#version 460
#extension GL_GOOGLE_include_directive : require

#include "prism_common.glsl"
#include "lumina_common.glsl"

// Per-pixel irradiance from the merged level-0 cascade: cosine-weighted sum over each probe's
// directions, blended over the 4 nearest probes with depth/normal-aware bilinear weights.
// Output is E / pi, i.e. outgoing diffuse radiance for albedo 1.

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
} pc;

vec3 probe_irradiance(ivec2 probe, vec3 n) {
    vec3 sum = vec3(0.0);
    for (int y = 0; y < pc.block; ++y) {
        for (int x = 0; x < pc.block; ++x) {
            vec3 dir = lumina_block_direction(ivec2(x, y), pc.block);
            float cos_theta = max(dot(dir, n), 0.0);
            sum += texelFetch(u_cascade0_tex, probe * pc.block + ivec2(x, y), 0).rgb * cos_theta;
        }
    }
    // Each octahedral texel covers ~4pi/N sr (approximate: octahedral texels are not equal-area).
    // E = sum(L * cos * dw); diffuse radiance for albedo 1 is E / pi.
    float d_omega = 4.0 * 3.14159265359 / float(pc.block * pc.block);
    return sum * d_omega / 3.14159265359;
}

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
    float z = prism_linearize_depth(depth, pc.depth_params);
    vec3 n = lumina_decode_normal(texelFetch(u_normal_roughness_tex, pixel, 0));

    vec2 coord = (vec2(pixel) + 0.5) / pc.probe_spacing_px - 0.5;
    ivec2 p0 = ivec2(floor(coord));
    vec2 f = coord - vec2(p0);
    ivec2 max_probe = pc.probe_count - ivec2(1);

    vec3 total = vec3(0.0);
    float total_w = 0.0;
    for (int j = 0; j < 4; ++j) {
        ivec2 o = ivec2(j & 1, j >> 1);
        ivec2 probe = clamp(p0 + o, ivec2(0), max_probe);
        float w = (o.x == 1 ? f.x : 1.0 - f.x) * (o.y == 1 ? f.y : 1.0 - f.y);

        ivec2 probe_px = clamp(ivec2((vec2(probe) + 0.5) * pc.probe_spacing_px), ivec2(0), pc.internal_resolution - ivec2(1));
        float probe_depth = texelFetch(u_depth_tex, probe_px, 0).r;
        if (lumina_is_sky(probe_depth)) {
            continue;
        }
        float probe_z = prism_linearize_depth(probe_depth, pc.depth_params);
        vec3 probe_n = lumina_decode_normal(texelFetch(u_normal_roughness_tex, probe_px, 0));
        w *= exp(-abs(probe_z - z) / max(0.05 * z, 1e-3));   // reject probes across depth edges
        w *= pow(max(dot(probe_n, n), 0.0), 8.0);            // and across creases
        if (w <= 1e-5) {
            continue;
        }
        total += probe_irradiance(probe, n) * w;
        total_w += w;
    }

    vec3 irradiance = total_w > 1e-5 ? total / total_w : vec3(0.0);
    imageStore(u_irradiance_out, pixel, vec4(irradiance, 1.0));
}
