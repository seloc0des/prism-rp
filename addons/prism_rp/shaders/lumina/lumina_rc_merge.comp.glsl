#version 460
#extension GL_GOOGLE_include_directive : require

#include "prism_common.glsl"
#include "lumina_common.glsl"

// Radiance cascade merge, run from the top level down. For each (probe, direction) of level i:
//   merged = interval_i.rgb + interval_i.a * parent
// where `parent` is level i+1 (already merged), averaged over the 2x2 child directions this
// direction owns and interpolated over the 4 nearest parent probes: bilinearly, weighted by how close
// each parent probe's depth is to this probe's. Plain screen-space bilinear weights let a probe behind
// an object inherit light gathered by parent probes in front of it (light leaking past silhouettes);
// if every parent is rejected, the plain bilinear blend is used.

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D u_parent_tex;    // level i+1, merged
layout(rgba16f, set = 0, binding = 1) uniform image2D u_cascade_io;  // level i: interval in, merged out
layout(set = 0, binding = 2) uniform sampler2D u_depth_tex;

layout(push_constant, std430) uniform PushConstants {
    ivec2 probe_count;
    ivec2 parent_probe_count;
    int block;
    int parent_block;          // = 2 * block
    float probe_spacing_px;
    float parent_spacing_px;
    ivec2 internal_resolution;
    vec2 depth_params;
} pc;

const float kParentDepthTolerance = 0.1;   // relative linear-depth difference, as an exponential falloff

float probe_linear_depth(vec2 center_px) {
    ivec2 px = clamp(ivec2(center_px), ivec2(0), pc.internal_resolution - ivec2(1));
    float d = texelFetch(u_depth_tex, px, 0).r;
    return lumina_is_sky(d) ? 0.0 : prism_linearize_depth(d, pc.depth_params);
}

// The 2x2 child directions' average is one bilinear tap at their shared corner (needs a linear sampler).
vec4 parent_direction_average(ivec2 parent_probe, ivec2 dir_texel) {
    ivec2 corner = parent_probe * pc.parent_block + dir_texel * 2 + ivec2(1);
    return textureLod(u_parent_tex, vec2(corner) / vec2(textureSize(u_parent_tex, 0)), 0.0);
}

void main() {
    ivec2 texel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 probe = texel / pc.block;
    ivec2 dir_texel = texel - probe * pc.block;
    if (probe.x >= pc.probe_count.x || probe.y >= pc.probe_count.y) {
        return;
    }

    vec4 interval = imageLoad(u_cascade_io, texel);

    // Probe center in parent-probe coordinates.
    vec2 center_px = (vec2(probe) + 0.5) * pc.probe_spacing_px;
    vec2 parent_coord = center_px / pc.parent_spacing_px - 0.5;
    ivec2 p0 = ivec2(floor(parent_coord));
    vec2 f = parent_coord - vec2(p0);
    ivec2 max_probe = pc.parent_probe_count - ivec2(1);

    float child_z = probe_linear_depth(center_px);
    vec4 plain = vec4(0.0);
    vec4 parent = vec4(0.0);
    float weight_sum = 0.0;
    for (int j = 0; j < 4; ++j) {
        ivec2 o = ivec2(j & 1, j >> 1);
        ivec2 pp = clamp(p0 + o, ivec2(0), max_probe);
        float w = (o.x == 1 ? f.x : 1.0 - f.x) * (o.y == 1 ? f.y : 1.0 - f.y);
        vec4 v = parent_direction_average(pp, dir_texel);
        plain += v * w;
        float parent_z = probe_linear_depth((vec2(pp) + 0.5) * pc.parent_spacing_px);
        float depth_w;
        if (child_z > 0.0 && parent_z > 0.0) {
            depth_w = exp(-abs(parent_z - child_z) / (kParentDepthTolerance * child_z));
        } else {
            depth_w = (child_z > 0.0) == (parent_z > 0.0) ? 1.0 : 0.0;   // sky matches only sky
        }
        parent += v * (w * depth_w);
        weight_sum += w * depth_w;
    }
    parent = weight_sum > 1e-3 ? parent / weight_sum : plain;

    vec3 radiance = interval.rgb + interval.a * parent.rgb;
    float transmittance = interval.a * parent.a;
    imageStore(u_cascade_io, texel, vec4(radiance, transmittance));
}
