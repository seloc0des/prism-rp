#version 460
#extension GL_GOOGLE_include_directive : require

#include "prism_common.glsl"
#include "lumina_common.glsl"
#include "lumina_world_common.glsl"

// Merge for the TOP screen-space cascade level, and for lone probes below it: instead of ending in nothing, rays that make it
// through the level's interval (transmittance a) pick up the world-space radiance cache at the
// probe's world position, in the matching world direction (trilinear over the 8 nearest cells that lie
// in front of the probe's surface).
// flags bit 0 (debug "world cascade only"): ignore the screen interval, show only the cache.
// flags bit 1: a level below the top, where only lone probes take the cache (lumina_parent_support): they trace
// to the edge of the screen like the top level, and the screen merge gives them no parent light. They end at
// transmittance 0. Every other probe is left for the screen merge.

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D u_depth_tex;
layout(set = 0, binding = 1) uniform sampler2D u_world_probes_tex;
layout(set = 0, binding = 2) uniform sampler2D u_state_tex;
layout(rgba16f, set = 0, binding = 3) uniform image2D u_cascade_io;
layout(set = 0, binding = 4) uniform sampler2D u_normal_roughness_tex;

void main() {
    ivec2 res = wpc.a0.xy;
    int block = wpc.a0.w;
    float probe_spacing_px = wpc.misc.z;
    ivec2 probe_count = (res + ivec2(probe_spacing_px) - 1) / ivec2(probe_spacing_px);

    ivec2 texel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 probe = texel / block;
    ivec2 dir_texel = texel - probe * block;
    if (probe.x >= probe_count.x || probe.y >= probe_count.y) {
        return;
    }

    ivec2 center_px = clamp(ivec2((vec2(probe) + 0.5) * probe_spacing_px), ivec2(0), res - ivec2(1));
    float depth = textureLod(u_depth_tex, (vec2(center_px) + 0.5) / vec2(res), 0.0).r;
    bool lone_only = (wpc.a1.w & 2) != 0;
    if (lone_only && (lumina_is_sky(depth) ||
                      lumina_parent_support(u_depth_tex, probe, probe_spacing_px, res, wpc.misc.xy) >= kLuminaMinParentSupport)) {
        return;
    }

    vec4 interval = imageLoad(u_cascade_io, texel);
    if ((wpc.a1.w & 1) != 0) {
        interval = vec4(0.0, 0.0, 0.0, 1.0);
    }
    if (lumina_is_sky(depth) || interval.a <= 0.0) {
        imageStore(u_cascade_io, texel, interval);
        return;
    }

    vec3 view_pos = lumina_view_position((vec2(center_px) + 0.5) / vec2(res), depth, wpc.proj, wpc.misc.xy);
    vec3 world_pos = lumina_view_to_world(view_pos);
    vec3 world_normal = lumina_view_to_world_dir(lumina_decode_normal(textureLod(u_normal_roughness_tex, (vec2(center_px) + 0.5) / vec2(res), 0.0)));
    vec3 world_dir = lumina_view_to_world_dir(lumina_block_direction(dir_texel, block));

    float coverage;
    vec3 far_field = lumina_world_sample(u_world_probes_tex, u_state_tex, world_pos, world_normal,
        lumina_direction_texel(world_dir, WORLD_BLOCK), coverage);
    float transmittance = lone_only ? 0.0 : interval.a * (1.0 - min(coverage, 1.0));
    imageStore(u_cascade_io, texel, vec4(interval.rgb + interval.a * far_field, transmittance));
}
