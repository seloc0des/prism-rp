#version 460
#extension GL_GOOGLE_include_directive : require

#include "prism_common.glsl"
#include "lumina_common.glsl"
#include "lumina_world_common.glsl"

// Merge for the TOP screen-space cascade level: instead of ending in nothing, rays that make it
// through the level's interval (transmittance a) pick up the world-space radiance cache at the
// probe's world position, in the matching world direction (trilinear over the 8 nearest cells that lie
// in front of the probe's surface).
// flags bit 0 (debug "world cascade only"): ignore the screen interval, show only the cache.

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D u_depth_tex;
layout(set = 0, binding = 1) uniform sampler2D u_world_probes_tex;
layout(set = 0, binding = 2) uniform sampler2D u_state_tex;
layout(rgba16f, set = 0, binding = 3) uniform image2D u_cascade_io;
layout(set = 0, binding = 4) uniform sampler2D u_normal_roughness_tex;

vec3 sample_world(vec3 p, vec3 world_normal, vec3 world_dir, out float coverage) {
    int n = wpc.a0.z;
    float spacing = wpc.cam_pos.w;
    vec3 coord = p / spacing - 0.5;
    ivec3 c0 = ivec3(floor(coord));
    vec3 f = coord - vec3(c0);
    ivec2 dir_texel = lumina_direction_texel(world_dir, WORLD_BLOCK);

    vec3 sum = vec3(0.0);
    float weight = 0.0;
    float behind = 0.0;   // trilinear weight of cells behind the surface: not missing, just not applicable
    for (int i = 0; i < 8; ++i) {
        ivec3 o = ivec3(i & 1, (i >> 1) & 1, i >> 2);
        ivec3 cell = c0 + o;
        vec3 w3 = mix(vec3(1.0) - f, f, vec3(o));
        float w = w3.x * w3.y * w3.z;
        // Behind the probe's surface, it sees another space (outside the room, inside a pillar). The tolerance
        // keeps cells lying on the surface's plane, where normal quantisation alone would decide the sign.
        if (dot(lumina_cell_center(cell, spacing) - p, world_normal) < -0.01 * spacing) {
            behind += w;
            continue;
        }
        if (any(lessThan(cell, wpc.a1.xyz)) || any(greaterThanEqual(cell, wpc.a1.xyz + n))) {
            continue;   // outside the live grid
        }
        ivec3 slot = lumina_positive_mod(cell, n);
        vec4 state = texelFetch(u_state_tex, lumina_state_texel(slot, n), 0);
        if (state.b > 0.5) {
            continue;   // buried probe: its rays start inside geometry
        }
        vec4 probe = texelFetch(u_world_probes_tex, lumina_world_atlas_texel(slot, dir_texel, n), 0);
        if (probe.a <= 0.0) {
            continue;   // never sampled
        }
        sum += probe.rgb * w;
        weight += w;
    }
    float applicable = 1.0 - behind;
    if (weight <= 0.0 || weight <= 1e-4 * applicable) {
        coverage = 0.0;
        return vec3(0.0);
    }
    coverage = weight / applicable;
    return sum / weight;
}

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

    vec4 interval = imageLoad(u_cascade_io, texel);
    if ((wpc.a1.w & 1) != 0) {
        interval = vec4(0.0, 0.0, 0.0, 1.0);
    }

    ivec2 center_px = clamp(ivec2((vec2(probe) + 0.5) * probe_spacing_px), ivec2(0), res - ivec2(1));
    float depth = textureLod(u_depth_tex, (vec2(center_px) + 0.5) / vec2(res), 0.0).r;
    if (lumina_is_sky(depth) || interval.a <= 0.0) {
        imageStore(u_cascade_io, texel, interval);
        return;
    }

    vec3 view_pos = lumina_view_position((vec2(center_px) + 0.5) / vec2(res), depth, wpc.proj, wpc.misc.xy);
    vec3 world_pos = lumina_view_to_world(view_pos);
    vec3 world_normal = lumina_view_to_world_dir(lumina_decode_normal(textureLod(u_normal_roughness_tex, (vec2(center_px) + 0.5) / vec2(res), 0.0)));
    vec3 world_dir = lumina_view_to_world_dir(lumina_block_direction(dir_texel, block));

    float coverage;
    vec3 far_field = sample_world(world_pos, world_normal, world_dir, coverage);
    imageStore(u_cascade_io, texel, vec4(interval.rgb + interval.a * far_field, interval.a * (1.0 - min(coverage, 1.0))));
}
