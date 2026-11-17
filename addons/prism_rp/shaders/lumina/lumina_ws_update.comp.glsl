#version 460
#extension GL_GOOGLE_include_directive : require

#include "prism_common.glsl"
#include "lumina_common.glsl"
#include "lumina_world_common.glsl"

// One invocation per (world probe, direction) atlas texel. Marches a world-space ray through the
// depth buffer. Outcomes:
//   hit on screen           -> blend the hit's radiance into the cache
//   stayed on screen, clear -> blend in "nothing there" (black)
//   left the screen         -> no information: keep the cached value (this is what makes the cache
//                              remember light that is no longer visible)
// rgb = radiance, a = number of samples blended (capped), used for the running-average weight.

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D u_depth_tex;
layout(set = 0, binding = 1) uniform sampler2D u_radiance_tex;
layout(set = 0, binding = 2) uniform sampler2D u_state_tex;
layout(rgba16f, set = 0, binding = 3) uniform image2D u_world_probes;

const float MAX_SAMPLES = 32.0;

void main() {
    int n = wpc.a0.z;
    ivec2 texel = ivec2(gl_GlobalInvocationID.xy);
    if (texel.x >= n * n * WORLD_BLOCK || texel.y >= n * WORLD_BLOCK) {
        return;
    }
    ivec2 probe2d = texel / WORLD_BLOCK;
    ivec2 dir_texel = texel - probe2d * WORLD_BLOCK;
    ivec3 slot = ivec3(probe2d.x % n, probe2d.y, probe2d.x / n);

    vec4 state = texelFetch(u_state_tex, lumina_state_texel(slot, n), 0);
    vec4 cached = state.g > 0.5 ? vec4(0.0) : imageLoad(u_world_probes, texel);
    if (state.b > 0.5) {
        imageStore(u_world_probes, texel, cached);
        return;
    }

    ivec3 cell = lumina_cell_for_slot(slot, wpc.a1.xyz, n);
    vec3 origin = lumina_cell_center(cell, wpc.cam_pos.w);
    vec3 dir = lumina_block_direction(dir_texel, WORLD_BLOCK);

    int steps = int(wpc.misc.w);
    float jitter = prism_interleaved_gradient_noise(vec2(texel), uint(wpc.a1.w));
    bool known = true;      // false once the ray leaves what the depth buffer can tell us
    bool hit = false;
    vec3 radiance = vec3(0.0);
    for (int i = 0; i < steps; ++i) {
        float t = wpc.misc.z * (float(i) + jitter) / float(steps);
        vec3 v = lumina_world_to_view(origin + dir * t);
        if (v.z >= -1e-3) { known = false; break; }
        vec2 uv = lumina_project_to_uv(v, wpc.proj);
        if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) { known = false; break; }
        float d = textureLod(u_depth_tex, uv, 0.0).r;
        if (lumina_is_sky(d)) {
            continue;
        }
        float scene_z = prism_linearize_depth(d, wpc.misc.xy);
        float ray_z = -v.z;
        if (ray_z > scene_z) {
            if (ray_z - scene_z < wpc.cam_right.w * ray_z) {
                hit = true;
                radiance = max(textureLod(u_radiance_tex, uv, 0.0).rgb, vec3(0.0));
            } else {
                known = false;   // passed behind something: what's there is hidden from the camera
            }
            break;
        }
    }

    vec4 result = cached;
    if (hit || known) {
        float count = min(cached.a + 1.0, MAX_SAMPLES);
        float alpha = max(1.0 / count, wpc.cam_up.w);
        result = vec4(mix(cached.rgb, radiance, alpha), count);
    }
    imageStore(u_world_probes, texel, result);
}
