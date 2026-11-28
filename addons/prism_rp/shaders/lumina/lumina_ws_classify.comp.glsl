#version 460
#extension GL_GOOGLE_include_directive : require

#include "prism_common.glsl"
#include "lumina_common.glsl"
#include "lumina_world_common.glsl"

// One invocation per world probe slot of every level (z runs through level 0's slots, then level 1's). Writes the slot's state for this frame:
//   r: id of the live cell now in this slot
//   g: 1 if the slot changed cells since last frame (update must clear its radiance)
//   b: 1 if the probe is visibly buried behind on-screen geometry (update leaves it untouched)
// Run before lumina_ws_update; a separate pass so no update thread can read a half-written id.

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

layout(set = 0, binding = 0) uniform sampler2D u_depth_tex;
layout(rgba32f, set = 0, binding = 1) uniform image2D u_state;

void main() {
    int n = wpc.a0.z;
    ivec3 id3 = ivec3(gl_GlobalInvocationID.xyz);
    if (any(greaterThanEqual(id3.xy, ivec2(n))) || id3.z >= n * WORLD_LEVELS) {
        return;
    }
    int level = id3.z / n;
    ivec3 slot = ivec3(id3.xy, id3.z - level * n);
    ivec2 st = lumina_state_texel(slot, level, n);
    vec4 prev = imageLoad(u_state, st);

    ivec3 cell = lumina_cell_for_slot(slot, lumina_world_base(level, n), n);
    float id = lumina_encode_cell(cell);
    float changed = (prev.r == id && prev.a > 0.5) ? 0.0 : 1.0;   // a: slot has ever been written

    float buried = 0.0;
    vec3 v = lumina_world_to_view(lumina_cell_center(cell, lumina_world_spacing(level)));
    if (v.z < -1e-3) {
        vec2 uv = lumina_project_to_uv(v, wpc.proj);
        if (all(greaterThanEqual(uv, vec2(0.0))) && all(lessThanEqual(uv, vec2(1.0)))) {
            float d = textureLod(u_depth_tex, uv, 0.0).r;
            if (!lumina_is_sky(d)) {
                float surface_z = prism_linearize_depth(d, wpc.misc.xy);
                // Behind the visible surface (by more than depth noise): inside or behind geometry. Every
                // ray from here starts hidden, and the update would read the surface's lit front as a hit
                // in almost every direction, turning the probe into a lamp.
                buried = (-v.z - surface_z) > 0.01 * surface_z ? 1.0 : 0.0;
            }
        }
    }
    imageStore(u_state, st, vec4(id, changed, buried, 1.0));
}
