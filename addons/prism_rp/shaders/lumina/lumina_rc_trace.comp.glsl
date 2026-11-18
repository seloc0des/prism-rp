#version 460
#extension GL_GOOGLE_include_directive : require

#include "prism_common.glsl"
#include "lumina_common.glsl"

// Radiance cascade trace: one texel = one (probe, direction) pair of a single cascade level.
// Writes the radiance found inside this level's interval (rgb) and its transmittance (a):
// a = 0 when the interval hit geometry, 1 when it was clear (or left the screen).

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D u_depth_tex;
layout(set = 0, binding = 1) uniform sampler2D u_normal_roughness_tex;
layout(set = 0, binding = 2) uniform sampler2D u_radiance_tex;     // current opaque lighting
layout(rgba16f, set = 0, binding = 3) uniform writeonly image2D u_cascade_out;

layout(push_constant, std430) uniform PushConstants {
    ivec2 internal_resolution;
    ivec2 probe_count;
    int block;                 // directions per probe = block * block
    float probe_spacing_px;
    float interval_start_px;   // screen-space length at the probe's depth
    float interval_length_px;
    int steps;
    float thickness;           // occluder thickness, as a fraction of the ray's view depth
    uint frame_index;
    float normal_bias;         // origin push along the normal, fraction of view depth
    vec4 proj;                 // see lumina_view_position
    vec2 depth_params;
    vec2 _pad;
} pc;

void main() {
    ivec2 texel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 probe = texel / pc.block;
    ivec2 dir_texel = texel - probe * pc.block;
    if (probe.x >= pc.probe_count.x || probe.y >= pc.probe_count.y) {
        return;
    }

    ivec2 center_px = clamp(ivec2((vec2(probe) + 0.5) * pc.probe_spacing_px), ivec2(0), pc.internal_resolution - ivec2(1));
    float depth = texelFetch(u_depth_tex, center_px, 0).r;
    if (lumina_is_sky(depth)) {
        imageStore(u_cascade_out, texel, vec4(0.0, 0.0, 0.0, 1.0));
        return;
    }

    vec2 res_f = vec2(pc.internal_resolution);
    vec2 uv = (vec2(center_px) + 0.5) / res_f;
    vec3 origin = lumina_view_position(uv, depth, pc.proj, pc.depth_params);
    vec3 normal = lumina_decode_normal(texelFetch(u_normal_roughness_tex, center_px, 0));
    float view_depth = -origin.z;
    origin += normal * (pc.normal_bias * view_depth);

    vec3 dir = lumina_block_direction(dir_texel, pc.block);

    // World length of one screen pixel at this depth, so intervals stay a fixed screen size.
    float world_per_px = 2.0 * view_depth / (abs(pc.proj.y) * res_f.y);
    float t0 = pc.interval_start_px * world_per_px;
    float t1 = (pc.interval_start_px + pc.interval_length_px) * world_per_px;

    // Per-texel jitter along the ray; SVGF integrates it over frames.
    float jitter = prism_interleaved_gradient_noise(vec2(texel), pc.frame_index);

    // A hit needs the ray to cross from in front of the depth buffer to behind it between two steps.
    // A ray that is already behind a surface when it reaches it on screen passed behind that object;
    // counting it as a hit (it lies within the thickness) gave it the lit front face's colour.
    // Levels above 0 start their interval away from the probe, where the ray may already be behind a
    // surface; any crossing before that point is the previous level's hit, so start from the real state.
    bool was_in_front = true;
    vec3 p_start = origin + dir * t0;
    if (p_start.z < -1e-3) {
        vec2 start_uv = lumina_project_to_uv(p_start, pc.proj);
        if (all(greaterThanEqual(start_uv, vec2(0.0))) && all(lessThanEqual(start_uv, vec2(1.0)))) {
            float start_depth = textureLod(u_depth_tex, start_uv, 0.0).r;
            if (!lumina_is_sky(start_depth)) {
                was_in_front = -p_start.z <= prism_linearize_depth(start_depth, pc.depth_params);
            }
        }
    }
    for (int i = 0; i < pc.steps; ++i) {
        float t = mix(t0, t1, (float(i) + jitter) / float(pc.steps));
        vec3 p = origin + dir * t;
        if (p.z >= -1e-3) {
            break;  // went behind the camera: treat as clear
        }
        vec2 sample_uv = lumina_project_to_uv(p, pc.proj);
        if (any(lessThan(sample_uv, vec2(0.0))) || any(greaterThan(sample_uv, vec2(1.0)))) {
            break;  // left the screen: no information, treat as clear
        }
        float scene_depth = textureLod(u_depth_tex, sample_uv, 0.0).r;
        if (lumina_is_sky(scene_depth)) {
            was_in_front = true;
            continue;
        }
        float scene_z = prism_linearize_depth(scene_depth, pc.depth_params);
        float ray_z = -p.z;
        bool behind = ray_z > scene_z;
        if (behind && was_in_front && (ray_z - scene_z) < pc.thickness * ray_z) {
            // The depth buffer only holds front faces. A ray meeting a surface that faces away from it
            // has reached that object's back, which the colour buffer doesn't show: it is blocked, and
            // the lit front face's colour must not leak through (bright, saturated smears behind
            // objects next to coloured lights).
            vec3 hit_normal = lumina_decode_normal(textureLod(u_normal_roughness_tex, sample_uv, 0.0));
            vec3 radiance = dot(hit_normal, dir) < 0.0 ? max(textureLod(u_radiance_tex, sample_uv, 0.0).rgb, vec3(0.0)) : vec3(0.0);
            imageStore(u_cascade_out, texel, vec4(radiance, 0.0));
            return;
        }
        was_in_front = !behind;
    }

    imageStore(u_cascade_out, texel, vec4(0.0, 0.0, 0.0, 1.0));
}
