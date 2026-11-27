#version 460
#extension GL_GOOGLE_include_directive : require

#include "prism_common.glsl"
#include "lumina_common.glsl"

// Radiance cascade trace: one texel = one (probe, direction) pair of a single cascade level.
// Writes the radiance found inside this level's interval (rgb) and its transmittance (a), averaged over
// sub_rays x sub_rays rays spread across the direction's octahedral cell: a is the fraction of them that
// found nothing in the interval (or left the screen).
//
// Intervals are measured in screen pixels along the ray's projection: level i covers
// [interval_start_px, interval_start_px + interval_length_px) of it, and the top level
// (interval_length_px <= 0) runs on until the ray leaves the screen. Steps are spaced evenly on screen and
// placed on the ray with perspective-correct interpolation. The jitter moves the first step, and the last
// always lands on the interval's end: with steps at (i + jitter) / steps, a ray crossing a surface after
// the last one was missed by its level, and the next level, starting behind the surface, missed it too. (Converting the pixel lengths to world
// units at the probe's own depth gave each probe a reach of about its own distance from the camera,
// so walls and floor near the camera never saw the far side of the room.)
//
// A hit samples the radiance mip chain at the mip whose texels are as wide as its ray's blur cone is at
// that distance, on screen. Each direction stands for a cone of about 4pi / block^2 sr; sampled
// sharper than that, a small bright patch is hit at full strength by one direction or missed by all,
// and the light it throws flickers and gathers into probe-sized blobs. A blurred texel that takes in
// something well in front of the hit, such as the edge of an object the ray passed behind, would hand
// the ray light from a surface it can't see: the floor behind a bright card got the card's lit front.
// The chain keeps each texel's nearest depth, and the hit steps to finer mips until nothing in its texel
// is nearer than kNearerOccluder times the hit's depth.
//
// Near silhouettes that check falls back to sharp mips, and one ray per direction misses a small bright
// patch again. Levels between 0 and the top (and a lone level 0) trace 2 x 2 sub-rays: each covers a quarter
// of the cone, so it needs less blur and falls back less. The top level has directions enough to do without. More sub-rays leak: a partly blocked direction's fractional transmittance lets
// parent probes' light through past objects. (A probe that traces on by itself takes no parent light, so it can use more.)

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D u_depth_tex;
layout(set = 0, binding = 1) uniform sampler2D u_normal_roughness_tex;
layout(set = 0, binding = 2) uniform sampler2D u_radiance_tex;     // current opaque lighting, full mip chain
layout(rgba16f, set = 0, binding = 3) uniform writeonly image2D u_cascade_out;

layout(push_constant, std430) uniform PushConstants {
    ivec2 internal_resolution;
    ivec2 probe_count;
    int block;                 // directions per probe = block * block
    float probe_spacing_px;
    float interval_start_px;   // pixels along the ray's projection on screen
    float interval_length_px;  // <= 0: on to the edge of the screen (top level)
    int steps;
    float thickness;           // occluder thickness, as a fraction of the ray's view depth
    uint frame_index;
    float normal_bias;         // origin push along the normal, fraction of view depth
    vec4 proj;                 // see lumina_view_position
    vec2 depth_params;
    int sub_rays;              // per axis: sub_rays x sub_rays rays per direction
    float _pad;
} pc;

const float kNearerOccluder = 0.9;
const float kMinParentSupport = 0.3;

// How well this probe's parents (the next level up, at twice the spacing) can stand in for it: their
// bilinear weights times the merge's depth weights (lumina_rc_merge.comp.glsl). 1 on a flat surface.
float parent_support(ivec2 probe, float child_z) {
    float parent_spacing = 2.0 * pc.probe_spacing_px;
    ivec2 parent_count = (pc.internal_resolution + ivec2(parent_spacing) - 1) / ivec2(parent_spacing);
    vec2 coord = (vec2(probe) + 0.5) * pc.probe_spacing_px / parent_spacing - 0.5;
    ivec2 p0 = ivec2(floor(coord));
    vec2 f = coord - vec2(p0);
    float support = 0.0;
    for (int j = 0; j < 4; ++j) {
        ivec2 o = ivec2(j & 1, j >> 1);
        ivec2 pp = clamp(p0 + o, ivec2(0), parent_count - ivec2(1));
        ivec2 px = clamp(ivec2((vec2(pp) + 0.5) * parent_spacing), ivec2(0), pc.internal_resolution - ivec2(1));
        float d = texelFetch(u_depth_tex, px, 0).r;
        if (!lumina_is_sky(d)) {
            float w = (o.x == 1 ? f.x : 1.0 - f.x) * (o.y == 1 ? f.y : 1.0 - f.y);
            support += w * lumina_parent_depth_weight(child_z, prism_linearize_depth(d, pc.depth_params));
        }
    }
    return support;
}

// One ray of a direction's interval: radiance found (rgb) and transmittance (a, 0 or 1). `cone_angle` is
// the angular width its hit is blurred over; `jitter` places its steps along the interval.
vec4 trace_ray(vec3 origin, float origin_depth, vec3 dir, float cone_angle, float jitter, float interval_length_px, int steps) {
    vec2 res_f = vec2(pc.internal_resolution);

    // The ray's far end: far enough to leave the view, kept in front of the near plane.
    float near_z = pc.depth_params.x / (1.0 + pc.depth_params.y);
    float far_z = pc.depth_params.x / max(pc.depth_params.y, 1e-6);
    float t_end = far_z;
    if (dir.z > 1e-6) {
        t_end = min(t_end, (origin_depth - 1.01 * near_z) / dir.z);
    }
    if (t_end <= 0.0) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    vec3 ray_end = origin + dir * t_end;

    // Screen-space segment (pixels) with 1/w for perspective-correct positions along it.
    vec2 s0 = lumina_project_to_uv(origin, pc.proj) * res_f;
    vec2 s1 = lumina_project_to_uv(ray_end, pc.proj) * res_f;
    float k0 = 1.0 / origin_depth;
    float k1 = 1.0 / -ray_end.z;
    vec2 delta = s1 - s0;
    float screen_len = length(delta);
    if (screen_len < 0.5) {
        // Pointing straight along the view: the depth buffer has nothing more to say about it.
        return vec4(0.0, 0.0, 0.0, 1.0);
    }

    // Where the segment leaves the screen.
    float u_exit = 1.0;
    for (int axis = 0; axis < 2; ++axis) {
        if (delta[axis] > 1e-6) {
            u_exit = min(u_exit, (res_f[axis] - s0[axis]) / delta[axis]);
        } else if (delta[axis] < -1e-6) {
            u_exit = min(u_exit, -s0[axis] / delta[axis]);
        }
    }
    float u0 = pc.interval_start_px / screen_len;
    float u1 = interval_length_px > 0.0 ? (pc.interval_start_px + interval_length_px) / screen_len : 1.0;
    u1 = min(u1, u_exit);
    if (u0 >= u1) {
        return vec4(0.0, 0.0, 0.0, 1.0);   // left the screen before this interval
    }

    // A hit needs the ray to cross from in front of the depth buffer to behind it between two steps.
    // A ray that is already behind a surface when it reaches it on screen passed behind that object;
    // counting it as a hit gave it the lit front face's colour. Levels above 0 start their interval
    // away from the probe, where the ray may already be behind a surface; any crossing before that
    // point is the previous level's hit, so start from the real state.
    bool was_in_front = true;
    {
        float k = mix(k0, k1, u0);
        vec3 p_start = mix(origin * k0, ray_end * k1, u0) / k;
        vec2 start_uv = mix(s0, s1, u0) / res_f;
        float start_depth = textureLod(u_depth_tex, start_uv, 0.0).r;
        if (!lumina_is_sky(start_depth)) {
            was_in_front = -p_start.z <= prism_linearize_depth(start_depth, pc.depth_params);
        }
    }
    float u_prev = u0;
    float step_div = float(steps - 1) + jitter;
    for (int i = 0; i < steps; ++i) {
        float u = mix(u0, u1, (float(i) + jitter) / max(step_div, 1e-3));
        float ray_z = 1.0 / mix(k0, k1, u);
        vec2 sample_uv = mix(s0, s1, u) / res_f;
        float scene_depth = textureLod(u_depth_tex, sample_uv, 0.0).r;
        if (lumina_is_sky(scene_depth)) {
            was_in_front = true;
            u_prev = u;
            continue;
        }
        float scene_z = prism_linearize_depth(scene_depth, pc.depth_params);
        bool behind = ray_z > scene_z;
        if (behind && was_in_front) {
            // Crossed between u_prev and u: bisect to the crossing, then judge the thickness there.
            // A long step past a wall otherwise lands far behind it and the hit is lost.
            float lo = u_prev;
            float hi = u;
            for (int b = 0; b < 5; ++b) {
                float mid = 0.5 * (lo + hi);
                float mid_depth = textureLod(u_depth_tex, mix(s0, s1, mid) / res_f, 0.0).r;
                bool mid_behind = !lumina_is_sky(mid_depth) &&
                    1.0 / mix(k0, k1, mid) > prism_linearize_depth(mid_depth, pc.depth_params);
                if (mid_behind) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            vec2 hit_uv = mix(s0, s1, hi) / res_f;
            float hit_ray_z = 1.0 / mix(k0, k1, hi);
            float hit_scene_z = prism_linearize_depth(textureLod(u_depth_tex, hit_uv, 0.0).r, pc.depth_params);
            if ((hit_ray_z - hit_scene_z) < pc.thickness * hit_ray_z) {
                // The depth buffer only holds front faces. A ray meeting a surface that faces away from
                // it has reached that object's back, which the colour buffer doesn't show: it is
                // blocked, and the lit front face's colour must not leak through (bright, saturated
                // smears behind objects next to coloured lights).
                vec3 hit_normal = lumina_decode_normal(textureLod(u_normal_roughness_tex, hit_uv, 0.0));
                // The cone's width at the hit, projected to screen pixels.
                vec3 hit_pos = mix(origin * k0, ray_end * k1, hi) * hit_ray_z;
                float footprint_px = length(hit_pos - origin) * cone_angle * abs(pc.proj.y) * 0.5 * res_f.y / hit_ray_z;
                float lod = log2(max(footprint_px, 1.0));
                vec4 texel_value = textureLod(u_radiance_tex, hit_uv, lod);
                while (lod > 0.0 && texel_value.a < kNearerOccluder * hit_scene_z) {
                    lod = max(lod - 1.0, 0.0);
                    texel_value = textureLod(u_radiance_tex, hit_uv, lod);
                }
                vec3 radiance = dot(hit_normal, dir) < 0.0 ? texel_value.rgb : vec3(0.0);
                return vec4(radiance, 0.0);
            }
        }
        was_in_front = !behind;
        u_prev = u;
    }

    return vec4(0.0, 0.0, 0.0, 1.0);
}

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
    float origin_depth = -origin.z;

    // The merge hands this probe's rays its parents' light past the interval's end, but a parent at another
    // depth sees from somewhere else. On a floor seen at a grazing angle, a parent a probe spacing away on
    // screen can be most of a metre away, in front of an object this probe is behind: a floor probe behind
    // a bright card inherited the light of its lit front face. The merge's depth weights rank every parent
    // as poor there, then normalise the least poor to full weight. When no parent can stand in for this
    // probe, it traces on to the edge of the screen by itself, with the top level's steps, and takes no
    // parent light at all (transmittance 0): even rays that leave the screen picked up the parent's hits.
    // Its own rays then carry everything the parents would have, so it traces 4 x 4 sub-rays per
    // direction to match their angular detail (with 2 x 2, level 0's 16 directions came out 30-45% dark
    // behind the card and missed a small bright patch). Few probes qualify: none on smooth surfaces, under
    // 1% of level 0 at 1080p in the card scene, mostly along silhouettes.
    float interval_length_px = pc.interval_length_px;
    int steps = pc.steps;
    int sub_rays = pc.sub_rays;
    bool on_its_own = interval_length_px > 0.0 && parent_support(probe, view_depth) < kMinParentSupport;
    if (on_its_own) {
        interval_length_px = -1.0;
        steps = 2 * pc.steps;
        sub_rays = 4;
    }

    // Each sub-ray blurs over twice its own share of the direction's 4pi / block^2 sr, so neighbouring
    // sub-rays' footprints overlap: with exactly its share a moving hotspot flickers more, and with the
    // whole cone the hotspot loses light. The jitter along the ray changes per texel and frame (SVGF
    // integrates it); golden-ratio offsets keep the sub-rays' steps apart.
    float cone_angle = sqrt(2.0) * sqrt(4.0 * 3.14159265) / float(pc.block * sub_rays);
    float jitter = prism_interleaved_gradient_noise(vec2(texel), pc.frame_index);
    vec4 sum = vec4(0.0);
    for (int sy = 0; sy < sub_rays; ++sy) {
        for (int sx = 0; sx < sub_rays; ++sx) {
            vec2 cell = (vec2(dir_texel) + (vec2(sx, sy) + 0.5) / float(sub_rays)) / float(pc.block);
            float sub_jitter = fract(jitter + 0.618034 * float(sy * sub_rays + sx));
            sum += trace_ray(origin, origin_depth, lumina_oct_decode(cell * 2.0 - 1.0), cone_angle, sub_jitter, interval_length_px, steps);
        }
    }
    sum /= float(sub_rays * sub_rays);
    if (on_its_own) {
        sum.a = 0.0;
    }
    imageStore(u_cascade_out, texel, sum);
}
