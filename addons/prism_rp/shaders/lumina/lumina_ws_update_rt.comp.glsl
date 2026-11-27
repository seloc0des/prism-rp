#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_query : require
#extension GL_EXT_nonuniform_qualifier : require

#include "prism_common.glsl"
#include "lumina_common.glsl"
#include "lumina_world_common.glsl"
#include "lumina_scene_common.glsl"

// lumina_ws_update where ray queries work: each (world probe, direction) atlas texel traces a ray through Lumina's
// scene mirror (LuminaSceneMirror) instead of marching the depth buffer, so every direction gets an answer every
// frame, whether the camera sees what is there or not. Outcomes:
//   miss                              -> nothing there: black (Lumina adds no sky light)
//   back of a one-sided surface       -> blocked: black (the ray started inside something or behind it)
//   a surface the camera sees         -> its radiance from the mip chain at the cone's width, as lumina_ws_update
//                                        reads it, plus albedo x the cache's irradiance there unless the
//                                        screen-space bounce already put it in the mips
//   a surface the camera doesn't see  -> albedo x (the scene's lights, shadow rays included, + the cache's
//                                        irradiance there) + emission
// Godot's colour buffer holds only direct light, so both kinds of hit add the cache's irradiance (last frame's, from
// lumina_ws_irradiance): light bounces once more each frame, and surfaces lit only by other surfaces aren't black.
// misc.w scales that term off screen and misc.z on screen: 0 keeps to one bounce, as Gate 2's single-bounce
// reference measures. With the screen-space bounce on, the radiance copy has already added last frame's GI to the
// mips, so misc.z is 0 and on-screen hits don't count that light a second time.
// rgb = radiance, a = number of samples blended (capped), as in lumina_ws_update.
//
// Each texel's ray takes a new direction inside the texel's cone every frame, so the running average covers the cone
// rather than holding whatever its centre line hits.

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D u_depth_tex;
layout(set = 0, binding = 1) uniform sampler2D u_radiance_tex;          // radiance mip chain: rgb, nearest linear depth
layout(set = 0, binding = 2) uniform sampler2D u_state_tex;
layout(rgba16f, set = 0, binding = 3) uniform image2D u_world_probes;
layout(set = 0, binding = 4) uniform sampler2D u_world_irradiance_tex;  // lumina_ws_irradiance's output

const float MAX_SAMPLES = 32.0;
// As lumina_ws_update: twice the direction's share of the sphere.
const float kConeAngle = sqrt(2.0) * sqrt(4.0 * 3.14159265) / float(WORLD_BLOCK);
const float kNearerOccluder = 0.9;
// A hit is the surface the camera sees when the depth buffer there is this close to it, as a share of its depth.
const float kOnScreenDepthTolerance = 0.02;

// The radiance mip chain where the camera sees world point p, which the ray reached after t, if the camera sees it:
// the camera has to be on the side the ray arrived from (normal faces the ray), or the screen holds the surface's
// other side, the lit front of a curtain the ray reached from behind or the inside of a wall it met from outside.
bool screen_radiance(vec3 p, vec3 normal, float t, out vec3 radiance) {
    radiance = vec3(0.0);
    if (dot(normal, wpc.cam_pos.xyz - p) <= 0.0) {
        return false;
    }
    vec3 v = lumina_world_to_view(p);
    if (v.z >= -1e-3) {
        return false;
    }
    vec2 uv = lumina_project_to_uv(v, wpc.proj);
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) {
        return false;
    }
    float d = textureLod(u_depth_tex, uv, 0.0).r;
    if (lumina_is_sky(d)) {
        return false;
    }
    float scene_z = prism_linearize_depth(d, wpc.misc.xy);
    float ray_z = -v.z;
    if (abs(ray_z - scene_z) > kOnScreenDepthTolerance * ray_z) {
        return false;
    }
    float footprint_px = t * kConeAngle * abs(wpc.proj.y) * 0.5 * float(wpc.a0.y) / ray_z;
    float lod = log2(max(footprint_px, 1.0));
    vec4 texel_value = textureLod(u_radiance_tex, uv, lod);
    while (lod > 0.0 && texel_value.a < kNearerOccluder * scene_z) {
        lod = max(lod - 1.0, 0.0);
        texel_value = textureLod(u_radiance_tex, uv, lod);
    }
    radiance = texel_value.rgb;
    return true;
}

// Light leaving the hit back along the ray. normal faces the ray.
vec3 shade(LuminaScenePoint hit, vec3 p, vec3 normal, vec3 dir, float t) {
    float cone_width = t * kConeAngle;
    vec3 albedo = lumina_scene_albedo(hit, dir, cone_width).rgb;
    float coverage;
    vec3 irradiance = lumina_world_sample(u_world_irradiance_tex, u_state_tex, p, normal,
        lumina_direction_texel(normal, WORLD_BLOCK), coverage);
    vec3 on_screen;
    if (screen_radiance(p, normal, t, on_screen)) {
        return on_screen + albedo * irradiance * wpc.misc.z;
    }
    vec3 bounce = irradiance * wpc.misc.w;
    // Godot sums negative lights into the same diffuse light as the indirect light; the total can't go below zero.
    return albedo * max(lumina_scene_direct_light(p, normal) + bounce, vec3(0.0)) + lumina_scene_emission(hit, dir, cone_width);
}

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
    // An R2 sequence over frames, offset per texel so neighbouring texels don't step in lockstep.
    vec2 offset = vec2(prism_interleaved_gradient_noise(vec2(texel), 0u), prism_interleaved_gradient_noise(vec2(texel.yx) + 17.0, 0u));
    vec2 jitter = fract(offset + float(uint(wpc.a1.w) & 1023u) * vec2(0.7548776662, 0.5698402910));
    vec3 dir = lumina_oct_decode((vec2(dir_texel) + jitter) / float(WORLD_BLOCK) * 2.0 - 1.0);
    float t_max = prism_linearize_depth(0.0, wpc.misc.xy);   // as far as the camera sees

    // No face culling: a back face is how the ray finds it started inside or behind something.
    rayQueryEXT query;
    rayQueryInitializeEXT(query, u_scene_tlas, gl_RayFlagsNoneEXT, LUMINA_MASK_CAMERA, origin, 0.0, dir, t_max);
    while (rayQueryProceedEXT(query)) {
        if (rayQueryGetIntersectionTypeEXT(query, false) == gl_RayQueryCandidateIntersectionTriangleEXT &&
                lumina_scene_candidate_blocks(LUMINA_SCENE_QUERY_POINT(query, false), dir, rayQueryGetIntersectionTEXT(query, false) * kConeAngle)) {
            rayQueryConfirmIntersectionEXT(query);
        }
    }

    vec3 radiance = vec3(0.0);
    if (rayQueryGetIntersectionTypeEXT(query, true) == gl_RayQueryCommittedIntersectionTriangleEXT) {
        LuminaScenePoint hit = LUMINA_SCENE_QUERY_POINT(query, true);
        bool front = rayQueryGetIntersectionFrontFaceEXT(query, true);
        if (front || (u_scene_materials.data[hit.material].flags & LUMINA_MATERIAL_DOUBLE_SIDED) != 0u) {
            float t = rayQueryGetIntersectionTEXT(query, true);
            vec3 normal = dot(hit.geometric_normal, dir) < 0.0 ? hit.geometric_normal : -hit.geometric_normal;
            radiance = shade(hit, origin + dir * t, normal, dir, t);
        }
    }
    // The running average never forgets a NaN or an infinity (a degenerate light, half-float overflow upstream).
    if (any(isnan(radiance)) || any(isinf(radiance))) {
        radiance = vec3(0.0);
    }

    float count = min(cached.a + 1.0, MAX_SAMPLES);
    float alpha = max(1.0 / count, wpc.cam_up.w);
    imageStore(u_world_probes, texel, vec4(mix(cached.rgb, radiance, alpha), count));
}
