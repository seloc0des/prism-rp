#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_query : require
#extension GL_EXT_nonuniform_qualifier : require

#include "prism_common.glsl"
#include "lumina_common.glsl"
#include "lumina_scene_common.glsl"

// One camera ray per pixel through Lumina's scene mirror, writing the albedo where it hits: the albedo the screen
// bounce reflects light off, and the debug view "Ray-Traced Albedo", which should look like Godot's unshaded view.
// Pixels whose ray misses keep what the target held. Rays cull back faces and alpha-test as Godot's raster does.
//
// The ray only has to find the surface Godot drew at the pixel, so it looks for it within kDepthWindow of the depth
// buffer's depth there, not from the near plane to the far plane, and sky pixels trace nothing. Where the mirror
// lacks what Godot drew (MultiMesh foliage and particles, so far, and meshes with gi_mode disabled), the ray misses
// instead of taking the albedo of whatever the mirror holds behind it: the ground under the foliage scene's leaves.

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(rgba16f, set = 0, binding = 0) uniform image2D u_color_io;
layout(set = 0, binding = 1) uniform sampler2D u_depth_tex;

// A share of the pixel's depth either side of it, as lumina_ws_update_rt's kOnScreenDepthTolerance.
const float kDepthWindow = 0.02;

layout(push_constant, std430) uniform PushConstants {
    vec4 cam_x;           // world-space camera basis column x, eye.x
    vec4 cam_y;
    vec4 cam_z;
    vec4 proj;            // see lumina_view_position
    ivec2 size;
    vec2 depth_params;    // see prism_linearize_depth
} pc;

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= pc.size.x || pixel.y >= pc.size.y) {
        return;
    }
    float depth = texelFetch(u_depth_tex, pixel, 0).r;
    if (lumina_is_sky(depth)) {
        return;
    }

    vec2 ndc = (vec2(pixel) + 0.5) / vec2(pc.size) * 2.0 - 1.0;
    vec3 view_dir = normalize(vec3((ndc.x + pc.proj.z) / pc.proj.x, (ndc.y + pc.proj.w) / pc.proj.y, -1.0));
    vec3 dir = normalize(mat3(pc.cam_x.xyz, pc.cam_y.xyz, pc.cam_z.xyz) * view_dir);
    vec3 origin = vec3(pc.cam_x.w, pc.cam_y.w, pc.cam_z.w);
    float t = prism_linearize_depth(depth, pc.depth_params) / max(-view_dir.z, 1e-6);
    // One pixel's angle, so a texture's mip matches the raster's.
    float spread = 2.0 / (abs(pc.proj.y) * float(pc.size.y));

    rayQueryEXT query;
    rayQueryInitializeEXT(query, u_scene_tlas, gl_RayFlagsCullBackFacingTrianglesEXT, LUMINA_MASK_CAMERA, origin,
        t * (1.0 - kDepthWindow), dir, t * (1.0 + kDepthWindow));
    while (rayQueryProceedEXT(query)) {
        if (rayQueryGetIntersectionTypeEXT(query, false) != gl_RayQueryCandidateIntersectionTriangleEXT) {
            continue;
        }
        LuminaScenePoint candidate = lumina_scene_point(
            uint(rayQueryGetIntersectionInstanceCustomIndexEXT(query, false)), uint(rayQueryGetIntersectionGeometryIndexEXT(query, false)),
            uint(rayQueryGetIntersectionPrimitiveIndexEXT(query, false)), rayQueryGetIntersectionBarycentricsEXT(query, false),
            rayQueryGetIntersectionObjectToWorldEXT(query, false));
        if (lumina_scene_candidate_blocks(candidate, dir, rayQueryGetIntersectionTEXT(query, false) * spread)) {
            rayQueryConfirmIntersectionEXT(query);
        }
    }
    if (rayQueryGetIntersectionTypeEXT(query, true) != gl_RayQueryCommittedIntersectionTriangleEXT) {
        return;
    }

    LuminaScenePoint hit = lumina_scene_point(
        uint(rayQueryGetIntersectionInstanceCustomIndexEXT(query, true)), uint(rayQueryGetIntersectionGeometryIndexEXT(query, true)),
        uint(rayQueryGetIntersectionPrimitiveIndexEXT(query, true)), rayQueryGetIntersectionBarycentricsEXT(query, true),
        rayQueryGetIntersectionObjectToWorldEXT(query, true));
    vec4 albedo = lumina_scene_albedo(hit, dir, rayQueryGetIntersectionTEXT(query, true) * spread);
    imageStore(u_color_io, pixel, vec4(albedo.rgb, 1.0));
}
