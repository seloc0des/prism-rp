#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_query : require
#extension GL_EXT_nonuniform_qualifier : require

#include "prism_common.glsl"
#include "lumina_common.glsl"
#include "lumina_scene_common.glsl"

// Debug view "Ray-Traced Albedo": one camera ray per pixel through Lumina's scene mirror, writing the albedo where
// it hits. It shows what the ray-query passes see, and should look like Godot's unshaded view. Pixels whose ray
// misses keep what Godot drew. Rays cull back faces and alpha-test as Godot's raster does.

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(rgba16f, set = 0, binding = 0) uniform image2D u_color_io;

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

    vec2 ndc = (vec2(pixel) + 0.5) / vec2(pc.size) * 2.0 - 1.0;
    vec3 view_dir = normalize(vec3((ndc.x + pc.proj.z) / pc.proj.x, (ndc.y + pc.proj.w) / pc.proj.y, -1.0));
    vec3 dir = normalize(mat3(pc.cam_x.xyz, pc.cam_y.xyz, pc.cam_z.xyz) * view_dir);
    vec3 origin = vec3(pc.cam_x.w, pc.cam_y.w, pc.cam_z.w);
    // From the near plane to the far plane, as the raster draws.
    float near = prism_linearize_depth(1.0, pc.depth_params);
    float far = prism_linearize_depth(0.0, pc.depth_params);
    float per_z = 1.0 / max(-view_dir.z, 1e-6);
    // One pixel's angle, so a texture's mip matches the raster's.
    float spread = 2.0 / (abs(pc.proj.y) * float(pc.size.y));

    rayQueryEXT query;
    rayQueryInitializeEXT(query, u_scene_tlas, gl_RayFlagsCullBackFacingTrianglesEXT, LUMINA_MASK_CAMERA, origin, near * per_z, dir, far * per_z);
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
