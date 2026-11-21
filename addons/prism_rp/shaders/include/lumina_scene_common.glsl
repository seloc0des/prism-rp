#ifndef LUMINA_SCENE_COMMON_GLSL
#define LUMINA_SCENE_COMMON_GLSL

// Lumina's ray-query copy of the scene (LuminaSceneMirror), bound as one uniform set. The including shader enables
// GL_EXT_ray_query and GL_EXT_nonuniform_qualifier.
//
// A TLAS instance's custom index is its LuminaInstance. Its mesh's geometries are its consecutive LuminaSurfaces
// from surface_base, one per triangle surface, in the BLAS's geometry order. Triangles aren't indexed for the BLAS
// (see LuminaSceneMirror::_build_blas): corner c of a surface's triangle p is position corner_base + 3p + c, and
// u_scene_indices at the same place names its vertex in the attribute pool, counted from vertex_base.
//
// Facing: a hit's front face is the side Godot draws (the instance flags see to it), so culling back faces culls
// what Godot culls.

#ifndef LUMINA_SCENE_SET
#define LUMINA_SCENE_SET 1
#endif

#define LUMINA_SCENE_MAX_TEXTURES 1024   // LuminaSceneMirror::kMaxTextures

#define LUMINA_MASK_CAMERA 0x01u   // LuminaSceneMirror::kMaskCamera
#define LUMINA_MASK_SHADOW 0x02u

#define LUMINA_MATERIAL_ALPHA_TEST 1u
#define LUMINA_MATERIAL_SKIP 2u
#define LUMINA_MATERIAL_VERTEX_COLOR 4u
#define LUMINA_MATERIAL_VERTEX_COLOR_SRGB 8u

struct LuminaSurface {
    uint corner_base;
    uint vertex_base;
    uint triangle_count;
    uint material_slot;   // which of the instance's materials
};

struct LuminaInstance {
    uint surface_base;
    uint material_base;
    uint _pad0;
    uint _pad1;
};

struct LuminaMaterial {
    vec4 albedo;          // linear rgb, alpha
    vec4 uv_transform;    // uv1 scale.xy, offset.xy
    uint albedo_texture;  // slot in u_scene_textures; 0 is white
    uint flags;
    float alpha_scissor;
    uint _pad;
};

layout(set = LUMINA_SCENE_SET, binding = 0) uniform accelerationStructureEXT u_scene_tlas;
layout(set = LUMINA_SCENE_SET, binding = 1, std430) restrict readonly buffer LuminaScenePositions { float data[]; } u_scene_positions;
layout(set = LUMINA_SCENE_SET, binding = 2, std430) restrict readonly buffer LuminaSceneIndices { uint data[]; } u_scene_indices;
layout(set = LUMINA_SCENE_SET, binding = 3, std430) restrict readonly buffer LuminaSceneAttributes { uvec4 data[]; } u_scene_attributes;
layout(set = LUMINA_SCENE_SET, binding = 4, std430) restrict readonly buffer LuminaSceneSurfaces { LuminaSurface data[]; } u_scene_surfaces;
layout(set = LUMINA_SCENE_SET, binding = 5, std430) restrict readonly buffer LuminaSceneInstances { LuminaInstance data[]; } u_scene_instances;
layout(set = LUMINA_SCENE_SET, binding = 6, std430) restrict readonly buffer LuminaSceneInstanceMaterials { uint data[]; } u_scene_instance_materials;
layout(set = LUMINA_SCENE_SET, binding = 7, std430) restrict readonly buffer LuminaSceneMaterials { LuminaMaterial data[]; } u_scene_materials;
layout(set = LUMINA_SCENE_SET, binding = 8) uniform sampler u_scene_sampler;
layout(set = LUMINA_SCENE_SET, binding = 9) uniform texture2D u_scene_textures[LUMINA_SCENE_MAX_TEXTURES];

// What a ray needs to know about the point it hit (a candidate or the committed hit).
struct LuminaScenePoint {
    uint material;
    vec2 uv;                 // before the material's uv1 transform
    vec4 color;              // vertex colour as stored
    vec3 geometric_normal;   // world space, unit, on the side Godot's winding calls front
    float world_area;        // twice the triangle's world-space area
    float uv_area;           // twice its area in uv space
};

vec3 lumina_scene_corner(uint corner) {
    return vec3(u_scene_positions.data[corner * 3u], u_scene_positions.data[corner * 3u + 1u], u_scene_positions.data[corner * 3u + 2u]);
}

// Arguments as rayQueryGetIntersection*EXT returns them.
LuminaScenePoint lumina_scene_point(uint instance_index, uint geometry, uint primitive, vec2 barycentrics, mat4x3 object_to_world) {
    LuminaInstance instance = u_scene_instances.data[instance_index];
    LuminaSurface surface = u_scene_surfaces.data[instance.surface_base + geometry];
    uint corner = surface.corner_base + primitive * 3u;
    vec3 w = vec3(1.0 - barycentrics.x - barycentrics.y, barycentrics.x, barycentrics.y);

    vec3 p0 = object_to_world * vec4(lumina_scene_corner(corner), 1.0);
    vec3 p1 = object_to_world * vec4(lumina_scene_corner(corner + 1u), 1.0);
    vec3 p2 = object_to_world * vec4(lumina_scene_corner(corner + 2u), 1.0);
    uvec4 a0 = u_scene_attributes.data[surface.vertex_base + u_scene_indices.data[corner]];
    uvec4 a1 = u_scene_attributes.data[surface.vertex_base + u_scene_indices.data[corner + 1u]];
    uvec4 a2 = u_scene_attributes.data[surface.vertex_base + u_scene_indices.data[corner + 2u]];
    vec2 uv0 = uintBitsToFloat(a0.xy);
    vec2 uv1 = uintBitsToFloat(a1.xy);
    vec2 uv2 = uintBitsToFloat(a2.xy);

    LuminaScenePoint point;
    point.material = u_scene_instance_materials.data[instance.material_base + surface.material_slot];
    point.uv = uv0 * w.x + uv1 * w.y + uv2 * w.z;
    point.color = unpackUnorm4x8(a0.z) * w.x + unpackUnorm4x8(a1.z) * w.y + unpackUnorm4x8(a2.z) * w.z;
    // Godot's front faces are clockwise seen from the front, so with p2 first this points out of the front. A mirroring
    // transform flips the world-space winding, while facing is decided in object space.
    vec3 n = cross(p2 - p0, p1 - p0) * sign(determinant(mat3(object_to_world)));
    point.world_area = length(n);
    point.geometric_normal = n / max(point.world_area, 1e-20);
    vec2 e1 = uv1 - uv0;
    vec2 e2 = uv2 - uv0;
    point.uv_area = abs(e1.x * e2.y - e1.y * e2.x);
    return point;
}

vec3 lumina_scene_srgb_to_linear(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), greaterThan(c, vec3(0.04045)));
}

// Albedo and alpha at a point reached by a ray cone of width cone_width (world units) travelling along dir. The
// mip follows from the cone's footprint on the triangle (ray cones, Akenine-Moller et al. 2021).
vec4 lumina_scene_albedo(LuminaScenePoint point, vec3 dir, float cone_width) {
    LuminaMaterial material = u_scene_materials.data[point.material];
    vec4 albedo = material.albedo;
    if (material.albedo_texture != 0u) {
        uint slot = material.albedo_texture;
        vec2 uv = point.uv * material.uv_transform.xy + material.uv_transform.zw;
        vec2 size = vec2(textureSize(sampler2D(u_scene_textures[nonuniformEXT(slot)], u_scene_sampler), 0));
        float texel_area = point.uv_area * abs(material.uv_transform.x * material.uv_transform.y) * size.x * size.y;
        float lod = 0.5 * log2(max(texel_area, 1e-20) / max(point.world_area, 1e-20)) +
            log2(max(cone_width, 1e-10) / max(abs(dot(point.geometric_normal, dir)), 1e-3));
        albedo *= textureLod(sampler2D(u_scene_textures[nonuniformEXT(slot)], u_scene_sampler), uv, max(lod, 0.0));
    }
    if ((material.flags & LUMINA_MATERIAL_VERTEX_COLOR) != 0u) {
        vec4 color = point.color;
        if ((material.flags & LUMINA_MATERIAL_VERTEX_COLOR_SRGB) != 0u) {
            color.rgb = lumina_scene_srgb_to_linear(color.rgb);
        }
        albedo *= color;
    }
    return albedo;
}

// Whether a candidate hit on a non-opaque instance stops the ray. Opaque materials always do; alpha-tested ones
// where their alpha reaches the scissor threshold; alpha-blended ones never.
bool lumina_scene_candidate_blocks(LuminaScenePoint point, vec3 dir, float cone_width) {
    uint flags = u_scene_materials.data[point.material].flags;
    if ((flags & LUMINA_MATERIAL_SKIP) != 0u) {
        return false;
    }
    if ((flags & LUMINA_MATERIAL_ALPHA_TEST) == 0u) {
        return true;
    }
    return lumina_scene_albedo(point, dir, cone_width).a >= u_scene_materials.data[point.material].alpha_scissor;
}

#endif // LUMINA_SCENE_COMMON_GLSL
