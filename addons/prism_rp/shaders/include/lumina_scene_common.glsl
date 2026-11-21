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
//
// Lights (u_scene_lights) are the mirrored world's DirectionalLight3D, OmniLight3D and SpotLight3D nodes that light
// GI (bake mode not Disabled), as Godot's forward renderer applies them: lumina_scene_direct_light.

#ifndef LUMINA_SCENE_SET
#define LUMINA_SCENE_SET 1
#endif

#define LUMINA_SCENE_MAX_TEXTURES 1024   // LuminaSceneMirror::kMaxTextures
#define LUMINA_SCENE_MAX_LIGHTS 64       // LuminaSceneMirror::kMaxLights

#define LUMINA_MASK_CAMERA 0x01u   // LuminaSceneMirror::kMaskCamera
#define LUMINA_MASK_SHADOW 0x02u

#define LUMINA_MATERIAL_ALPHA_TEST 1u
#define LUMINA_MATERIAL_SKIP 2u
#define LUMINA_MATERIAL_VERTEX_COLOR 4u
#define LUMINA_MATERIAL_VERTEX_COLOR_SRGB 8u
#define LUMINA_MATERIAL_EMISSION_MULTIPLY 16u   // Godot's Multiply emission operator; else Add
#define LUMINA_MATERIAL_DOUBLE_SIDED 32u        // cull disabled: both sides are surfaces

#define LUMINA_LIGHT_DIRECTIONAL 0u   // LuminaSceneMirror::kLightDirectional
#define LUMINA_LIGHT_OMNI 1u
#define LUMINA_LIGHT_SPOT 2u

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
    vec4 albedo;             // linear rgb, alpha
    vec4 uv_transform;       // uv1 scale.xy, offset.xy
    vec4 emission;           // linear rgb, energy (0 with emission off)
    uint albedo_texture;     // slot in u_scene_textures; 0 is white
    uint flags;
    float alpha_scissor;
    uint emission_texture;   // slot in u_scene_textures; 0: none, which reads black as Godot's hint_default_black
};

// Colour is linear and already multiplied by energy and indirect energy (negative for a negative light).
struct LuminaLight {
    vec4 position;    // xyz: world position (omni, spot); w: 1 / range
    vec4 direction;   // xyz: unit direction the light shines in (directional, spot); w: distance attenuation exponent
    vec4 color;       // rgb
    vec4 spot;        // x: cos(spot angle), y: 1 / spot angle attenuation (Godot's cone_attenuation)
    uvec4 info;       // x: LUMINA_LIGHT_*, y: 1 if it casts shadows
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
layout(set = LUMINA_SCENE_SET, binding = 10, std430) restrict readonly buffer LuminaSceneLights {
    uvec4 header;   // x: light count
    LuminaLight data[];
} u_scene_lights;

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

// lumina_scene_point for a ray query's candidate (committed = false) or committed hit.
#define LUMINA_SCENE_QUERY_POINT(query, committed) lumina_scene_point( \
    uint(rayQueryGetIntersectionInstanceCustomIndexEXT(query, committed)), uint(rayQueryGetIntersectionGeometryIndexEXT(query, committed)), \
    uint(rayQueryGetIntersectionPrimitiveIndexEXT(query, committed)), rayQueryGetIntersectionBarycentricsEXT(query, committed), \
    rayQueryGetIntersectionObjectToWorldEXT(query, committed))

vec3 lumina_scene_srgb_to_linear(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), greaterThan(c, vec3(0.04045)));
}

// One of the material's textures at a point reached by a ray cone of width cone_width (world units) travelling along
// dir. The mip follows from the cone's footprint on the triangle (ray cones, Akenine-Moller et al. 2021).
vec4 lumina_scene_texture(uint slot, LuminaMaterial material, LuminaScenePoint point, vec3 dir, float cone_width) {
    vec2 uv = point.uv * material.uv_transform.xy + material.uv_transform.zw;
    vec2 size = vec2(textureSize(sampler2D(u_scene_textures[nonuniformEXT(slot)], u_scene_sampler), 0));
    float texel_area = point.uv_area * abs(material.uv_transform.x * material.uv_transform.y) * size.x * size.y;
    float lod = 0.5 * log2(max(texel_area, 1e-20) / max(point.world_area, 1e-20)) +
        log2(max(cone_width, 1e-10) / max(abs(dot(point.geometric_normal, dir)), 1e-3));
    return textureLod(sampler2D(u_scene_textures[nonuniformEXT(slot)], u_scene_sampler), uv, max(lod, 0.0));
}

// Albedo and alpha at a point reached by a ray cone (see lumina_scene_texture).
vec4 lumina_scene_albedo(LuminaScenePoint point, vec3 dir, float cone_width) {
    LuminaMaterial material = u_scene_materials.data[point.material];
    vec4 albedo = material.albedo;
    if (material.albedo_texture != 0u) {
        albedo *= lumina_scene_texture(material.albedo_texture, material, point, dir, cone_width);
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

// Light the point sends out by itself, as Godot's EMISSION: (colour + texture) x energy, or (colour x texture) x energy
// with the Multiply operator.
vec3 lumina_scene_emission(LuminaScenePoint point, vec3 dir, float cone_width) {
    LuminaMaterial material = u_scene_materials.data[point.material];
    if (material.emission.a == 0.0) {
        return vec3(0.0);
    }
    vec3 texel = material.emission_texture != 0u ? lumina_scene_texture(material.emission_texture, material, point, dir, cone_width).rgb : vec3(0.0);
    vec3 color = (material.flags & LUMINA_MATERIAL_EMISSION_MULTIPLY) != 0u ? material.emission.rgb * texel : material.emission.rgb + texel;
    return color * material.emission.a;
}

// Whether anything that casts shadows lies along dir from origin, closer than t_max. Both faces block, and alpha-tested
// materials block where their alpha reaches the scissor, as in Godot's shadow pass. Alpha-blended materials never
// block, depth pre-pass ones included, which Godot's shadow pass draws.
bool lumina_scene_occluded(vec3 origin, vec3 dir, float t_max) {
    rayQueryEXT query;
    rayQueryInitializeEXT(query, u_scene_tlas, gl_RayFlagsTerminateOnFirstHitEXT, LUMINA_MASK_SHADOW, origin, 0.0, dir, t_max);
    while (rayQueryProceedEXT(query)) {
        if (rayQueryGetIntersectionTypeEXT(query, false) == gl_RayQueryCandidateIntersectionTriangleEXT &&
                lumina_scene_candidate_blocks(LUMINA_SCENE_QUERY_POINT(query, false), dir, 0.0)) {
            rayQueryConfirmIntersectionEXT(query);
        }
    }
    return rayQueryGetIntersectionTypeEXT(query, true) != gl_RayQueryCommittedIntersectionNoneEXT;
}

// Godot's get_omni_attenuation: a smooth window to zero at the range, times distance^-decay.
float lumina_scene_distance_attenuation(float distance, float inv_range, float decay) {
    float nd = distance * inv_range;
    nd *= nd;
    nd *= nd;
    nd = max(1.0 - nd, 0.0);
    return nd * nd * pow(max(distance, 1e-4), -decay);
}

// The scene's lights at p on a surface facing normal, as a Lambertian surface turns them into outgoing light: it
// sends out albedo times this. Godot's diffuse term divides by pi and its light energy carries a factor of pi, so
// neither appears. Shadowed lights are tested with a ray from p, lifted off the surface. Negative lights subtract, so
// the sum can be negative: as in Godot, it takes away from the indirect light too, and the caller clamps the total.
vec3 lumina_scene_direct_light(vec3 p, vec3 normal) {
    vec3 origin = p + normal * (1e-3 + 1e-5 * max(abs(p.x), max(abs(p.y), abs(p.z))));
    uint count = min(u_scene_lights.header.x, uint(LUMINA_SCENE_MAX_LIGHTS));
    vec3 sum = vec3(0.0);
    for (uint i = 0u; i < count; ++i) {
        LuminaLight light = u_scene_lights.data[i];
        vec3 to_light = -light.direction.xyz;
        float distance = 1e5;
        float attenuation = 1.0;
        if (light.info.x != LUMINA_LIGHT_DIRECTIONAL) {
            vec3 offset = light.position.xyz - p;
            distance = length(offset);
            to_light = offset / max(distance, 1e-6);
            attenuation = lumina_scene_distance_attenuation(distance, light.position.w, light.direction.w);
            if (light.info.x == LUMINA_LIGHT_SPOT) {
                float cos_angle = max(dot(-to_light, light.direction.xyz), light.spot.x);
                float rim = max(1e-4, (1.0 - cos_angle) / (1.0 - light.spot.x));
                attenuation *= 1.0 - pow(rim, light.spot.y);
            }
        }
        float n_dot_l = dot(normal, to_light);
        if (n_dot_l <= 0.0 || attenuation <= 0.0) {
            continue;
        }
        if (light.info.y != 0u && lumina_scene_occluded(origin, to_light, distance)) {
            continue;
        }
        sum += light.color.rgb * (n_dot_l * attenuation);
    }
    return sum;
}

#endif // LUMINA_SCENE_COMMON_GLSL
