#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require

#include "vis_shading_common.glsl"

// ============================================================================
// Prism RP — Deferred Visibility Buffer Material Shading Pass
// Dispatched indirectly on ShadeDispatch (from vis_tile_classify): (PRISM_SHADE_DISPATCH_WIDTH, rows, 1).
// Each 8x8 workgroup shades the pixels of one material bin within one 8x8 tile, so a wave runs one
// material's code and textures.
// ============================================================================

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std140) uniform ShadeUniformBlock {
    VisibilityShadeUBO u_shade;
};

layout(set = 0, binding = 1) uniform usampler2D u_vis_buffer;

layout(set = 0, binding = 2, std430) readonly buffer MeshletBuffer {
    GpuMeshlet b_meshlets[];
};

layout(set = 0, binding = 3, std430) readonly buffer InstanceBuffer {
    GpuInstance b_instances[];
};

layout(set = 0, binding = 4, std430) readonly buffer GlobalVertexPosBuffer {
    vec4 b_positions[];
};

layout(set = 0, binding = 5, std430) readonly buffer GlobalVertexAttrBuffer {
    GpuVertexAttributes b_attributes[];
};

layout(set = 0, binding = 6, std430) readonly buffer MeshletVertexIndicesBuffer {
    uint b_meshlet_vertices[];
};

layout(set = 0, binding = 7, std430) readonly buffer MeshletTrianglesBuffer {
    uint b_meshlet_triangles[];
};

layout(set = 0, binding = 8, std430) readonly buffer MaterialBuffer {
    GpuPbrMaterial b_materials[];
};

layout(set = 0, binding = 9, std430) readonly buffer MaterialWorkBuffer {
    MaterialTileWorkItem b_work_items[];
};

// Bindless material texture array (albedo / emissive through sRGB views)
layout(set = 0, binding = 10) uniform sampler2D u_textures[];

// Outputs: HDR radiance (linear Rec.709), albedo + roughness, world-space octahedral normal +
// metallic + AO, motion vectors (Godot convention: prev_uv - curr_uv)
layout(set = 0, binding = 11, rgba16f) uniform writeonly image2D out_hdr_radiance;
layout(set = 0, binding = 12, rgba8)   uniform writeonly image2D out_albedo_roughness;
layout(set = 0, binding = 13, rgba16f) uniform writeonly image2D out_normal_metallic;
layout(set = 0, binding = 14, rg16f)   uniform writeonly image2D out_motion_vectors;

layout(set = 0, binding = 15, std430) readonly buffer ShadeDispatchBuffer {
    ShadeDispatch b_dispatch;
};

const float PI = 3.14159265359;

// GGX / Trowbridge-Reitz Normal Distribution Function
float distribution_ggx(float n_dot_h, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float d  = (n_dot_h * n_dot_h) * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-7);
}

// Schlick-GGX Geometry Term
float geometry_smith(float n_dot_v, float n_dot_l, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) * 0.125;
    float gv = n_dot_v / (n_dot_v * (1.0 - k) + k);
    float gl = n_dot_l / (n_dot_l * (1.0 - k) + k);
    return gv * gl;
}

vec3 fresnel_schlick(float cos_theta, vec3 f0) {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

void main() {
    uint work_idx = gl_WorkGroupID.y * PRISM_SHADE_DISPATCH_WIDTH + gl_WorkGroupID.x;
    if (work_idx >= min(b_dispatch.work_item_count, u_shade.viewport_and_limits.w)) {
        return;
    }

    MaterialTileWorkItem work = b_work_items[work_idx];
    ivec2 tile_origin = ivec2(
        int(work.packed_tile_xy & 0xFFFFu) * int(PRISM_VIS_TILE_SIZE),
        int((work.packed_tile_xy >> 16u) & 0xFFFFu) * int(PRISM_VIS_TILE_SIZE)
    );
    ivec2 pixel = tile_origin + ivec2(gl_LocalInvocationID.xy);

    ivec2 viewport = ivec2(u_shade.viewport_and_limits.xy);
    if (pixel.x >= viewport.x || pixel.y >= viewport.y) {
        return;
    }

    uvec2 payload = texelFetch(u_vis_buffer, pixel, 0).rg;
    if (vis_payload_is_empty(payload)) {
        return;
    }

    uint instance_id;
    uint meshlet_local_id;
    uint primitive_id;
    bool alpha_tested;
    unpack_visibility_payload(payload, instance_id, meshlet_local_id, primitive_id, alpha_tested);

    GpuInstance inst = b_instances[instance_id];
    // Only shade pixels in this workgroup's material bin
    if (min(inst.material_id, PRISM_MAX_TILE_MATERIALS - 1u) != work.material_id) {
        return;
    }

    GpuPbrMaterial mat = b_materials[inst.material_id];
    GpuMeshlet m       = b_meshlets[inst.meshlet_offset + meshlet_local_id];

    // 1. Fetch the triangle's corners
    uint packed_tri = b_meshlet_triangles[m.triangle_offset + primitive_id];
    uint g0 = inst.vertex_base + b_meshlet_vertices[m.vertex_offset + ( packed_tri         & 0xFFu)];
    uint g1 = inst.vertex_base + b_meshlet_vertices[m.vertex_offset + ((packed_tri >>  8u) & 0xFFu)];
    uint g2 = inst.vertex_base + b_meshlet_vertices[m.vertex_offset + ((packed_tri >> 16u) & 0xFFu)];

    vec3 p0_local = b_positions[g0].xyz;
    vec3 p1_local = b_positions[g1].xyz;
    vec3 p2_local = b_positions[g2].xyz;

    vec3 p0_world = (inst.world_from_local * vec4(p0_local, 1.0)).xyz;
    vec3 p1_world = (inst.world_from_local * vec4(p1_local, 1.0)).xyz;
    vec3 p2_world = (inst.world_from_local * vec4(p2_local, 1.0)).xyz;

    // 2. Perspective-correct barycentrics and their screen-space derivatives
    vec2 inv_vp    = 1.0 / vec2(viewport);
    vec2 pixel_ndc = (vec2(pixel) + 0.5) * inv_vp * 2.0 - 1.0;

    vec3 bary;
    vec3 ddx_bary;
    vec3 ddy_bary;
    compute_barycentric_gradients(
        u_shade.view_proj * vec4(p0_world, 1.0),
        u_shade.view_proj * vec4(p1_world, 1.0),
        u_shade.view_proj * vec4(p2_world, 1.0),
        pixel_ndc, inv_vp, bary, ddx_bary, ddy_bary); // edge-on: centroid, zero gradients

    // 3. Interpolate attributes and UV gradients
    GpuVertexAttributes a0 = b_attributes[g0];
    GpuVertexAttributes a1 = b_attributes[g1];
    GpuVertexAttributes a2 = b_attributes[g2];

    vec2 uv     = a0.uv.xy * bary.x     + a1.uv.xy * bary.y     + a2.uv.xy * bary.z;
    vec2 ddx_uv = a0.uv.xy * ddx_bary.x + a1.uv.xy * ddx_bary.y + a2.uv.xy * ddx_bary.z;
    vec2 ddy_uv = a0.uv.xy * ddy_bary.x + a1.uv.xy * ddy_bary.y + a2.uv.xy * ddy_bary.z;

    vec3 world_pos      = p0_world * bary.x + p1_world * bary.y + p2_world * bary.z;
    vec3 local_pos      = p0_local * bary.x + p1_local * bary.y + p2_local * bary.z;
    vec3 prev_world_pos = (inst.prev_world_from_local * vec4(local_pos, 1.0)).xyz;

    // 4. Material textures, sampled with the analytic gradients
    vec4 albedo = mat.base_color_factor;
    if (mat.tex_indices.x != PRISM_INVALID_TEX_INDEX) {
        albedo *= textureGrad(u_textures[nonuniformEXT(mat.tex_indices.x)], uv, ddx_uv, ddy_uv);
    }

    float roughness = mat.emissive_and_roughness.a;
    float metallic  = mat.metallic_ao_alpha.x;
    float ao        = 1.0;
    if (mat.tex_indices.z != PRISM_INVALID_TEX_INDEX) {
        vec4 orm = textureGrad(u_textures[nonuniformEXT(mat.tex_indices.z)], uv, ddx_uv, ddy_uv);
        ao        = mix(1.0, orm.r, mat.metallic_ao_alpha.y);
        roughness *= orm.g;
        metallic  *= orm.b;
    }
    roughness = clamp(roughness, 0.045, 1.0);
    metallic  = clamp(metallic, 0.0, 1.0);

    // 5. Shading frame. Mirrored instances (negative determinant) flip the face winding and the
    //    tangent frame's handedness.
    mat3 M3 = mat3(inst.world_from_local);
    float det_sign = prism_det3(M3) < 0.0 ? -1.0 : 1.0;
    vec3 V = normalize(u_shade.camera_pos_and_exposure.xyz - world_pos);

    // Godot front faces are clockwise, so cross(p2 - p0, p1 - p0) points out of the front face.
    bool front_facing = dot(cross(p2_world - p0_world, p1_world - p0_world) * det_sign, V) >= 0.0;

    vec3 N = normalize(prism_normal_matrix(M3) *
        (a0.normal.xyz * bary.x + a1.normal.xyz * bary.y + a2.normal.xyz * bary.z));

    vec3 T = vec3(0.0);
    vec3 B = vec3(0.0);
    bool normal_mapped = mat.tex_indices.y != PRISM_INVALID_TEX_INDEX;
    if (normal_mapped) {
        T = normalize(M3 * (a0.tangent.xyz * bary.x + a1.tangent.xyz * bary.y + a2.tangent.xyz * bary.z));
        T = normalize(T - N * dot(T, N));
        B = cross(N, T) * a0.tangent.w * det_sign;
    }

    // Back face of a double-sided material: flip the normal before normal mapping (as Godot does
    // with gl_FrontFacing), keeping the tangent frame.
    if ((mat.flags & PRISM_MAT_FLAG_DOUBLE_SIDED) != 0u && !front_facing) {
        N = -N;
    }

    if (normal_mapped) {
        vec3 map_n = textureGrad(u_textures[nonuniformEXT(mat.tex_indices.y)], uv, ddx_uv, ddy_uv).xyz * 2.0 - 1.0;
        map_n.xy *= mat.metallic_ao_alpha.w;
        N = normalize(mat3(T, B, N) * map_n);
    }

    // 6. Motion vectors, Godot convention (prev_uv - curr_uv), for Chronos
    vec4 curr_clip = u_shade.view_proj * vec4(world_pos, 1.0);
    vec4 prev_clip = u_shade.prev_view_proj * vec4(prev_world_pos, 1.0);
    vec2 curr_uv   = (curr_clip.xy / max(curr_clip.w, 1e-6)) * 0.5 + 0.5;
    vec2 prev_uv   = (prev_clip.xy / max(prev_clip.w, 1e-6)) * 0.5 + 0.5;
    vec2 velocity  = prev_uv - curr_uv;

    // 7. Directional sun + emissive, linear Rec.709
    vec3 emissive = mat.emissive_and_roughness.rgb;
    if (mat.tex_indices.w != PRISM_INVALID_TEX_INDEX) {
        emissive *= textureGrad(u_textures[nonuniformEXT(mat.tex_indices.w)], uv, ddx_uv, ddy_uv).rgb;
    }

    vec3 radiance = emissive;
    if ((mat.flags & PRISM_MAT_FLAG_UNSHADED) != 0u) {
        radiance += albedo.rgb;
    } else {
        vec3 L = normalize(u_shade.sun_dir_and_intensity.xyz);
        vec3 H = normalize(V + L);

        float n_dot_v = max(dot(N, V), 1e-4);
        float n_dot_l = max(dot(N, L), 0.0);
        float n_dot_h = max(dot(N, H), 0.0);
        float h_dot_v = max(dot(H, V), 0.0);

        vec3 f0 = mix(vec3(0.04), albedo.rgb, metallic);
        vec3 F  = fresnel_schlick(h_dot_v, f0);
        float D = distribution_ggx(n_dot_h, roughness);
        float G = geometry_smith(n_dot_v, n_dot_l, roughness);

        vec3 specular = (D * G * F) / max(4.0 * n_dot_v * n_dot_l, 1e-4);
        vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
        vec3 diffuse = kD * albedo.rgb / PI;

        vec3 sun_radiance = u_shade.sun_color_and_ambient.rgb * u_shade.sun_dir_and_intensity.w;
        radiance += (diffuse + specular) * sun_radiance * n_dot_l;
        radiance += albedo.rgb * u_shade.sun_color_and_ambient.a * ao;
    }

    // 8. G-buffer & HDR outputs
    imageStore(out_hdr_radiance,     pixel, vec4(radiance, 1.0));
    imageStore(out_albedo_roughness, pixel, vec4(albedo.rgb, roughness));
    imageStore(out_normal_metallic,  pixel, vec4(encode_octahedral_normal(N), metallic, ao));
    imageStore(out_motion_vectors,   pixel, vec4(velocity, 0.0, 0.0));
}
