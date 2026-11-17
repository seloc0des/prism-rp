#version 460
#extension GL_EXT_nonuniform_qualifier : require

#include "../include/vis_shading_common.glsl"

// ============================================================================
// Prism RP — Deferred Visibility Buffer Material Shading Pass
// Dispatched indirectly: workgroups = (b_indirect_dispatch.x, 1, 1).
// Each 8x8 workgroup evaluates a single MaterialID within a single 8x8 tile,
// giving 100% instruction & texture-cache coherence per wave/subgroup.
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

// Bindless material texture array
layout(set = 0, binding = 10) uniform sampler2D u_textures[];

// Output G-Buffer & HDR targets consumed by Lumina-Cascades GI & Chronos VPU
layout(set = 0, binding = 11, rgba16f) uniform writeonly image2D out_hdr_radiance;
layout(set = 0, binding = 12, rgba8)   uniform writeonly image2D out_albedo_roughness;
layout(set = 0, binding = 13, rgba16f) uniform writeonly image2D out_normal_metallic;
layout(set = 0, binding = 14, rg16f)   uniform writeonly image2D out_motion_vectors;

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
    uint work_idx = gl_WorkGroupID.x;
    if (work_idx >= u_shade.viewport_and_limits.w) {
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
    if ((payload.x | payload.y) == 0u) {
        return;
    }

    uint instance_id;
    uint meshlet_local_id;
    uint primitive_id;
    vec3 stored_bary;
    bool alpha_tested;
    unpack_visibility_payload(
        payload,
        instance_id,
        meshlet_local_id,
        primitive_id,
        stored_bary,
        alpha_tested
    );

    GpuInstance inst = b_instances[instance_id];
    uint target_mat_id = work.material_id;

    // Only shade pixels belonging to this workgroup's assigned material_id
    if (min(inst.material_id, PRISM_MAX_TILE_MATERIALS - 1u) != target_mat_id) {
        return;
    }

    GpuPbrMaterial mat = b_materials[inst.material_id];
    GpuMeshlet m       = b_meshlets[inst.meshlet_offset + meshlet_local_id];

    // 1. Fetch triangle indices and 3 corner vertices
    uint packed_tri = b_meshlet_triangles[m.triangle_offset + primitive_id];
    uint l0 = (packed_tri >>  0u) & 0xFFu;
    uint l1 = (packed_tri >>  8u) & 0xFFu;
    uint l2 = (packed_tri >> 16u) & 0xFFu;

    uint g0 = inst.vertex_base + b_meshlet_vertices[m.vertex_offset + l0];
    uint g1 = inst.vertex_base + b_meshlet_vertices[m.vertex_offset + l1];
    uint g2 = inst.vertex_base + b_meshlet_vertices[m.vertex_offset + l2];

    vec3 p0_local = b_positions[g0].xyz;
    vec3 p1_local = b_positions[g1].xyz;
    vec3 p2_local = b_positions[g2].xyz;

    vec3 p0_world = (inst.world_from_local * vec4(p0_local, 1.0)).xyz;
    vec3 p1_world = (inst.world_from_local * vec4(p1_local, 1.0)).xyz;
    vec3 p2_world = (inst.world_from_local * vec4(p2_local, 1.0)).xyz;

    vec4 clip0 = u_shade.view_proj * vec4(p0_world, 1.0);
    vec4 clip1 = u_shade.view_proj * vec4(p1_world, 1.0);
    vec4 clip2 = u_shade.view_proj * vec4(p2_world, 1.0);

    // 2. Reconstruct perspective-correct barycentrics & analytic screen-space derivatives
    vec2 inv_vp    = 1.0 / vec2(viewport);
    vec2 pixel_uv  = (vec2(pixel) + 0.5) * inv_vp;
    vec2 pixel_ndc = pixel_uv * 2.0 - 1.0;

    vec3 bary;
    vec3 ddx_bary;
    vec3 ddy_bary;
    compute_barycentric_gradients(clip0, clip1, clip2, pixel_ndc, inv_vp, bary, ddx_bary, ddy_bary);

    // Fall back to stored 12-bit barycentrics on grazing/sub-pixel degenerate triangles
    if (dot(bary, bary) <= 0.0 || any(isnan(bary))) {
        bary     = stored_bary;
        ddx_bary = vec3(0.0);
        ddy_bary = vec3(0.0);
    }

    // 3. Interpolate attributes and UV gradients
    GpuVertexAttributes a0 = b_attributes[g0];
    GpuVertexAttributes a1 = b_attributes[g1];
    GpuVertexAttributes a2 = b_attributes[g2];

    vec2 uv0 = a0.uv_and_color.xy;
    vec2 uv1 = a1.uv_and_color.xy;
    vec2 uv2 = a2.uv_and_color.xy;

    vec2 uv     = uv0 * bary.x     + uv1 * bary.y     + uv2 * bary.z;
    vec2 ddx_uv = uv0 * ddx_bary.x + uv1 * ddx_bary.y + uv2 * ddx_bary.z;
    vec2 ddy_uv = uv0 * ddy_bary.x + uv1 * ddy_bary.y + uv2 * ddy_bary.z;

    vec3 world_pos = p0_world * bary.x + p1_world * bary.y + p2_world * bary.z;
    vec3 prev_local_pos = p0_local * bary.x + p1_local * bary.y + p2_local * bary.z;
    vec3 prev_world_pos = (inst.prev_world_from_local * vec4(prev_local_pos, 1.0)).xyz;

    // 4. Sample bindless PBR textures using analytic ray-differential UV gradients
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

    // 5. Reconstruct world normal + normal mapping
    mat3 normal_mat = transpose(inverse(mat3(inst.world_from_local)));
    vec3 local_n = normalize(
        a0.normal_and_uv_x.xyz * bary.x +
        a1.normal_and_uv_x.xyz * bary.y +
        a2.normal_and_uv_x.xyz * bary.z
    );
    vec3 N = normalize(normal_mat * local_n);

    if (mat.tex_indices.y != PRISM_INVALID_TEX_INDEX) {
        vec3 local_t = normalize(
            a0.tangent_and_uv_y.xyz * bary.x +
            a1.tangent_and_uv_y.xyz * bary.y +
            a2.tangent_and_uv_y.xyz * bary.z
        );
        float bitangent_sign = a0.tangent_and_uv_y.w;
        vec3 T = normalize(mat3(inst.world_from_local) * local_t);
        T = normalize(T - N * dot(T, N));
        vec3 B = cross(N, T) * bitangent_sign;

        vec3 map_n = textureGrad(u_textures[nonuniformEXT(mat.tex_indices.y)], uv, ddx_uv, ddy_uv).xyz * 2.0 - 1.0;
        map_n.xy *= mat.metallic_ao_alpha.w;
        N = normalize(mat3(T, B, N) * map_n);
    }

    vec3 V = normalize(u_shade.camera_pos_and_exposure.xyz - world_pos);
    if ((mat.flags & PRISM_MAT_FLAG_DOUBLE_SIDED) != 0u && dot(N, V) < 0.0) {
        N = -N;
    }

    // 6. Compute Godot-sign motion vectors (prev_uv - curr_uv) for Chronos TAAU
    vec4 curr_clip = u_shade.view_proj * vec4(world_pos, 1.0);
    vec4 prev_clip = u_shade.prev_view_proj * vec4(prev_world_pos, 1.0);
    vec2 curr_uv   = (curr_clip.xy / max(curr_clip.w, 1e-6)) * 0.5 + 0.5;
    vec2 prev_uv   = (prev_clip.xy / max(prev_clip.w, 1e-6)) * 0.5 + 0.5;
    vec2 velocity  = prev_uv - curr_uv;

    // 7. Direct Directional Sun + Emissive PBR Evaluation (ACEScg linear)
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

    // 8. Write G-Buffer & HDR Outputs (feeds Lumina-Cascades GI & Chronos VPU)
    imageStore(out_hdr_radiance,     pixel, vec4(radiance, 1.0));
    imageStore(out_albedo_roughness, pixel, vec4(albedo.rgb, roughness));
    imageStore(out_normal_metallic,  pixel, vec4(encode_octahedral_normal(N), metallic, ao));
    imageStore(out_motion_vectors,   pixel, vec4(velocity, 0.0, 0.0));
}
