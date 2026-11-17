#version 460

#include "../include/vis_shading_common.glsl"
#include "../include/vsp_common.glsl"

// ============================================================================
// Prism RP — Virtual Shadow Pages Contact-Hardening PCSS Sampler
// Reads R32G32_UINT Visibility Buffer + D32_SFLOAT depth, computes geometric
// normal bias, performs Vogel-disk blocker search + PCSS against the 8192x8192
// physical shadow atlas, and outputs R8_UNORM directional shadow visibility.
// ============================================================================

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std140) uniform VspUniformBlock {
    VspUniformUBO u_vsp;
};

layout(set = 0, binding = 1) uniform sampler2D u_depth_buffer;
layout(set = 0, binding = 2) uniform usampler2D u_vis_buffer;

layout(set = 0, binding = 3, std430) readonly buffer MeshletBuffer {
    GpuMeshlet b_meshlets[];
};

layout(set = 0, binding = 4, std430) readonly buffer InstanceBuffer {
    GpuInstance b_instances[];
};

layout(set = 0, binding = 5, std430) readonly buffer GlobalVertexPosBuffer {
    vec4 b_positions[];
};

layout(set = 0, binding = 6, std430) readonly buffer MeshletVertexIndicesBuffer {
    uint b_meshlet_vertices[];
};

layout(set = 0, binding = 7, std430) readonly buffer MeshletTrianglesBuffer {
    uint b_meshlet_triangles[];
};

// Page Table Array [16 x 16 x 8 layers]
layout(set = 0, binding = 8) uniform usampler2DArray u_page_table_tex;

// Physical 8192x8192 R32_SFLOAT Shadow Atlas
layout(set = 0, binding = 9) uniform sampler2D u_physical_shadow_atlas_tex;

// Output directional shadow mask [0 = full shadow, 1 = fully lit]
layout(set = 0, binding = 10, r8) uniform writeonly image2D out_shadow_mask;

const uint  PCSS_BLOCKER_SAMPLES = 8u;
const uint  PCSS_FILTER_SAMPLES  = 16u;
const float GOLDEN_ANGLE         = 2.39996323;

float interleaved_gradient_noise(ivec2 pixel) {
    return fract(52.9829189 * fract(dot(vec2(pixel), vec2(0.06711056, 0.00583715))));
}

vec2 vogel_disk_sample(uint idx, uint count, float rotation) {
    float r     = sqrt((float(idx) + 0.5) / float(count));
    float theta = float(idx) * GOLDEN_ANGLE + rotation;
    return vec2(r * cos(theta), r * sin(theta));
}

// Sample physical atlas at clipmap UV within the given clipmap level (falls back to coarser level if unmapped)
bool sample_vsp_atlas_depth(uint start_level, vec2 clip_uv, out float out_atlas_depth) {
    for (uint lvl_idx = start_level; lvl_idx < PRISM_VSP_CLIPMAP_LEVELS; ++lvl_idx) {
        VspClipmapLevel lvl = u_vsp.levels[lvl_idx];
        ivec2 vpage = clip_uv_to_toroidal_vpage(clip_uv, lvl.page_offset_and_pad.xy);
        uint pte = texelFetch(u_page_table_tex, ivec3(vpage, int(lvl_idx)), 0).r;

        if ((pte & VSP_PTE_VALID) != 0u) {
            uint phys_page = pte & VSP_PTE_PAGE_MASK;
            vec2 intra_uv  = fract(clip_uv * float(PRISM_VSP_VIRTUAL_PAGES_DIM));
            vec2 atlas_uv  = physical_page_to_atlas_uv(phys_page, intra_uv);
            out_atlas_depth = textureLod(u_physical_shadow_atlas_tex, atlas_uv, 0.0).r;
            return true;
        }
    }
    out_atlas_depth = 1.0;
    return false;
}

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= int(u_vsp.viewport_and_counts.x) ||
        pixel.y >= int(u_vsp.viewport_and_counts.y)) {
        return;
    }

    float rev_z = texelFetch(u_depth_buffer, pixel, 0).r;
    if (rev_z <= 0.0) {
        imageStore(out_shadow_mask, pixel, vec4(1.0, 0.0, 0.0, 0.0));
        return;
    }

    uvec2 payload = texelFetch(u_vis_buffer, pixel, 0).rg;
    vec2 uv  = (vec2(pixel) + 0.5) / vec2(u_vsp.viewport_and_counts.xy);
    vec2 ndc = uv * 2.0 - 1.0;

    vec4 world_h   = u_vsp.inv_view_proj * vec4(ndc, rev_z, 1.0);
    vec3 world_pos = world_h.xyz / max(world_h.w, 1e-6);

    // Reconstruct geometric face normal from Visibility Buffer triangle for self-shadow bias
    vec3 geom_normal = vec3(0.0, 1.0, 0.0);
    if ((payload.x | payload.y) != 0u) {
        uint instance_id, meshlet_id, prim_id;
        vec3 bary;
        bool alpha_tested;
        unpack_visibility_payload(payload, instance_id, meshlet_id, prim_id, bary, alpha_tested);

        GpuInstance inst = b_instances[instance_id];
        GpuMeshlet m     = b_meshlets[inst.meshlet_offset + meshlet_id];
        uint packed_tri  = b_meshlet_triangles[m.triangle_offset + prim_id];

        uint g0 = inst.vertex_base + b_meshlet_vertices[m.vertex_offset + ((packed_tri >>  0u) & 0xFFu)];
        uint g1 = inst.vertex_base + b_meshlet_vertices[m.vertex_offset + ((packed_tri >>  8u) & 0xFFu)];
        uint g2 = inst.vertex_base + b_meshlet_vertices[m.vertex_offset + ((packed_tri >> 16u) & 0xFFu)];

        vec3 p0 = (inst.world_from_local * vec4(b_positions[g0].xyz, 1.0)).xyz;
        vec3 p1 = (inst.world_from_local * vec4(b_positions[g1].xyz, 1.0)).xyz;
        vec3 p2 = (inst.world_from_local * vec4(b_positions[g2].xyz, 1.0)).xyz;

        // Godot clockwise front-face winding -> cross(p2 - p0, p1 - p0) points outward
        vec3 cross_n = cross(p2 - p0, p1 - p0);
        if (dot(cross_n, cross_n) > 1e-12) {
            geom_normal = normalize(cross_n);
        }
    }

    vec3 L = normalize(u_vsp.sun_dir_and_bias.xyz);
    float n_dot_l = dot(geom_normal, L);
    if (n_dot_l <= 0.0) {
        imageStore(out_shadow_mask, pixel, vec4(0.0, 0.0, 0.0, 0.0));
        return;
    }

    vec3 view_pos   = (u_vsp.view * vec4(world_pos, 1.0)).xyz;
    float view_dist = max(-view_pos.z, u_vsp.camera_pos_and_near.w);
    uint level      = select_vsp_clipmap_level(
        view_dist,
        u_vsp.levels[0].world_origin_scale.w,
        u_vsp.pcss_params.w
    );

    VspClipmapLevel lvl = u_vsp.levels[level];
    float texel_world_size = lvl.world_origin_scale.w / float(PRISM_VSP_VIRTUAL_PAGES_DIM * PRISM_VSP_PAGE_SIZE);

    // Apply normal + slope-scaled world bias before projecting to light space
    float slope_bias = (1.0 - n_dot_l) * 1.5;
    vec3 biased_world_pos = world_pos + geom_normal * (texel_world_size * (1.25 + slope_bias));

    vec2 clip_uv;
    float receiver_z;
    if (!project_world_to_vsp_level(biased_world_pos, lvl, clip_uv, receiver_z)) {
        imageStore(out_shadow_mask, pixel, vec4(1.0, 0.0, 0.0, 0.0));
        return;
    }

    float depth_bias = u_vsp.sun_dir_and_bias.w;
    float rotation   = interleaved_gradient_noise(pixel) * 6.2831853;
    float inv_virtual_res = 1.0 / float(PRISM_VSP_VIRTUAL_PAGES_DIM * PRISM_VSP_PAGE_SIZE);

    // 1. PCSS Blocker Search
    float search_uv = u_vsp.pcss_params.x;
    float blocker_sum = 0.0;
    uint blocker_count = 0u;

    for (uint i = 0u; i < PCSS_BLOCKER_SAMPLES; ++i) {
        vec2 offset = vogel_disk_sample(i, PCSS_BLOCKER_SAMPLES, rotation) * search_uv;
        vec2 sample_uv = clip_uv + offset;
        if (all(greaterThanEqual(sample_uv, vec2(0.0))) && all(lessThanEqual(sample_uv, vec2(1.0)))) {
            float map_z;
            if (sample_vsp_atlas_depth(level, sample_uv, map_z)) {
                if (map_z + depth_bias < receiver_z) {
                    blocker_sum += map_z;
                    blocker_count++;
                }
            }
        }
    }

    if (blocker_count == 0u) {
        imageStore(out_shadow_mask, pixel, vec4(1.0, 0.0, 0.0, 0.0));
        return;
    }

    // 2. Contact-Hardening Penumbra Estimation
    float avg_blocker_z = blocker_sum / float(blocker_count);
    float penumbra_ratio = clamp((receiver_z - avg_blocker_z) / max(avg_blocker_z, 1e-4), 0.0, 1.0);
    float filter_texels = mix(u_vsp.pcss_params.y, u_vsp.pcss_params.z, penumbra_ratio);
    float filter_uv = filter_texels * inv_virtual_res;

    // 3. 16-Tap Vogel-Disk PCF Filtering
    float lit_sum = 0.0;
    for (uint i = 0u; i < PCSS_FILTER_SAMPLES; ++i) {
        vec2 offset = vogel_disk_sample(i, PCSS_FILTER_SAMPLES, rotation) * filter_uv;
        vec2 sample_uv = clamp(clip_uv + offset, vec2(0.0), vec2(1.0));
        float map_z;
        if (sample_vsp_atlas_depth(level, sample_uv, map_z)) {
            lit_sum += (receiver_z <= map_z + depth_bias) ? 1.0 : 0.0;
        } else {
            lit_sum += 1.0;
        }
    }

    float shadow_visibility = lit_sum / float(PCSS_FILTER_SAMPLES);
    imageStore(out_shadow_mask, pixel, vec4(shadow_visibility, 0.0, 0.0, 0.0));
}
