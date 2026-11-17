#version 460
#extension GL_GOOGLE_include_directive : require

#include "prism_common.glsl"
#include "visibility_common.glsl"
#include "vsp_common.glsl"

// ============================================================================
// Prism RP — Virtual Shadow Pages contact-hardening PCSS sampler
// Dispatch (ceil(width / 8), ceil(height / 8), 1). Reads the visibility buffer + depth, offsets
// each receiver along its geometric normal, searches for blockers and filters with Vogel-disk
// taps placed in world space on the plane facing the sun, and writes an R8 shadow mask
// (0 = full shadow, 1 = lit).
//
// Each tap is projected into the pixel's level, or the first coarser level whose page is valid
// (vsp_page_request_and_invalidate requests the footprint's pages and the coarsest level's).
// Depths are compared within the level they came from and converted to meters, so blocker
// distances and the penumbra are consistent across levels: for the sun (orthographic) the
// penumbra is blocker-receiver distance * tan(sun angular radius).
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

// Page table [128 x 128 x 8 levels] and physical atlas (4096x4096 R32_UINT), read with texelFetch
layout(set = 0, binding = 8) uniform usampler2DArray u_page_table_tex;
layout(set = 0, binding = 9) uniform usampler2D u_physical_shadow_atlas_tex;

layout(set = 0, binding = 10, r8) uniform writeonly image2D out_shadow_mask;

const uint  PCSS_BLOCKER_SAMPLES = 8u;
const uint  PCSS_FILTER_SAMPLES  = 16u;
const float GOLDEN_ANGLE         = 2.39996323;

vec2 vogel_disk_sample(uint idx, uint count, float rotation) {
    float r     = sqrt((float(idx) + 0.5) / float(count));
    float theta = float(idx) * GOLDEN_ANGLE + rotation;
    return vec2(r * cos(theta), r * sin(theta));
}

// Shadow-map depth at `world_pos`, from `start_level` or the first coarser level with a valid page
// there. Returns the stored and the receiver's light depth (same level) and that level's light
// depth per meter.
bool vsp_fetch_depth(vec3 world_pos, uint start_level, out float map_z, out float receiver_z, out float depth_per_meter) {
    for (uint level = start_level; level < PRISM_VSP_CLIPMAP_LEVELS; ++level) {
        VspClipmapLevel lvl = u_vsp.levels[level];
        vec2 clip_uv;
        float z;
        if (!project_world_to_vsp_level(world_pos, lvl, clip_uv, z)) {
            continue;
        }
        uint pte = texelFetch(u_page_table_tex, ivec3(clip_uv_to_toroidal_vpage(clip_uv, lvl.page_offset_and_pad.xy), int(level)), 0).r;
        if ((pte & VSP_PTE_VALID) == 0u) {
            continue;
        }
        map_z = ordered_uint_to_float_depth(texelFetch(u_physical_shadow_atlas_tex, vsp_atlas_texel(pte & VSP_PTE_PAGE_MASK, clip_uv), 0).r);
        receiver_z = z;
        depth_per_meter = vsp_depth_per_meter(lvl);
        return true;
    }
    return false;
}

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(pixel, ivec2(u_vsp.viewport_and_counts.xy)))) {
        return;
    }

    float rev_z = texelFetch(u_depth_buffer, pixel, 0).r;
    if (rev_z <= 0.0) {
        imageStore(out_shadow_mask, pixel, vec4(1.0));
        return;
    }

    vec2 ndc       = (vec2(pixel) + 0.5) / vec2(u_vsp.viewport_and_counts.xy) * 2.0 - 1.0;
    vec4 world_h   = u_vsp.inv_view_proj * vec4(ndc, rev_z, 1.0);
    vec3 world_pos = world_h.xyz / world_h.w;
    vec3 to_camera = u_vsp.camera_pos_and_near.xyz - world_pos;

    // Geometric normal of the visible triangle, for the self-shadowing offset
    vec3 geom_normal = vec3(0.0, 1.0, 0.0);
    uvec2 payload = texelFetch(u_vis_buffer, pixel, 0).rg;
    if (!vis_payload_is_empty(payload)) {
        uint instance_id, meshlet_id, prim_id;
        bool alpha_tested;
        unpack_visibility_payload(payload, instance_id, meshlet_id, prim_id, alpha_tested);

        GpuInstance inst = b_instances[instance_id];
        GpuMeshlet m     = b_meshlets[inst.meshlet_offset + meshlet_id];
        uint packed_tri  = b_meshlet_triangles[m.triangle_offset + prim_id];

        uint g0 = inst.vertex_base + b_meshlet_vertices[m.vertex_offset + ( packed_tri         & 0xFFu)];
        uint g1 = inst.vertex_base + b_meshlet_vertices[m.vertex_offset + ((packed_tri >>  8u) & 0xFFu)];
        uint g2 = inst.vertex_base + b_meshlet_vertices[m.vertex_offset + ((packed_tri >> 16u) & 0xFFu)];

        vec3 p0 = (inst.world_from_local * vec4(b_positions[g0].xyz, 1.0)).xyz;
        vec3 p1 = (inst.world_from_local * vec4(b_positions[g1].xyz, 1.0)).xyz;
        vec3 p2 = (inst.world_from_local * vec4(b_positions[g2].xyz, 1.0)).xyz;

        // Godot front faces are clockwise, so cross(p2 - p0, p1 - p0) points out of the front face;
        // a mirrored transform reverses it.
        vec3 cross_n = cross(p2 - p0, p1 - p0) * (prism_det3(mat3(inst.world_from_local)) < 0.0 ? -1.0 : 1.0);
        if (dot(cross_n, cross_n) > 1e-20) {
            geom_normal = normalize(cross_n);
            // The back of a two-sided surface is lit from its own side
            if ((inst.flags & PRISM_INSTANCE_FLAG_TWO_SIDED) != 0u && dot(geom_normal, to_camera) < 0.0) {
                geom_normal = -geom_normal;
            }
        }
    }

    vec3 L = normalize(u_vsp.sun_dir_and_bias.xyz);
    float n_dot_l = dot(geom_normal, L);
    if (n_dot_l <= 0.0) {
        imageStore(out_shadow_mask, pixel, vec4(0.0)); // faces away from the sun
        return;
    }

    // Level: the selected one, or the first coarser level containing the point (as mode 1 does)
    float view_dist = max(-(u_vsp.view * vec4(world_pos, 1.0)).z, u_vsp.camera_pos_and_near.w);
    uint level = select_vsp_clipmap_level(view_dist, u_vsp.levels[0].world_origin_scale.w, u_vsp.pcss_params.w);
    vec2 clip_uv;
    float light_z;
    while (!project_world_to_vsp_level(world_pos, u_vsp.levels[level], clip_uv, light_z)) {
        if (++level >= PRISM_VSP_CLIPMAP_LEVELS) {
            imageStore(out_shadow_mask, pixel, vec4(1.0)); // beyond the shadow range
            return;
        }
    }

    float texel_world = u_vsp.levels[level].world_origin_scale.w / float(PRISM_VSP_VIRTUAL_RES);
    vec3 receiver = world_pos + geom_normal * (texel_world * (1.25 + 1.5 * (1.0 - n_dot_l)));

    // Tap basis on the plane facing the sun
    vec3 tangent   = normalize(cross(L, abs(L.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
    vec3 bitangent = cross(L, tangent);
    float rotation = prism_interleaved_gradient_noise(vec2(pixel), u_vsp.frame_index) * 6.2831853;
    float bias_m   = u_vsp.sun_dir_and_bias.w;
    float min_radius = u_vsp.pcss_params.y * texel_world;
    float max_radius = u_vsp.pcss_params.z * texel_world;

    // 1. Blocker search over the largest filter footprint
    float blocker_dist_sum = 0.0;
    uint blocker_count = 0u;
    for (uint i = 0u; i < PCSS_BLOCKER_SAMPLES; ++i) {
        vec2 o = vogel_disk_sample(i, PCSS_BLOCKER_SAMPLES, rotation) * max_radius;
        float map_z, recv_z, dz_per_m;
        if (vsp_fetch_depth(receiver + tangent * o.x + bitangent * o.y, level, map_z, recv_z, dz_per_m)) {
            float dist_m = (recv_z - map_z) / dz_per_m;
            if (dist_m > bias_m) {
                blocker_dist_sum += dist_m;
                blocker_count++;
            }
        }
    }
    if (blocker_count == 0u) {
        imageStore(out_shadow_mask, pixel, vec4(1.0));
        return;
    }

    // 2. Penumbra from the average blocker distance (orthographic sun)
    float avg_blocker_dist = blocker_dist_sum / float(blocker_count);
    float filter_radius = clamp(avg_blocker_dist * u_vsp.pcss_params.x, min_radius, max_radius);

    // 3. Vogel-disk PCF
    float lit_sum = 0.0;
    for (uint i = 0u; i < PCSS_FILTER_SAMPLES; ++i) {
        vec2 o = vogel_disk_sample(i, PCSS_FILTER_SAMPLES, rotation) * filter_radius;
        float map_z, recv_z, dz_per_m;
        if (vsp_fetch_depth(receiver + tangent * o.x + bitangent * o.y, level, map_z, recv_z, dz_per_m)) {
            lit_sum += ((recv_z - map_z) / dz_per_m <= bias_m) ? 1.0 : 0.0;
        } else {
            lit_sum += 1.0; // no shadow data anywhere for this tap
        }
    }
    imageStore(out_shadow_mask, pixel, vec4(lit_sum / float(PCSS_FILTER_SAMPLES)));
}
