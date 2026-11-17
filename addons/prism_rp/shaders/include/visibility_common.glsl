#ifndef PRISM_VISIBILITY_COMMON_GLSL
#define PRISM_VISIBILITY_COMMON_GLSL

// ============================================================================
// Prism RP — Phase 4 Visibility Buffer & Two-Phase Meshlet Culling Definitions
// Target: GLSL 4.60 / Vulkan 1.3 (Reverse-Z: 1.0 = near, 0.0 = far)
// ============================================================================

const uint PRISM_MESHLET_MAX_VERTICES  = 64u;
const uint PRISM_MESHLET_MAX_TRIANGLES = 124u;

// Instance flags
const uint PRISM_INSTANCE_FLAG_VISIBLE    = 1u << 0;
const uint PRISM_INSTANCE_FLAG_TWO_SIDED  = 1u << 1;
const uint PRISM_INSTANCE_FLAG_ALPHA_TEST = 1u << 2;
const uint PRISM_INSTANCE_FLAG_CAST_SHADOW= 1u << 3;

// Culling pass modes
const uint PRISM_CULL_PASS_PHASE1_MAIN         = 0u;
const uint PRISM_CULL_PASS_PHASE2_DISOCCLUSION = 1u;

// ----------------------------------------------------------------------------
// 64-byte std430 GpuMeshlet descriptor (matches PrismMeshletData layout)
// ----------------------------------------------------------------------------
struct GpuMeshlet {
    uint  vertex_offset;
    uint  triangle_offset;
    uint  vertex_count;
    uint  triangle_count;
    vec4  bound_sphere;   // xyz = local center, w = radius
    vec4  cone_apex;      // xyz = local apex,   w = unused
    vec4  cone_axis_cut;  // xyz = local unit axis, w = cos(half_angle + pi/2)
};

// ----------------------------------------------------------------------------
// Per-instance scene descriptor (112 bytes, std430)
// ----------------------------------------------------------------------------
struct GpuInstance {
    mat4  world_from_local;
    mat4  prev_world_from_local;
    uint  meshlet_offset;     // Base index into global GpuMeshlet SSBO
    uint  meshlet_count;      // Number of meshlets in this surface
    uint  vertex_base;        // Base offset into global vertex buffer
    uint  index_base;         // Base offset into global micro-triangle buffer
    uint  material_id;        // Material table index for 8x8 tile binning
    uint  flags;              // PRISM_INSTANCE_FLAG_*
    uint  _pad0;
    uint  _pad1;
};

// ----------------------------------------------------------------------------
// Compact work item queued for Phase 2 disocclusion re-testing
// ----------------------------------------------------------------------------
struct FalseNegativeItem {
    uint instance_id;
    uint meshlet_local_id;
};

// ----------------------------------------------------------------------------
// Visible meshlet entry consumed by rasterizer (mesh shader or vertex pull)
// ----------------------------------------------------------------------------
struct VisibleMeshlet {
    uint instance_id;
    uint meshlet_local_id;
};

// ----------------------------------------------------------------------------
// Standard Vulkan indirect command layouts
// ----------------------------------------------------------------------------
struct VkDrawIndexedIndirectCommand {
    uint indexCount;
    uint instanceCount;
    uint firstIndex;
    int  vertexOffset;
    uint firstInstance;
};

struct VkDrawMeshTasksIndirectCommandEXT {
    uint groupCountX;
    uint groupCountY;
    uint groupCountZ;
};

// ----------------------------------------------------------------------------
// Frame & Culling Uniforms (std140)
// ----------------------------------------------------------------------------
struct VisibilityCullUBO {
    mat4  view_proj;
    mat4  prev_view_proj;
    mat4  view;
    vec4  camera_pos_and_near;     // xyz = world camera pos, w = near plane
    vec4  frustum_planes[6];       // xyz = inward unit normal, w = plane distance
    vec4  hiz_viewport_info;       // xy = Hi-Z mip0 size, zw = 1.0 / Hi-Z mip0 size
    uint  hiz_mip_count;
    uint  total_candidate_count;   // Phase 1: total work items; Phase 2: unused (reads counter)
    uint  max_visible_meshlets;
    uint  max_false_negatives;
    uint  cull_pass_mode;          // PRISM_CULL_PASS_PHASE1_MAIN or PHASE2_DISOCCLUSION
    uint  enable_frustum_cull;
    uint  enable_cone_cull;
    uint  enable_hiz_cull;
};

// ============================================================================
// 64-bit Visibility Buffer (R32G32_UINT) Payload Packing & Unpacking
//   R[31:12] = instance_id      (20 bits -> up to 1,048,576 instances)
//   R[11:0]  = meshlet_local_id (12 bits -> up to 4,096 meshlets / surface)
//   G[31:25] = primitive_id     (7 bits  -> 0..123 triangles / meshlet)
//   G[24:13] = bary_lambda1     (12-bit UNORM)
//   G[12:1]  = bary_lambda2     (12-bit UNORM)
//   G[0]     = alpha_test_bit   (1 bit)
// ============================================================================
uvec2 pack_visibility_payload(
    uint instance_id,
    uint meshlet_local_id,
    uint primitive_id,
    vec2 bary_12,
    bool alpha_tested
) {
    uint r = ((instance_id & 0xFFFFFu) << 12u) | (meshlet_local_id & 0xFFFu);
    uint b1 = uint(clamp(round(bary_12.x * 4095.0), 0.0, 4095.0));
    uint b2 = uint(clamp(round(bary_12.y * 4095.0), 0.0, 4095.0));
    uint g = ((primitive_id & 0x7Fu) << 25u)
           | ((b1 & 0xFFFu) << 13u)
           | ((b2 & 0xFFFu) << 1u)
           | (alpha_tested ? 1u : 0u);
    return uvec2(r, g);
}

void unpack_visibility_payload(
    uvec2 payload,
    out uint instance_id,
    out uint meshlet_local_id,
    out uint primitive_id,
    out vec3 barycentrics,
    out bool alpha_tested
) {
    instance_id      = (payload.x >> 12u) & 0xFFFFFu;
    meshlet_local_id = payload.x & 0xFFFu;
    primitive_id     = (payload.y >> 25u) & 0x7Fu;
    float l1         = float((payload.y >> 13u) & 0xFFFu) / 4095.0;
    float l2         = float((payload.y >> 1u)  & 0xFFFu) / 4095.0;
    barycentrics     = vec3(max(0.0, 1.0 - l1 - l2), l1, l2);
    alpha_tested     = (payload.y & 1u) != 0u;
}

// ============================================================================
// Frustum & Normal-Cone Culling Helpers
// ============================================================================
bool sphere_in_frustum(vec3 world_center, float world_radius, vec4 planes[6]) {
    for (int i = 0; i < 6; ++i) {
        if (dot(planes[i].xyz, world_center) + planes[i].w < -world_radius) {
            return false;
        }
    }
    return true;
}

// Returns true if the entire meshlet cluster faces away from camera_pos
bool cone_backface_culled(
    vec3 world_apex,
    vec3 world_cone_axis,
    float cone_cutoff,
    vec3 camera_pos
) {
    // cone_cutoff >= 1.0 indicates a degenerate/omnidirectional cluster (never cull)
    if (cone_cutoff >= 1.0) {
        return false;
    }
    vec3 view_dir = world_apex - camera_pos;
    float dist_sq = dot(view_dir, view_dir);
    if (dist_sq <= 1e-8) {
        return false;
    }
    // Equivalent to dot(normalize(apex - cam), axis) >= cone_cutoff without rsqrt
    float d = dot(view_dir, world_cone_axis);
    return (d > 0.0) && (d * d >= cone_cutoff * cone_cutoff * dist_sq);
}

// ============================================================================
// Reverse-Z Bounding Sphere Projection & Hi-Z Occlusion Test
// Projects world-space sphere into screen UV AABB [min_uv, max_uv] and
// computes conservative nearest Reverse-Z depth (largest Z in [0, 1]).
// Returns false if sphere intersects or lies behind the near plane (always visible).
// ============================================================================
bool project_sphere_reverse_z(
    vec3 world_center,
    float world_radius,
    mat4 view,
    mat4 view_proj,
    float near_plane,
    out vec4 out_uv_rect,
    out float out_nearest_rev_z
) {
    // View-space center (Godot camera looks down -Z)
    vec3 view_center = (view * vec4(world_center, 1.0)).xyz;
    float nearest_view_z = -view_center.z - world_radius;

    // If sphere crosses or is in front of the near clip plane, conservatively accept
    if (nearest_view_z <= near_plane) {
        out_uv_rect = vec4(0.0, 0.0, 1.0, 1.0);
        out_nearest_rev_z = 1.0;
        return false;
    }

    // Build 8 corners of world-space AABB around the sphere for robust NDC bounds
    vec3 min_w = world_center - vec3(world_radius);
    vec3 max_w = world_center + vec3(world_radius);

    vec2 ndc_min = vec2( 1.0);
    vec2 ndc_max = vec2(-1.0);
    float max_rev_z = 0.0;

    for (int i = 0; i < 8; ++i) {
        vec3 corner = vec3(
            (i & 1) != 0 ? max_w.x : min_w.x,
            (i & 2) != 0 ? max_w.y : min_w.y,
            (i & 4) != 0 ? max_w.z : min_w.z
        );
        vec4 clip = view_proj * vec4(corner, 1.0);
        if (clip.w <= 1e-6) {
            out_uv_rect = vec4(0.0, 0.0, 1.0, 1.0);
            out_nearest_rev_z = 1.0;
            return false;
        }
        vec3 ndc = clip.xyz / clip.w;
        ndc_min = min(ndc_min, ndc.xy);
        ndc_max = max(ndc_max, ndc.xy);
        max_rev_z = max(max_rev_z, ndc.z);
    }

    // Vulkan clip/NDC space: X in [-1, 1], Y in [-1, 1] (top-down UV [0, 1])
    vec2 uv_min = clamp(ndc_min * 0.5 + 0.5, vec2(0.0), vec2(1.0));
    vec2 uv_max = clamp(ndc_max * 0.5 + 0.5, vec2(0.0), vec2(1.0));

    out_uv_rect = vec4(uv_min, uv_max);
    out_nearest_rev_z = clamp(max_rev_z, 0.0, 1.0);
    return true;
}

// Samples conservative farthest depth (minimum Reverse-Z) across the sphere's UV rect.
// Returns true if occluded (nearest_sphere_rev_z < farthest_occluder_rev_z).
bool is_hiz_occluded(
    sampler2D hiz_pyramid,
    vec4 uv_rect,
    float nearest_sphere_rev_z,
    vec2 hiz_mip0_size,
    uint hiz_mip_count
) {
    vec2 rect_px = max((uv_rect.zw - uv_rect.xy) * hiz_mip0_size, vec2(1.0));
    float max_dim = max(rect_px.x, rect_px.y);
    float mip = clamp(ceil(log2(max_dim)), 0.0, float(max(hiz_mip_count, 1u) - 1u));

    // Sample all 4 corners of the projected rect at the selected mip
    float d00 = textureLod(hiz_pyramid, vec2(uv_rect.x, uv_rect.y), mip).r;
    float d10 = textureLod(hiz_pyramid, vec2(uv_rect.z, uv_rect.y), mip).r;
    float d01 = textureLod(hiz_pyramid, vec2(uv_rect.x, uv_rect.w), mip).r;
    float d11 = textureLod(hiz_pyramid, vec2(uv_rect.z, uv_rect.w), mip).r;

    // Reverse-Z: smaller value = farther from camera. Occluder blocks sphere only if
    // the FARTHEST depth in the footprint is still closer (greater) than the sphere's NEAREST depth.
    float farthest_occluder_rev_z = min(min(d00, d10), min(d01, d11));
    return nearest_sphere_rev_z < farthest_occluder_rev_z;
}

#endif // PRISM_VISIBILITY_COMMON_GLSL
