#ifndef PRISM_VISIBILITY_COMMON_GLSL
#define PRISM_VISIBILITY_COMMON_GLSL

// ============================================================================
// Prism RP — Track B visibility buffer & two-phase meshlet culling definitions
// Target: GLSL 4.60 / Vulkan 1.3.
//
// Conventions (match Godot's Vulkan renderer):
//   * Matrices are Godot's depth-corrected projections: reverse-Z (1 = near, 0 = far) and Y flipped
//     for Vulkan, so NDC y = -1 is the top row of the framebuffer and uv = ndc * 0.5 + 0.5.
//   * Front faces are clockwise on screen. In that NDC (+y down) a front face has a positive
//     signed area (vis_ndc_twice_area).
//   * clip.w is the distance in front of the camera (-view-space z).
// ============================================================================

const uint PRISM_MESHLET_MAX_VERTICES  = 64u;
const uint PRISM_MESHLET_MAX_TRIANGLES = 124u;

// Instance flags
const uint PRISM_INSTANCE_FLAG_VISIBLE     = 1u << 0;
const uint PRISM_INSTANCE_FLAG_TWO_SIDED   = 1u << 1;
const uint PRISM_INSTANCE_FLAG_ALPHA_TEST  = 1u << 2; // routed to the masked raster pipelines
const uint PRISM_INSTANCE_FLAG_CAST_SHADOW = 1u << 3;

// Culling pass modes
const uint PRISM_CULL_PASS_PHASE1_MAIN         = 0u;
const uint PRISM_CULL_PASS_PHASE2_DISOCCLUSION = 1u;

// Visible-meshlet lists written by vis_cull_meshlets: entries [0, max) are opaque, entries
// [max, 2 * max) are alpha-tested ("masked"), where max = VisibilityCullUBO.max_visible_meshlets.
const uint PRISM_VIS_LIST_OPAQUE = 0u;
const uint PRISM_VIS_LIST_MASKED = 1u;

// ----------------------------------------------------------------------------
// 64-byte GpuMeshlet (std430), byte-identical to PrismMeshletData's records. vertex_offset and
// triangle_offset index the scene-wide meshlet vertex / triangle buffers: the host rebases
// PrismMeshletData's surface-local offsets when it packs several surfaces into one buffer.
// ----------------------------------------------------------------------------
struct GpuMeshlet {
    uint  vertex_offset;
    uint  triangle_offset;
    uint  vertex_count;
    uint  triangle_count;
    vec4  bound_sphere;   // xyz = local center, w = radius
    vec4  cone_apex;      // xyz = local apex,   w = unused
    vec4  cone_axis_cut;  // xyz = local unit axis, w = cutoff (reject if dot(normalize(apex - eye), axis) >= cutoff)
};

// ----------------------------------------------------------------------------
// Per-instance scene descriptor (160 bytes, std430)
// ----------------------------------------------------------------------------
struct GpuInstance {
    mat4  world_from_local;
    mat4  prev_world_from_local;  // equal to world_from_local on an instance's first frame
    uint  meshlet_offset;         // first GpuMeshlet of this instance's surface
    uint  meshlet_count;          // meshlets in the surface
    uint  vertex_base;            // added to PrismMeshletData's surface vertex indices
    uint  material_id;            // GpuPbrMaterial index; also the 8x8 tile-binning key
    uint  flags;                  // PRISM_INSTANCE_FLAG_*
    uint  _pad0;
    uint  _pad1;
    uint  _pad2;
};

// Compact work item queued for Phase 2 disocclusion re-testing
struct FalseNegativeItem {
    uint instance_id;
    uint meshlet_local_id;
};

// Visible meshlet entry consumed by the rasterizers (mesh shader or vertex pull)
struct VisibleMeshlet {
    uint instance_id;
    uint meshlet_local_id;
};

// ----------------------------------------------------------------------------
// Standard Vulkan indirect command layouts
// ----------------------------------------------------------------------------
struct VkDrawIndirectCommand {
    uint vertexCount;
    uint instanceCount;
    uint firstVertex;
    uint firstInstance;
};

struct VkDrawMeshTasksIndirectCommandEXT {
    uint groupCountX;
    uint groupCountY;
    uint groupCountZ;
};

// ----------------------------------------------------------------------------
// Frame & culling uniforms (std140, 272 bytes)
// ----------------------------------------------------------------------------
struct VisibilityCullUBO {
    mat4  view_proj;
    mat4  prev_view_proj;
    vec4  camera_pos_and_near;     // xyz = world camera pos, w = near plane distance
    vec4  frustum_planes[6];       // xyz = inward unit normal, w = plane distance
    uint  hiz_mip_count;           // mips in the bound Hi-Z pyramid (see is_hiz_occluded)
    uint  total_candidate_count;   // Phase 1: candidate items; Phase 2: unused (reads the counter)
    uint  max_visible_meshlets;    // capacity of EACH visible list (opaque, masked)
    uint  max_false_negatives;
    uint  cull_pass_mode;          // PRISM_CULL_PASS_PHASE1_MAIN or PHASE2_DISOCCLUSION
    uint  enable_frustum_cull;
    uint  enable_cone_cull;
    uint  enable_hiz_cull;
};

// ============================================================================
// 64-bit visibility buffer (R32G32_UINT) payload
//   X       = instance_id
//   Y[31:8] = meshlet_local_id (24 bits: up to 16,777,216 meshlets per surface)
//   Y[7:1]  = primitive_id     (7 bits: 0..123)
//   Y[0]    = alpha-tested
// Barycentrics are not stored: vis_shade_materials rebuilds them exactly from the triangle.
// Clear the buffer to PRISM_VIS_PAYLOAD_EMPTY: its primitive field is 127, which no triangle has.
// ============================================================================
const uvec2 PRISM_VIS_PAYLOAD_EMPTY = uvec2(0xFFFFFFFFu, 0xFFFFFFFFu);

uvec2 pack_visibility_payload(uint instance_id, uint meshlet_local_id, uint primitive_id, bool alpha_tested) {
    return uvec2(instance_id,
                 ((meshlet_local_id & 0xFFFFFFu) << 8u) | ((primitive_id & 0x7Fu) << 1u) | (alpha_tested ? 1u : 0u));
}

bool vis_payload_is_empty(uvec2 payload) {
    return all(equal(payload, PRISM_VIS_PAYLOAD_EMPTY));
}

void unpack_visibility_payload(
    uvec2 payload,
    out uint instance_id,
    out uint meshlet_local_id,
    out uint primitive_id,
    out bool alpha_tested
) {
    instance_id      = payload.x;
    meshlet_local_id = payload.y >> 8u;
    primitive_id     = (payload.y >> 1u) & 0x7Fu;
    alpha_tested     = (payload.y & 1u) != 0u;
}

// ============================================================================
// Transform helpers
// ============================================================================
float prism_det3(mat3 m) {
    return dot(m[0], cross(m[1], m[2]));
}

// Transforms normals like transpose(inverse(m)), up to a positive scale (normalize the result):
// the cofactor matrix, sign-corrected so mirrored transforms (negative determinant) keep normals
// pointing out of the surface. Much cheaper than inverse().
mat3 prism_normal_matrix(mat3 m) {
    mat3 cofactor = mat3(cross(m[1], m[2]), cross(m[2], m[0]), cross(m[0], m[1]));
    return prism_det3(m) < 0.0 ? -cofactor : cofactor;
}

// ============================================================================
// Triangle facing (mesh-shader primitive culling)
// ============================================================================

// Twice the signed area of a triangle in Vulkan NDC (+y down). Positive for a clockwise-on-screen
// triangle, i.e. a Godot front face.
float vis_ndc_twice_area(vec2 ndc0, vec2 ndc1, vec2 ndc2) {
    return (ndc1.x - ndc0.x) * (ndc2.y - ndc0.y) - (ndc1.y - ndc0.y) * (ndc2.x - ndc0.x);
}

// True if the triangle covers no area or, for one-sided geometry, faces away from the camera.
// `mirrored`: the instance transform has a negative determinant, which reverses on-screen winding
// (Godot flips its cull mode for such instances too).
bool vis_triangle_culled(vec2 ndc0, vec2 ndc1, vec2 ndc2, bool two_sided, bool mirrored) {
    float twice_area = vis_ndc_twice_area(ndc0, ndc1, ndc2);
    if (abs(twice_area) < 1e-11) {
        return true;
    }
    if (two_sided) {
        return false;
    }
    return mirrored ? twice_area > 0.0 : twice_area < 0.0;
}

// ============================================================================
// Frustum & normal-cone culling helpers
// ============================================================================
bool sphere_in_frustum(vec3 world_center, float world_radius, vec4 planes[6]) {
    for (int i = 0; i < 6; ++i) {
        if (dot(planes[i].xyz, world_center) + planes[i].w < -world_radius) {
            return false;
        }
    }
    return true;
}

// Returns true if every triangle of the cluster faces away from camera_pos (meshoptimizer's test).
bool cone_backface_culled(
    vec3 world_apex,
    vec3 world_cone_axis,
    float cone_cutoff,
    vec3 camera_pos
) {
    // cone_cutoff >= 1.0 marks a degenerate/omnidirectional cluster (never cull)
    if (cone_cutoff >= 1.0) {
        return false;
    }
    vec3 view_dir = world_apex - camera_pos;
    float dist_sq = dot(view_dir, view_dir);
    if (dist_sq <= 1e-8) {
        return false;
    }
    // dot(normalize(apex - cam), axis) >= cutoff without the square root; conservative for cutoff < 0
    float d = dot(view_dir, world_cone_axis);
    return (d > 0.0) && (d * d >= cone_cutoff * cone_cutoff * dist_sq);
}

// ============================================================================
// Reverse-Z bounding-sphere projection & Hi-Z occlusion test
// ============================================================================

// Projects the box around a world-space sphere to a screen UV rect [xy, zw] and its nearest
// reverse-Z depth. Returns false when the box reaches the near plane or behind the camera: the
// sphere can't be tested and must be treated as visible.
bool project_sphere_reverse_z(
    vec3 world_center,
    float world_radius,
    mat4 view_proj,
    float near_plane,
    out vec4 out_uv_rect,
    out float out_nearest_rev_z
) {
    vec3 min_w = world_center - vec3(world_radius);
    vec3 max_w = world_center + vec3(world_radius);

    vec2 ndc_min = vec2( 1e30);
    vec2 ndc_max = vec2(-1e30);
    float max_rev_z = 0.0;

    for (int i = 0; i < 8; ++i) {
        vec3 corner = vec3(
            (i & 1) != 0 ? max_w.x : min_w.x,
            (i & 2) != 0 ? max_w.y : min_w.y,
            (i & 4) != 0 ? max_w.z : min_w.z
        );
        vec4 clip = view_proj * vec4(corner, 1.0);
        if (clip.w <= near_plane) {
            out_uv_rect = vec4(0.0, 0.0, 1.0, 1.0);
            out_nearest_rev_z = 1.0;
            return false;
        }
        vec3 ndc = clip.xyz / clip.w;
        ndc_min = min(ndc_min, ndc.xy);
        ndc_max = max(ndc_max, ndc.xy);
        max_rev_z = max(max_rev_z, ndc.z);
    }

    out_uv_rect = vec4(clamp(ndc_min * 0.5 + 0.5, vec2(0.0), vec2(1.0)),
                       clamp(ndc_max * 0.5 + 0.5, vec2(0.0), vec2(1.0)));
    out_nearest_rev_z = clamp(max_rev_z, 0.0, 1.0);
    return true;
}

// Conservative Hi-Z occlusion test against the pyramid vis_hi_z_build makes: mip 0 has the depth
// buffer's size, mip L has Vulkan's max(1, size >> L), each texel holds the farthest (minimum
// reverse-Z) depth of the mip-0 pixels it covers, and the last texel of a row or column also covers
// what an odd-sized parent leaves over. So mip-0 pixel p lies in texel min(p >> L, size_L - 1).
// Reads texels directly: the sampler's filter doesn't matter. Returns true only if every pixel
// of the rect is in front of the sphere's nearest point.
bool is_hiz_occluded(sampler2D hiz_pyramid, uint hiz_mip_count, vec4 uv_rect, float nearest_sphere_rev_z) {
    ivec2 size0 = textureSize(hiz_pyramid, 0);
    ivec2 p_min = clamp(ivec2(floor(uv_rect.xy * vec2(size0))), ivec2(0), size0 - 1);
    ivec2 p_max = clamp(ivec2(floor(uv_rect.zw * vec2(size0))), ivec2(0), size0 - 1);

    // Smallest level whose texels are at least as large as the rect: it then spans <= 2x2 texels.
    int extent = max(p_max.x - p_min.x, p_max.y - p_min.y) + 1;
    int level = extent > 1 ? findMSB(extent - 1) + 1 : 0;
    if (level >= int(hiz_mip_count)) {
        return false; // the pyramid stops short of this footprint: can't prove occlusion
    }

    // Vulkan's mip size, computed rather than queried: cheaper, and textureSize() with a lod that
    // differs between invocations returns wrong sizes on lavapipe.
    ivec2 last = max(size0 >> level, ivec2(1)) - 1;
    ivec2 t_min = min(p_min >> level, last);
    ivec2 t_max = min(p_max >> level, last);
    float farthest = min(
        min(texelFetch(hiz_pyramid, t_min, level).r, texelFetch(hiz_pyramid, ivec2(t_max.x, t_min.y), level).r),
        min(texelFetch(hiz_pyramid, ivec2(t_min.x, t_max.y), level).r, texelFetch(hiz_pyramid, t_max, level).r));

    // Reverse-Z: larger is nearer. Occluded only if even the farthest occluder is nearer.
    return nearest_sphere_rev_z < farthest;
}

#endif // PRISM_VISIBILITY_COMMON_GLSL
