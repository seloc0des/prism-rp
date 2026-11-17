#ifndef PRISM_VSP_COMMON_GLSL
#define PRISM_VSP_COMMON_GLSL

// ============================================================================
// Prism RP — Phase 4 Virtual Shadow Pages (VSP) Clipmap Definitions
// Target: GLSL 4.60 / Vulkan 1.3
//
// Virtual Clipmap:
//   - 8 toroidal clipmap levels (16x16 virtual pages per level = 2048x2048 virtual texels/level)
//   - Physical Page Size: 128x128 texels
//   - Physical Shadow Atlas: 8192x8192 R32_SFLOAT (64x64 = 4,096 physical pages)
// ============================================================================

const uint PRISM_VSP_CLIPMAP_LEVELS     = 8u;
const uint PRISM_VSP_PAGE_SIZE          = 128u;
const uint PRISM_VSP_VIRTUAL_PAGES_DIM  = 16u;   // 16x16 pages per clipmap level
const uint PRISM_VSP_ATLAS_PAGES_DIM    = 64u;   // 64x64 pages in 8192x8192 physical atlas
const uint PRISM_VSP_MAX_PHYSICAL_PAGES = 4096u; // 64 * 64

// Page Table Entry (R32_UINT in u_page_table_array[level, vx, vy]) bit layout:
//   [31]    = VSP_PTE_VALID     (has valid cached shadow depth)
//   [30]    = VSP_PTE_DIRTY     (allocated, queued for rasterization this frame)
//   [29]    = VSP_PTE_ALLOCATED (owns a physical page index in [11:0])
//   [28]    = VSP_PTE_REQUESTED (touched by a screen pixel this frame)
//   [11:0]  = Physical page index [0 .. 4095]
const uint VSP_PTE_VALID     = 1u << 31;
const uint VSP_PTE_DIRTY     = 1u << 30;
const uint VSP_PTE_ALLOCATED = 1u << 29;
const uint VSP_PTE_REQUESTED = 1u << 28;
const uint VSP_PTE_PAGE_MASK = 0x0FFFu;

// Pass modes for vsp_page_request_and_invalidate.comp.glsl
const uint VSP_PASS_INVALIDATE        = 0u;
const uint VSP_PASS_REQUEST_AND_ALLOC = 1u;

// ----------------------------------------------------------------------------
// Per-clipmap level transform & toroidal scroll state (96 bytes, std140)
// ----------------------------------------------------------------------------
struct VspClipmapLevel {
    mat4  light_view_proj;     // Orthographic light matrix snapped to page texel grid
    vec4  world_origin_scale;  // xyz = snapped world center, w = world_extent (meters)
    ivec4 page_offset_and_pad; // xy = toroidal virtual page wrap offset, zw = delta scroll pages
};

// ----------------------------------------------------------------------------
// Dynamic object AABB for targeted shadow cache invalidation (32 bytes, std430)
// ----------------------------------------------------------------------------
struct VspDynamicDirtyAabb {
    vec4 min_ws_and_pad; // xyz = world-space min corner (union of prev & curr bounds)
    vec4 max_ws_and_pad; // xyz = world-space max corner
};

// ----------------------------------------------------------------------------
// Dirty page work item emitted for shadow rasterization (8 bytes, std430)
// ----------------------------------------------------------------------------
struct VspPageWorkItem {
    uint level_and_vpage; // [31:24] = clipmap_level, [15:8] = vpage_y, [7:0] = vpage_x
    uint physical_page;   // [11:0] = physical page index [0..4095]
};

// ----------------------------------------------------------------------------
// VSP Frame Uniforms (std140)
// ----------------------------------------------------------------------------
struct VspUniformUBO {
    mat4            inv_view_proj;
    mat4            view;
    vec4            camera_pos_and_near;      // xyz = world camera pos, w = near plane
    vec4            sun_dir_and_bias;         // xyz = unit vector toward sun, w = base depth bias
    vec4            pcss_params;              // x = light_radius_uv, y = min_filter_texels, z = max_filter_texels, w = lod_bias
    uvec4           viewport_and_counts;      // xy = render resolution, z = dirty_aabb_count, w = max_dirty_pages
    uint            pass_mode;                // VSP_PASS_INVALIDATE or VSP_PASS_REQUEST_AND_ALLOC
    uint            force_invalidate_all;     // 1 when sun angle changed beyond threshold
    uint            total_meshlet_candidates; // Total meshlets in scene for shadow raster pass
    uint            _pad0;
    VspClipmapLevel levels[PRISM_VSP_CLIPMAP_LEVELS];
};

// ----------------------------------------------------------------------------
// Coordinate Mapping Helpers
// ----------------------------------------------------------------------------

// Select finest clipmap level whose world texel size covers pixel footprint
uint select_vsp_clipmap_level(float view_depth, float base_world_extent, float lod_bias) {
    float desired_extent = max(view_depth * 2.2 * exp2(lod_bias), base_world_extent);
    float raw_level = ceil(log2(max(desired_extent / base_world_extent, 1.0)));
    return uint(clamp(raw_level, 0.0, float(PRISM_VSP_CLIPMAP_LEVELS - 1u)));
}

// Project world position into clipmap level UV [0, 1] and light depth [0, 1].
// Returns false if outside the clipmap level's XY or Z bounds.
bool project_world_to_vsp_level(
    vec3 world_pos,
    VspClipmapLevel lvl,
    out vec2 out_clip_uv,
    out float out_light_z
) {
    vec4 clip = lvl.light_view_proj * vec4(world_pos, 1.0);
    vec3 ndc  = clip.xyz / max(clip.w, 1e-6);

    if (ndc.x < -1.0 || ndc.x > 1.0 ||
        ndc.y < -1.0 || ndc.y > 1.0 ||
        ndc.z <  0.0 || ndc.z > 1.0) {
        out_clip_uv = vec2(0.0);
        out_light_z = 0.0;
        return false;
    }

    out_clip_uv = ndc.xy * 0.5 + 0.5;
    out_light_z = ndc.z;
    return true;
}

// Convert linear clipmap UV [0, 1) to toroidal virtual page coordinate [0 .. 15]
ivec2 clip_uv_to_toroidal_vpage(vec2 clip_uv, ivec2 toroidal_offset) {
    ivec2 linear_page = ivec2(floor(clamp(clip_uv, vec2(0.0), vec2(0.999999)) * float(PRISM_VSP_VIRTUAL_PAGES_DIM)));
    ivec2 wrapped = (linear_page + toroidal_offset) & int(PRISM_VSP_VIRTUAL_PAGES_DIM - 1u);
    return wrapped;
}

// Convert physical page index [0..4095] + intra-page UV [0, 1] to physical atlas pixel/UV
vec2 physical_page_to_atlas_uv(uint physical_page, vec2 intra_page_uv) {
    uint page_x = physical_page & (PRISM_VSP_ATLAS_PAGES_DIM - 1u);
    uint page_y = physical_page / PRISM_VSP_ATLAS_PAGES_DIM;

    // Clamp intra-page UV by 0.5 texel to prevent bilinear bleeding across adjacent physical pages
    float half_texel = 0.5 / float(PRISM_VSP_PAGE_SIZE);
    vec2 safe_uv = clamp(intra_page_uv, vec2(half_texel), vec2(1.0 - half_texel));

    vec2 atlas_page_origin = vec2(float(page_x), float(page_y));
    return (atlas_page_origin + safe_uv) / float(PRISM_VSP_ATLAS_PAGES_DIM);
}

// Ordered float <-> uint conversion so imageAtomicMin() works monotonically on positive [0, 1] depth
uint float_depth_to_ordered_uint(float d) {
    return floatBitsToUint(clamp(d, 0.0, 1.0));
}

float ordered_uint_to_float_depth(uint u) {
    return uintBitsToFloat(u);
}

#endif // PRISM_VSP_COMMON_GLSL
