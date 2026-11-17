#ifndef PRISM_VSP_COMMON_GLSL
#define PRISM_VSP_COMMON_GLSL

// ============================================================================
// Prism RP — Track B Virtual Shadow Pages (VSP) for the directional light
// Target: GLSL 4.60 / Vulkan 1.3
//
// Virtual: 8 toroidal clipmap levels, each 128x128 pages of 128x128 texels (16384^2 virtual texels).
// Physical: one 4096x4096 R32_UINT atlas = 32x32 pages of 128^2 = 1024 pages, 64 MiB (Gate 4's cap).
// Pages are allocated on demand from a free stack and freed after `evict_after_frames` frames unused.
//
// Frame order:
//   1. vsp_page_request_and_invalidate, mode 0 (VSP_PASS_INVALIDATE), over every virtual page:
//      ages and evicts pages, invalidates scrolled / dirty-AABB / sun-change pages.
//   2. vsp_page_request_and_invalidate, mode 1 (VSP_PASS_REQUEST_AND_ALLOC), over screen pixels:
//      marks the pages shading will read, allocates them, queues the ones to (re)render.
//   3. vsp_clear_pages: clears each queued page (indirect: VspFrameState.clear_dispatch).
//   4. vsp_cull_meshlets: pairs each shadow-casting meshlet with the queued pages it overlaps.
//   5. vsp_raster_pages: rasterizes each (meshlet, page) pair (indirect: VspFrameState.raster_dispatch).
//   6. vsp_sample_shadows: PCSS-filtered shadow mask.
//
// Light depth is standard (not reversed): light_view_proj is orthographic, maps world space to
// NDC xy in [-1, 1] and z in [0, 1], and SMALLER z is NEARER the sun. Atlas texels hold that z as
// float bits (ordered as uint for imageAtomicMin) and are cleared to 1.0.
// ============================================================================

const uint PRISM_VSP_CLIPMAP_LEVELS     = 8u;
const uint PRISM_VSP_PAGE_SIZE          = 128u;
const uint PRISM_VSP_VIRTUAL_PAGES_DIM  = 128u;   // 128x128 pages per clipmap level
const uint PRISM_VSP_VIRTUAL_RES        = PRISM_VSP_VIRTUAL_PAGES_DIM * PRISM_VSP_PAGE_SIZE; // 16384
const uint PRISM_VSP_ATLAS_PAGES_DIM    = 32u;    // 32x32 pages in the 4096x4096 physical atlas
const uint PRISM_VSP_MAX_PHYSICAL_PAGES = PRISM_VSP_ATLAS_PAGES_DIM * PRISM_VSP_ATLAS_PAGES_DIM; // 1024

// Page table entry (R32_UINT, u_page_table[vx, vy, level]; toroidal page coordinates):
//   [31]    VALID      holds this frame's shadow depth (or will, once rasterized this frame)
//   [30]    DIRTY      claimed by a pixel this frame (allocation / queueing attempted once)
//   [29]    ALLOCATED  owns the physical page in [11:0]
//   [28]    REQUESTED  a pixel needed it this frame
//   [27:20] AGE        frames since it was last requested (saturates at 255)
//   [11:0]  physical page index
const uint VSP_PTE_VALID      = 1u << 31;
const uint VSP_PTE_DIRTY      = 1u << 30;
const uint VSP_PTE_ALLOCATED  = 1u << 29;
const uint VSP_PTE_REQUESTED  = 1u << 28;
const uint VSP_PTE_AGE_SHIFT  = 20u;
const uint VSP_PTE_AGE_MASK   = 0xFFu << VSP_PTE_AGE_SHIFT;
const uint VSP_PTE_PAGE_MASK  = 0x0FFFu;

// Pass modes for vsp_page_request_and_invalidate.comp.glsl
const uint VSP_PASS_INVALIDATE        = 0u;
const uint VSP_PASS_REQUEST_AND_ALLOC = 1u;

// ----------------------------------------------------------------------------
// Per-clipmap level transform & toroidal scroll state (96 bytes, std140)
// ----------------------------------------------------------------------------
struct VspClipmapLevel {
    mat4  light_view_proj;     // orthographic, snapped to the page grid (see header)
    vec4  world_origin_scale;  // xyz = snapped world center, w = full width of the level (meters)
    ivec4 page_offset_and_pad; // xy = toroidal page offset (linear + offset = toroidal), zw = this frame's scroll in pages
};

// ----------------------------------------------------------------------------
// Dynamic object AABB for targeted shadow cache invalidation (32 bytes, std430)
// ----------------------------------------------------------------------------
struct VspDynamicDirtyAabb {
    vec4 min_ws_and_pad; // xyz = world-space min corner (union of prev & curr bounds)
    vec4 max_ws_and_pad; // xyz = world-space max corner
};

// ----------------------------------------------------------------------------
// Queued page to render this frame (8 bytes, std430)
// ----------------------------------------------------------------------------
struct VspPageWorkItem {
    uint level_and_vpage; // [31:24] = clipmap level, [15:8] = toroidal vpage_y, [7:0] = toroidal vpage_x
    uint physical_page;   // [11:0] = physical page index
};

// ----------------------------------------------------------------------------
// One shadow-casting meshlet overlapping one queued page (16 bytes, std430)
// ----------------------------------------------------------------------------
struct VspRasterItem {
    uint instance_id;
    uint meshlet_local_id;
    uint level_and_vpage;  // as VspPageWorkItem
    uint physical_page;
};

// ----------------------------------------------------------------------------
// GPU-written frame state (std430). free_page_count persists (initialize it once to
// PRISM_VSP_MAX_PHYSICAL_PAGES with b_free_pages = 0..1023); reset everything after it each frame
// before mode 1 to: counts 0, dispatches {0, 1, 1, 0}, dirty rects {~0u, ~0u, 0, 0}.
// ----------------------------------------------------------------------------
struct VspFrameState {
    uint  free_page_count;     // free-stack size
    uint  dirty_page_count;    // pages queued this frame (may exceed capacity; clamp)
    uint  raster_item_count;   // (meshlet, page) items this frame (may exceed capacity; clamp)
    uint  _pad0;
    uvec4 clear_dispatch;      // xyz: vkCmdDispatchIndirect args for vsp_clear_pages
    uvec4 raster_dispatch;     // xyz: vkCmdDispatchIndirect args for vsp_raster_pages
    uvec4 dirty_page_rect[PRISM_VSP_CLIPMAP_LEVELS]; // per level, linear page coords: xy = min, zw = max
};

// ----------------------------------------------------------------------------
// VSP frame uniforms (std140)
// ----------------------------------------------------------------------------
struct VspUniformUBO {
    mat4            inv_view_proj;
    mat4            view;
    vec4            camera_pos_and_near;      // xyz = world camera pos, w = near plane
    vec4            sun_dir_and_bias;         // xyz = unit vector toward sun, w = depth bias (meters)
    vec4            pcss_params;              // x = tan(sun angular radius), y = min filter texels, z = max filter texels, w = level LOD bias
    uvec4           viewport_and_counts;      // xy = render resolution, z = dirty_aabb_count, w = max_dirty_pages
    uint            pass_mode;                // VSP_PASS_INVALIDATE or VSP_PASS_REQUEST_AND_ALLOC
    uint            force_invalidate_all;     // 1 when the sun moved beyond the cache threshold
    uint            total_meshlet_candidates; // shadow-caster candidates for vsp_cull_meshlets
    uint            max_raster_items;         // capacity of the raster item queue
    uint            evict_after_frames;       // free a page once unused this many frames (>= 1)
    uint            frame_index;              // for the filter's rotating noise
    uint            _pad0;
    uint            _pad1;
    VspClipmapLevel levels[PRISM_VSP_CLIPMAP_LEVELS];
};

// ----------------------------------------------------------------------------
// Coordinate mapping helpers
// ----------------------------------------------------------------------------

// Finest clipmap level whose texels are fine enough for a pixel `view_depth` meters away
uint select_vsp_clipmap_level(float view_depth, float base_world_extent, float lod_bias) {
    float desired_extent = max(view_depth * 2.2 * exp2(lod_bias), base_world_extent);
    float raw_level = ceil(log2(max(desired_extent / base_world_extent, 1.0)));
    return uint(clamp(raw_level, 0.0, float(PRISM_VSP_CLIPMAP_LEVELS - 1u)));
}

// Projects a world position into a clipmap level: UV in [0, 1] and light depth in [0, 1].
// Returns false outside the level.
bool project_world_to_vsp_level(
    vec3 world_pos,
    VspClipmapLevel lvl,
    out vec2 out_clip_uv,
    out float out_light_z
) {
    vec4 clip = lvl.light_view_proj * vec4(world_pos, 1.0);
    vec3 ndc  = clip.xyz / clip.w;
    out_clip_uv = ndc.xy * 0.5 + 0.5;
    out_light_z = ndc.z;
    return all(greaterThanEqual(ndc, vec3(-1.0, -1.0, 0.0))) && all(lessThanEqual(ndc, vec3(1.0)));
}

// Light-depth units per meter along the light direction (row 2 of the orthographic matrix)
float vsp_depth_per_meter(VspClipmapLevel lvl) {
    return length(vec3(lvl.light_view_proj[0][2], lvl.light_view_proj[1][2], lvl.light_view_proj[2][2]));
}

// Linear page coordinate [0, 127] of a clipmap UV
ivec2 clip_uv_to_linear_vpage(vec2 clip_uv) {
    return clamp(ivec2(floor(clip_uv * float(PRISM_VSP_VIRTUAL_PAGES_DIM))), ivec2(0), ivec2(int(PRISM_VSP_VIRTUAL_PAGES_DIM) - 1));
}

ivec2 linear_to_toroidal_vpage(ivec2 linear_page, ivec2 toroidal_offset) {
    return (linear_page + toroidal_offset) & int(PRISM_VSP_VIRTUAL_PAGES_DIM - 1u);
}

ivec2 toroidal_to_linear_vpage(ivec2 toroidal_page, ivec2 toroidal_offset) {
    return (toroidal_page - toroidal_offset) & int(PRISM_VSP_VIRTUAL_PAGES_DIM - 1u);
}

ivec2 clip_uv_to_toroidal_vpage(vec2 clip_uv, ivec2 toroidal_offset) {
    return linear_to_toroidal_vpage(clip_uv_to_linear_vpage(clip_uv), toroidal_offset);
}

uint vsp_pack_level_vpage(uint level, ivec2 toroidal_page) {
    return (level << 24u) | ((uint(toroidal_page.y) & 0xFFu) << 8u) | (uint(toroidal_page.x) & 0xFFu);
}

void vsp_unpack_level_vpage(uint packed, out uint level, out ivec2 toroidal_page) {
    level = packed >> 24u;
    toroidal_page = ivec2(int(packed & 0xFFu), int((packed >> 8u) & 0xFFu));
}

// Top-left atlas texel of a physical page
ivec2 vsp_physical_page_origin(uint physical_page) {
    return ivec2(int(physical_page % PRISM_VSP_ATLAS_PAGES_DIM), int(physical_page / PRISM_VSP_ATLAS_PAGES_DIM)) * int(PRISM_VSP_PAGE_SIZE);
}

// Atlas texel holding clipmap UV `clip_uv`, inside the given physical page
ivec2 vsp_atlas_texel(uint physical_page, vec2 clip_uv) {
    ivec2 virtual_texel = ivec2(floor(clip_uv * float(PRISM_VSP_VIRTUAL_RES)));
    ivec2 in_page = clamp(virtual_texel - clip_uv_to_linear_vpage(clip_uv) * int(PRISM_VSP_PAGE_SIZE),
                          ivec2(0), ivec2(int(PRISM_VSP_PAGE_SIZE) - 1));
    return vsp_physical_page_origin(physical_page) + in_page;
}

// Light depth in [0, 1] <-> uint bits whose order matches the floats' (imageAtomicMin)
uint float_depth_to_ordered_uint(float d) {
    return floatBitsToUint(clamp(d, 0.0, 1.0));
}

float ordered_uint_to_float_depth(uint u) {
    return uintBitsToFloat(u);
}

#endif // PRISM_VSP_COMMON_GLSL
