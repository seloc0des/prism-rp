#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <godot_cpp/classes/compositor_effect.hpp>
#include <godot_cpp/classes/render_data.hpp>
#include <godot_cpp/classes/render_scene_buffers_rd.hpp>
#include <godot_cpp/classes/render_scene_data.hpp>
#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/vector2i.hpp>

namespace prism {

// Lumina-Cascades v1: screen-space radiance cascades + SVGF, as a Godot 4.4 CompositorEffect.
//
// Runs at POST_OPAQUE and adds one bounce of indirect diffuse light to the opaque color buffer:
//  1. trace:     per cascade level, screen probes with octahedral direction maps march view-space rays
//                against the depth buffer and pick up radiance from the current opaque lighting.
//                Level 0: probes every `probe_spacing` px with 16 directions; each level doubles the
//                spacing and quadruples the directions (constant memory per level).
//  2. merge:     top-down, interval + transmittance * parent (depth-weighted bilinear over parent probes).
//  3. integrate: cosine-weighted irradiance per pixel from the 4 nearest level-0 probes.
//  4. SVGF:      temporal accumulation with moments/variance, then variance-guided a-trous.
//  5. composite: color += albedo_proxy * irradiance (Forward+ has no albedo buffer; see the shader).
//
// World-space radiance cache (world_cascades_enabled, on by default): a 16^3 probe grid
// around the camera, 64 directions per probe. The top screen level's transmittance falls through to it.
// Where ray queries work (the custom build on a GPU that has them), each direction traces a ray through
// Lumina's scene mirror (LuminaSceneMirror) every frame: hits the camera sees take their screen radiance,
// others are lit from the scene's lights with shadow rays, plus emission, and with world_multi_bounce_enabled
// every hit adds the cache's own irradiance there, so light keeps bouncing (lumina_ws_update_rt,
// lumina_ws_irradiance). Elsewhere it marches world-space rays through the depth buffer, and rays that
// leave the screen keep their cached value, so light that was visible recently (an emitter now behind the
// camera) keeps contributing.
//
// Limits: without ray queries, light never seen on screen contributes nothing; the screen-space levels
// see one bounce; no sky light; no ReSTIR.
class LuminaCascadesCompositorEffect : public godot::CompositorEffect {
    GDCLASS(LuminaCascadesCompositorEffect, godot::CompositorEffect)

public:
    enum DebugViewMode : int32_t {
        DEBUG_VIEW_FINAL = 0,
        DEBUG_VIEW_RAW_IRRADIANCE = 1,
        DEBUG_VIEW_DENOISED_IRRADIANCE = 2,
        DEBUG_VIEW_HISTORY_LENGTH = 3,
        DEBUG_VIEW_VARIANCE = 4,
        DEBUG_VIEW_NORMALS = 5,
        DEBUG_VIEW_ALBEDO_PROXY = 6,
        DEBUG_VIEW_WORLD_CASCADE_ONLY = 7,   // far field from the world cache only (needs world_cascades_enabled)
        DEBUG_VIEW_RAY_TRACED_ALBEDO = 8,    // albedo where camera rays hit the scene mirror (needs ray queries)
    };

    enum AlbedoMode : int32_t {
        ALBEDO_PROXY = 0,
        ALBEDO_WHITE = 1,
    };

    LuminaCascadesCompositorEffect();
    ~LuminaCascadesCompositorEffect() override;

    void _render_callback(int32_t p_effect_callback_type, godot::RenderData *p_render_data) override;

    void set_gi_enabled(bool p_enabled);
    [[nodiscard]] bool is_gi_enabled() const;
    void set_temporal_enabled(bool p_enabled);
    [[nodiscard]] bool is_temporal_enabled() const;
    void set_atrous_iterations(int32_t p_iterations);
    [[nodiscard]] int32_t get_atrous_iterations() const;
    void set_cascade_count(int32_t p_count);
    [[nodiscard]] int32_t get_cascade_count() const;
    void set_probe_spacing(int32_t p_spacing);
    [[nodiscard]] int32_t get_probe_spacing() const;
    void set_steps_per_interval(int32_t p_steps);
    [[nodiscard]] int32_t get_steps_per_interval() const;
    void set_thickness(float p_thickness);
    [[nodiscard]] float get_thickness() const;
    void set_intensity(float p_intensity);
    [[nodiscard]] float get_intensity() const;
    void set_denoise_strength(float p_strength);
    [[nodiscard]] float get_denoise_strength() const;
    void set_albedo_mode(int32_t p_mode);
    [[nodiscard]] int32_t get_albedo_mode() const;
    void set_debug_view_mode(int32_t p_mode);
    [[nodiscard]] int32_t get_debug_view_mode() const;
    void set_world_cascades_enabled(bool p_enabled);
    [[nodiscard]] bool is_world_cascades_enabled() const;
    void set_world_probe_spacing(float p_spacing);
    [[nodiscard]] float get_world_probe_spacing() const;
    void set_world_multi_bounce_enabled(bool p_enabled);
    [[nodiscard]] bool is_world_multi_bounce_enabled() const;
    void set_screen_multi_bounce_enabled(bool p_enabled);
    [[nodiscard]] bool is_screen_multi_bounce_enabled() const;

    void reset_history();
    void reload_shaders();
    [[nodiscard]] godot::RID get_irradiance_rid(int32_t p_view = 0) const;

protected:
    static void _bind_methods();

private:
    static constexpr int32_t kMaxCascades = 4;
    static constexpr int32_t kBaseBlock = 4;   // level-0 directions = 4 x 4 = 16
    static constexpr int32_t kWorldGridN = 16; // world cache: 16^3 probes per level
    static constexpr int32_t kWorldLevels = 2; // level 0 at world_probe_spacing, level 1 at twice it (WORLD_LEVELS)
    static constexpr int32_t kWorldBlock = 8;  // 8 x 8 = 64 directions per world probe (WORLD_BLOCK)
    static constexpr uint32_t kMaxRadianceMips = 16;   // a full chain for up to 65535 px

    // Shared by lumina_ws_classify / _update / _merge (lumina_world_common.glsl). 112 bytes.
    struct alignas(16) WorldPushConstants {
        int32_t a0[4] = { 0, 0, kWorldGridN, kWorldBlock };   // res.xy, grid N, block
        int32_t a1[4] = { 0, 0, 0, 0 };                      // base cell xyz, frame / flags
        float cam_pos[4] = { 0, 0, 0, 1 };                   // xyz, probe spacing
        float cam_right[4] = { 1, 0, 0, 0.25f };             // xyz, thickness
        float cam_up[4] = { 0, 1, 0, 0.1f };                 // xyz, blend alpha minimum
        float proj[4] = { 1, 1, 0, 0 };
        float misc[4] = { 0.05f, 0, 8, 32 };                 // depth params, ray length / spacing px, steps
    };
    static_assert(sizeof(WorldPushConstants) == 112);

    struct alignas(16) TracePushConstants {
        int32_t internal_res_x = 0;
        int32_t internal_res_y = 0;
        int32_t probe_count_x = 0;
        int32_t probe_count_y = 0;
        int32_t block = kBaseBlock;
        float probe_spacing_px = 8.0f;
        float interval_start_px = 0.0f;
        float interval_length_px = 8.0f;   // <= 0: on to the edge of the screen (top level)
        int32_t steps = 8;
        float thickness = 0.1f;
        uint32_t frame_index = 0;
        float normal_bias = 0.01f;
        float proj[4] = { 1.0f, 1.0f, 0.0f, 0.0f };
        float depth_params[2] = { 0.05f, 0.0f };
        int32_t sub_rays = 2;              // per axis
        float _pad = 0.0f;
    };
    static_assert(sizeof(TracePushConstants) == 80);

    struct alignas(16) DownsamplePushConstants {
        int32_t dst_width = 0;
        int32_t dst_height = 0;
        int32_t _pad[2] = { 0, 0 };
    };
    static_assert(sizeof(DownsamplePushConstants) == 16);

    struct alignas(16) RadianceCopyPushConstants {
        int32_t width = 0;
        int32_t height = 0;
        float depth_params[2] = { 0.05f, 0.0f };
        float bounce_gain = 0.0f;   // 0: no previous GI to bounce
        float scene_albedo = 0.0f;  // 1: binding 7 holds this frame's albedo from the scene mirror
        float _pad[2] = { 0.0f, 0.0f };
    };
    static_assert(sizeof(RadianceCopyPushConstants) == 32);

    struct alignas(16) MergePushConstants {
        int32_t probe_count_x = 0;
        int32_t probe_count_y = 0;
        int32_t parent_probe_count_x = 0;
        int32_t parent_probe_count_y = 0;
        int32_t block = 0;
        int32_t parent_block = 0;
        float probe_spacing_px = 0.0f;
        float parent_spacing_px = 0.0f;
        int32_t internal_resolution[2] = { 0, 0 };
        float depth_params[2] = { 0.0f, 0.0f };
    };
    static_assert(sizeof(MergePushConstants) == 48);

    struct alignas(16) IntegratePushConstants {
        int32_t internal_res_x = 0;
        int32_t internal_res_y = 0;
        int32_t probe_count_x = 0;
        int32_t probe_count_y = 0;
        int32_t block = kBaseBlock;
        float probe_spacing_px = 8.0f;
        float depth_params[2] = { 0.05f, 0.0f };
        float proj[4] = { 1.0f, 1.0f, 0.0f, 0.0f };   // P00, P11, P20, P21 (see lumina_view_position)
    };
    static_assert(sizeof(IntegratePushConstants) == 48);

    struct alignas(16) TemporalPushConstants {
        int32_t internal_res_x = 0;
        int32_t internal_res_y = 0;
        float depth_params[2] = { 0.05f, 0.0f };
        uint32_t flags = 0;
        float alpha_min = 0.1f;
        float moments_alpha_min = 0.2f;
        float max_history = 32.0f;
    };
    static_assert(sizeof(TemporalPushConstants) == 32);

    struct alignas(16) AtrousPushConstants {
        int32_t internal_res_x = 0;
        int32_t internal_res_y = 0;
        int32_t step_px = 1;
        float phi_color = 4.0f;
        float phi_normal = 128.0f;
        float phi_depth = 1.0f;
        float _pad[2] = { 0.0f, 0.0f };
    };
    static_assert(sizeof(AtrousPushConstants) == 32);

    struct alignas(16) CompositePushConstants {
        int32_t internal_res_x = 0;
        int32_t internal_res_y = 0;
        float intensity = 1.0f;
        int32_t debug_view_mode = 0;
        int32_t albedo_mode = 0;
        float max_history = 32.0f;
        float _pad[2] = { 0.0f, 0.0f };
    };
    static_assert(sizeof(CompositePushConstants) == 32);

    struct alignas(16) RtAlbedoPushConstants {
        float cam_x[4] = { 1, 0, 0, 0 };   // world-space camera basis column, eye coordinate
        float cam_y[4] = { 0, 1, 0, 0 };
        float cam_z[4] = { 0, 0, 1, 0 };
        float proj[4] = { 1, 1, 0, 0 };
        int32_t size[2] = { 0, 0 };
        float depth_params[2] = { 0.05f, 0.0f };
    };
    static_assert(sizeof(RtAlbedoPushConstants) == 80);

    struct CascadeLevel {
        godot::RID texture;     // RGBA16F: rgb radiance, a transmittance
        godot::Vector2i probe_count;
        int32_t block = kBaseBlock;
        float spacing_px = 8.0f;
    };

    struct PerViewTargets {
        std::array<CascadeLevel, kMaxCascades> cascades;
        // Radiance mip chain every level traces against: RGBA16F, rgb radiance, a nearest linear depth.
        // radiance_mip_views[k] is mip k alone, written by its pass and read to make mip k + 1.
        godot::RID radiance_mips;
        std::array<godot::RID, kMaxRadianceMips> radiance_mip_views;
        uint32_t radiance_mip_count = 0;
        godot::RID irradiance_raw;                  // RGBA16F
        std::array<godot::RID, 2> history;          // RGBA16F: rgb, a = history length
        std::array<godot::RID, 2> moments;          // RG16F
        std::array<godot::RID, 2> geometry;         // RGBA16F: view normal, linear depth
        std::array<godot::RID, 2> filter;           // RGBA16F a-trous ping-pong: rgb, a = variance
        godot::RID scene_albedo;                    // RGBA16F: the scene mirror's albedo, a = 1 where a ray hit
        uint32_t history_index = 0;                 // which of [0], [1] holds last frame
        bool has_history = false;
        uint32_t last_gi = 0;                       // which filter[] holds last frame's denoised GI
        bool has_last_gi = false;
    };

    bool _ensure_render_resources();
    void _release_render_resources();
    bool _ensure_view_targets(const godot::Vector2i &p_internal_size, uint32_t p_view_count);
    void _release_view_targets();
    bool _render_ray_traced_albedo(godot::RenderSceneBuffersRD *p_buffers, godot::RenderSceneData *p_scene_data,
            const godot::Vector2i &p_size, uint32_t p_view_count);
    bool _ensure_rt_albedo_pipeline();
    void _trace_scene_albedo(godot::RenderSceneData *p_scene_data, uint32_t p_view, const godot::RID &p_target,
            const godot::RID &p_depth, const godot::Vector2i &p_size, const godot::RID &p_scene_set);

    godot::RenderingDevice *rd = nullptr;
    godot::RID nearest_sampler;
    godot::RID linear_sampler;

    godot::RID trace_shader, trace_pipeline;
    godot::RID radiance_copy_shader, radiance_copy_pipeline;
    godot::RID downsample_shader, downsample_pipeline;
    godot::RID merge_shader, merge_pipeline;
    godot::RID integrate_shader, integrate_pipeline;
    godot::RID temporal_shader, temporal_pipeline;
    godot::RID atrous_shader, atrous_pipeline;
    godot::RID composite_shader, composite_pipeline;
    godot::RID ws_classify_shader, ws_classify_pipeline;
    godot::RID ws_update_shader, ws_update_pipeline;
    godot::RID ws_merge_shader, ws_merge_pipeline;
    godot::RID rt_albedo_shader, rt_albedo_pipeline;   // built on first use, only where ray queries work
    bool rt_albedo_failed = false;
    godot::RID ws_update_rt_shader, ws_update_rt_pipeline;     // the same, for the ray-traced cache update
    godot::RID ws_irradiance_shader, ws_irradiance_pipeline;
    bool world_rt_failed = false;

    // World cache (view-independent; updated from view 0 once per frame).
    godot::RID world_probes;   // RGBA16F (N*N*block) x (levels*N*block)
    godot::RID world_irradiance;   // same layout: per cell, irradiance / pi by surface normal (read by the traced update)
    godot::RID world_state;    // RGBA32F (N*N) x (levels*N): cell id, changed, buried, written
    float cached_world_spacing = 0.0f;

    std::vector<PerViewTargets> view_targets;
    godot::Vector2i cached_internal_size = godot::Vector2i(0, 0);
    int32_t cached_cascade_count = 0;
    int32_t cached_probe_spacing = 0;
    uint32_t frame_counter = 0;
    bool pipeline_dirty = true;
    bool history_reset_requested = true;

    bool gi_enabled = true;
    bool temporal_enabled = true;
    int32_t atrous_iterations = 3;
    int32_t cascade_count = 4;
    int32_t probe_spacing = 8;
    int32_t steps_per_interval = 6;
    // Fraction of view depth behind a surface that still counts as a hit. 0.1 caught rays passing behind lit
    // pillars at Sponza's depths (a metre at ten) and took their light: Gate 2 gains 0.5 dB there at 0.05.
    float thickness = 0.05f;
    float intensity = 1.0f;
    float denoise_strength = 4.0f;
    int32_t albedo_mode = ALBEDO_PROXY;
    int32_t debug_view_mode = DEBUG_VIEW_FINAL;
    bool world_cascades_enabled = true;
    float world_probe_spacing = 1.0f;
    bool world_multi_bounce_enabled = true;   // the ray-traced update bounces the cache's own irradiance
    bool screen_multi_bounce_enabled = true;  // the radiance copy bounces last frame's GI off every pixel

    bool _ensure_world_targets();
    [[nodiscard]] godot::RID _world_scene_set();
    void _release_world_targets();
};

} // namespace prism
