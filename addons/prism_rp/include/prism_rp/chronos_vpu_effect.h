#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <vector>

#include "prism_rp/prism_rd_utils.h"

#include <godot_cpp/classes/compositor_effect.hpp>
#include <godot_cpp/classes/render_data.hpp>
#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/vector2.hpp>
#include <godot_cpp/variant/vector2i.hpp>

namespace prism {

// Chronos VPU, Godot 4.4 output contract:
//  - Runs at POST_TRANSPARENT, i.e. before Godot's own TAA/FSR2, upscale and tonemap.
//  - Everything resolves at the internal (render) resolution; Godot then upscales to the
//    output size. The compositor API has no way to hand Godot an upscaled frame.
//  - Writes linear Rec.709 back into the color buffer. With the SDR look enabled, that is
//    display-referred linear SDR, so the Environment must use TONE_MAPPER_LINEAR at exposure 1.
//  - Sub-pixel jitter: Godot only jitters when its own TAA/FSR2 runs (which would run after
//    us). Apply jitter through the camera instead (Camera3D PROJECTION_FRUSTUM +
//    frustum_offset); Chronos reads it from the frame's projection matrix.
class ChronosVPUCompositorEffect : public godot::CompositorEffect {
    GDCLASS(ChronosVPUCompositorEffect, godot::CompositorEffect)

public:
    enum OutputDisplayMode : int32_t {
        DISPLAY_MODE_SDR_SRGB = 0,
        DISPLAY_MODE_DISPLAY_P3 = 1,
        DISPLAY_MODE_HDR10_PQ = 2,
    };

    // How the SDR look (aces_odt_enabled) uses an HDR display. Godot 4.7 outputs HDR when the window
    // requests it (Window.hdr_output_requested) and the platform supports it.
    enum HdrOutputMode : int32_t {
        HDR_OUTPUT_FOLLOW_WINDOW = 0,  // extend highlights up to the main window's HDR peak when it outputs HDR
        HDR_OUTPUT_SDR_ONLY = 1,       // always map to [0, 1]
    };

    enum DebugViewMode : int32_t {
        DEBUG_VIEW_FINAL = 0,
        DEBUG_VIEW_DILATED_VELOCITY = 1,
        DEBUG_VIEW_REACTIVE_MASK = 2,
        DEBUG_VIEW_FFT_BLOOM = 3,
        DEBUG_VIEW_TAAU_CONFIDENCE = 4,
    };

    ChronosVPUCompositorEffect();
    ~ChronosVPUCompositorEffect() override;

    void _render_callback(int32_t p_effect_callback_type, godot::RenderData *p_render_data) override;

    // Stage toggles
    void set_dilate_velocity_enabled(bool p_enabled);
    [[nodiscard]] bool is_dilate_velocity_enabled() const;

    void set_reactive_mask_enabled(bool p_enabled);
    [[nodiscard]] bool is_reactive_mask_enabled() const;

    void set_taau_enabled(bool p_enabled);
    [[nodiscard]] bool is_taau_enabled() const;

    void set_fft_bloom_enabled(bool p_enabled);
    [[nodiscard]] bool is_fft_bloom_enabled() const;

    void set_aces_odt_enabled(bool p_enabled);
    [[nodiscard]] bool is_aces_odt_enabled() const;

    // Compute TAAU parameters
    void set_taau_sharpness(float p_sharpness);
    [[nodiscard]] float get_taau_sharpness() const;

    void set_taau_feedback_min(float p_min);
    [[nodiscard]] float get_taau_feedback_min() const;

    void set_taau_feedback_max(float p_max);
    [[nodiscard]] float get_taau_feedback_max() const;

    void reset_taau_history();

    // Color, optical bloom, and reactive tuning parameters.
    // output_display_mode's Display P3 / HDR10 values predate Godot's HDR output (4.7), which replaces
    // them: they're kept so saved scenes load, warn, and behave like SDR sRGB. HDR follows hdr_output.
    void set_output_display_mode(int32_t p_mode);
    [[nodiscard]] int32_t get_output_display_mode() const;

    void set_hdr_output(int32_t p_mode);
    [[nodiscard]] int32_t get_hdr_output() const;
    // The peak the SDR look maps to this frame, in linear units (1.0 = SDR white).
    [[nodiscard]] float get_output_max_linear() const;

    void set_debug_view_mode(int32_t p_mode);
    [[nodiscard]] int32_t get_debug_view_mode() const;

    void set_exposure_ev(float p_ev);
    [[nodiscard]] float get_exposure_ev() const;

    void set_peak_luminance_nits(float p_nits);
    [[nodiscard]] float get_peak_luminance_nits() const;

    void set_bloom_threshold(float p_threshold);
    [[nodiscard]] float get_bloom_threshold() const;

    void set_bloom_intensity(float p_intensity);
    [[nodiscard]] float get_bloom_intensity() const;

    void set_halation_intensity(float p_intensity);
    [[nodiscard]] float get_halation_intensity() const;

    void set_aperture_blades(float p_blades);
    [[nodiscard]] float get_aperture_blades() const;

    void set_reactive_sensitivity(float p_sensitivity);
    [[nodiscard]] float get_reactive_sensitivity() const;

    // Weight of the particle/transparency term in the reactive mask. Needs a
    // ChronosOpaqueCaptureEffect in the same Compositor; without one this has no effect.
    void set_transparency_reactive_scale(float p_scale);
    [[nodiscard]] float get_transparency_reactive_scale() const;

    // Hot-reload & downstream RID accessors
    void reload_shaders();
    [[nodiscard]] godot::RID get_dilated_velocity_rid(int32_t p_view = 0) const;
    [[nodiscard]] godot::RID get_reactive_mask_rid(int32_t p_view = 0) const;
    [[nodiscard]] godot::RID get_taau_output_rid(int32_t p_view = 0) const;
    [[nodiscard]] godot::RID get_bloom_texture_rid(int32_t p_view = 0) const;

protected:
    static void _bind_methods();

private:
    static constexpr uint32_t kFFTGridSize = 256;

    struct alignas(16) DilateVelocityPushConstants {
        int32_t resolution_x = 0;
        int32_t resolution_y = 0;
        float jitter_delta_uv_x = 0.0f;
        float jitter_delta_uv_y = 0.0f;
    };
    static_assert(sizeof(DilateVelocityPushConstants) == 16);

    struct alignas(16) ReactiveMaskPushConstants {
        int32_t resolution_x = 0;
        int32_t resolution_y = 0;
        float reactive_sensitivity = 1.0f;
        float luma_spark_threshold = 2.5f;
        float jitter_delta_uv_x = 0.0f;
        float jitter_delta_uv_y = 0.0f;
        float depth_param_a = 0.05f;
        float depth_param_b = 0.0f;
        float transparency_scale = 1.0f;
        uint32_t flags = 0;
        float _pad0 = 0.0f;
        float _pad1 = 0.0f;
    };
    static_assert(sizeof(ReactiveMaskPushConstants) == 48);

    struct alignas(16) TAAUPushConstants {
        int32_t internal_res_x = 0;
        int32_t internal_res_y = 0;
        int32_t target_res_x = 0;
        int32_t target_res_y = 0;
        float jitter_x = 0.0f;
        float jitter_y = 0.0f;
        float feedback_min = 0.05f;
        float feedback_max = 0.95f;
        float motion_scale = 18.0f;
        uint32_t frame_index = 0;
        uint32_t flags = 0;
        float _pad = 0.0f;
    };
    static_assert(sizeof(TAAUPushConstants) == 48);

    struct alignas(16) FFTForwardHorizPushConstants {
        float bloom_threshold = 1.15f;
        float bloom_soft_knee = 0.5f;
        float exposure_scale = 1.0f;
        float _pad = 0.0f;
    };
    static_assert(sizeof(FFTForwardHorizPushConstants) == 16);

    struct alignas(16) FFTVerticalKernelPushConstants {
        float bloom_radius = 1.0f;
        float diffraction_strength = 0.45f;
        float halation_strength = 0.35f;
        float aperture_blades = 6.0f;
        float aspect_ratio = 1.0f;
        float _pad0 = 0.0f;
        float _pad1 = 0.0f;
        float _pad2 = 0.0f;
    };
    static_assert(sizeof(FFTVerticalKernelPushConstants) == 32);

    struct alignas(16) FFTInvHorizPushConstants {
        float bloom_intensity = 0.18f;
        float _pad0 = 0.0f;
        float _pad1 = 0.0f;
        float _pad2 = 0.0f;
    };
    static_assert(sizeof(FFTInvHorizPushConstants) == 16);

    struct alignas(16) CompositePushConstants {
        int32_t resolution_x = 0;
        int32_t resolution_y = 0;
        float exposure_scale = 1.0f;
        float sharpness = 0.0f;
        int32_t debug_view_mode = 0;
        uint32_t flags = 0;
        float output_max_linear = 1.0f;  // display peak for the SDR look (<= 1: SDR display)
        float _pad1 = 0.0f;
    };
    static_assert(sizeof(CompositePushConstants) == 32);

    struct PerViewTargets {
        godot::RID dilated_velocity;                 // RG16F @ internal_size
        godot::RID reactive_mask;                    // R8_UNORM @ internal_size
        godot::RID prev_color_history;               // RGBA16F @ internal_size: raw frame copy (compute copy) (shock reference + composite input when TAA is off)
        std::array<godot::RID, 2> taau_history;      // RGBA16F ping-pong @ internal_size (linear HDR + confidence)
        godot::RID fft_ping_rg;                      // RGBA32F 256x256 complex
        godot::RID fft_ping_b;                       // RGBA32F 256x256 complex
        godot::RID fft_pong_rg;                      // RGBA32F 256x256 complex
        godot::RID fft_pong_b;                       // RGBA32F 256x256 complex
        godot::RID bloom_spatial;                    // RGBA16F 256x256 spatial bloom + halation
        uint32_t taau_write_index = 0;
        bool has_valid_history = false;
        bool has_prev_jitter = false;
        godot::Vector2 prev_jitter_uv;
    };

    bool _ensure_render_resources();
    bool _ensure_view_targets(const godot::Vector2i &p_internal_size, uint32_t p_view_count);
    void _release_render_resources();
    void _release_view_targets();

    [[nodiscard]] godot::RID _create_2d_texture(
        uint32_t p_width,
        uint32_t p_height,
        godot::RenderingDevice::DataFormat p_format,
        uint32_t p_usage_bits
    ) const;

    godot::RenderingDevice *rd = nullptr;

    godot::RID nearest_sampler;
    godot::RID linear_sampler;

    godot::RID dilate_shader;
    godot::RID dilate_pipeline;

    godot::RID reactive_shader;
    godot::RID reactive_pipeline;

    godot::RID taau_shader;
    godot::RID taau_pipeline;

    godot::RID fft_horiz_shader;
    godot::RID fft_horiz_pipeline;

    godot::RID fft_vert_shader;
    godot::RID fft_vert_pipeline;

    godot::RID fft_inv_horiz_shader;
    godot::RID fft_inv_horiz_pipeline;

    godot::RID aces_odt_shader;
    godot::RID aces_odt_pipeline;

    rd_utils::ComputeCopy color_copy;

    std::vector<PerViewTargets> view_targets;
    godot::Vector2i cached_internal_size = godot::Vector2i(0, 0);
    uint32_t frame_counter = 0;

    bool dilate_velocity_enabled = true;
    bool reactive_mask_enabled = true;
    bool taau_enabled = true;
    bool fft_bloom_enabled = true;
    bool aces_odt_enabled = true;
    bool taau_reset_requested = true;
    bool pipeline_dirty = true;

    float taau_sharpness = 0.25f;
    float taau_feedback_min = 0.05f;
    float taau_feedback_max = 0.95f;

    int32_t output_display_mode = DISPLAY_MODE_SDR_SRGB;
    int32_t debug_view_mode = DEBUG_VIEW_FINAL;
    float exposure_ev = 0.0f;
    float peak_luminance_nits = 100.0f;  // unused: kept so saved scenes load; the HDR peak comes from the window
    float bloom_threshold = 1.15f;
    float bloom_intensity = 0.18f;
    float halation_intensity = 0.35f;
    float aperture_blades = 6.0f;
    float reactive_sensitivity = 1.0f;
    float transparency_reactive_scale = 1.0f;

    bool godot_temporal_warned = false;

    // HDR: the render thread can't query the window, so each frame it queues _sample_display_output()
    // on the main thread, which stores the main window's peak here (one frame of latency).
    void _sample_display_output();
    int32_t hdr_output = HDR_OUTPUT_FOLLOW_WINDOW;
    std::atomic<float> display_max_linear{1.0f};
    bool hdr_detected_logged = false;
};

} // namespace prism
