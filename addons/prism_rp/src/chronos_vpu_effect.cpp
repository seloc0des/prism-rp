#include "prism_rp/chronos_vpu_effect.h"
#include "prism_rp/prism_rd_utils.h"
#include "prism_rp/chronos_opaque_capture_effect.h"
#include "prism_rp/prism_telemetry.h"

#include <cmath>
#include <cstring>
#include <string>

#include <godot_cpp/classes/rd_uniform.hpp>
#include <godot_cpp/classes/render_scene_buffers.hpp>
#include <godot_cpp/classes/render_scene_buffers_rd.hpp>
#include <godot_cpp/classes/render_scene_data.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/classes/uniform_set_cache_rd.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/projection.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/variant/vector3.hpp>

namespace prism {

using rd_utils::make_image_uniform;
using rd_utils::make_push_constant_bytes;
using rd_utils::make_sampler_texture_uniform;

ChronosVPUCompositorEffect::ChronosVPUCompositorEffect() {
    set_effect_callback_type(EFFECT_CALLBACK_TYPE_POST_TRANSPARENT);
    set_access_resolved_color(true);
    set_access_resolved_depth(true);
    set_needs_motion_vectors(true);

    if (godot::RenderingServer::get_singleton() != nullptr) {
        rd = godot::RenderingServer::get_singleton()->get_rendering_device();
    }
}

ChronosVPUCompositorEffect::~ChronosVPUCompositorEffect() {
    _release_view_targets();
    _release_render_resources();
}

void ChronosVPUCompositorEffect::_bind_methods() {
    godot::ClassDB::bind_method(godot::D_METHOD("set_dilate_velocity_enabled", "enabled"), &ChronosVPUCompositorEffect::set_dilate_velocity_enabled);
    godot::ClassDB::bind_method(godot::D_METHOD("is_dilate_velocity_enabled"), &ChronosVPUCompositorEffect::is_dilate_velocity_enabled);
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::BOOL, "dilate_velocity_enabled"), "set_dilate_velocity_enabled", "is_dilate_velocity_enabled");

    godot::ClassDB::bind_method(godot::D_METHOD("set_reactive_mask_enabled", "enabled"), &ChronosVPUCompositorEffect::set_reactive_mask_enabled);
    godot::ClassDB::bind_method(godot::D_METHOD("is_reactive_mask_enabled"), &ChronosVPUCompositorEffect::is_reactive_mask_enabled);
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::BOOL, "reactive_mask_enabled"), "set_reactive_mask_enabled", "is_reactive_mask_enabled");

    godot::ClassDB::bind_method(godot::D_METHOD("set_taau_enabled", "enabled"), &ChronosVPUCompositorEffect::set_taau_enabled);
    godot::ClassDB::bind_method(godot::D_METHOD("is_taau_enabled"), &ChronosVPUCompositorEffect::is_taau_enabled);
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::BOOL, "taau_enabled"), "set_taau_enabled", "is_taau_enabled");

    godot::ClassDB::bind_method(godot::D_METHOD("set_fft_bloom_enabled", "enabled"), &ChronosVPUCompositorEffect::set_fft_bloom_enabled);
    godot::ClassDB::bind_method(godot::D_METHOD("is_fft_bloom_enabled"), &ChronosVPUCompositorEffect::is_fft_bloom_enabled);
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::BOOL, "fft_bloom_enabled"), "set_fft_bloom_enabled", "is_fft_bloom_enabled");

    godot::ClassDB::bind_method(godot::D_METHOD("set_aces_odt_enabled", "enabled"), &ChronosVPUCompositorEffect::set_aces_odt_enabled);
    godot::ClassDB::bind_method(godot::D_METHOD("is_aces_odt_enabled"), &ChronosVPUCompositorEffect::is_aces_odt_enabled);
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::BOOL, "aces_odt_enabled"), "set_aces_odt_enabled", "is_aces_odt_enabled");

    godot::ClassDB::bind_method(godot::D_METHOD("set_taau_sharpness", "sharpness"), &ChronosVPUCompositorEffect::set_taau_sharpness);
    godot::ClassDB::bind_method(godot::D_METHOD("get_taau_sharpness"), &ChronosVPUCompositorEffect::get_taau_sharpness);
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::FLOAT, "taau_sharpness", godot::PROPERTY_HINT_RANGE, "0.0,1.0,0.02"), "set_taau_sharpness", "get_taau_sharpness");

    godot::ClassDB::bind_method(godot::D_METHOD("set_taau_feedback_min", "min"), &ChronosVPUCompositorEffect::set_taau_feedback_min);
    godot::ClassDB::bind_method(godot::D_METHOD("get_taau_feedback_min"), &ChronosVPUCompositorEffect::get_taau_feedback_min);
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::FLOAT, "taau_feedback_min", godot::PROPERTY_HINT_RANGE, "0.0,0.5,0.01"), "set_taau_feedback_min", "get_taau_feedback_min");

    godot::ClassDB::bind_method(godot::D_METHOD("set_taau_feedback_max", "max"), &ChronosVPUCompositorEffect::set_taau_feedback_max);
    godot::ClassDB::bind_method(godot::D_METHOD("get_taau_feedback_max"), &ChronosVPUCompositorEffect::get_taau_feedback_max);
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::FLOAT, "taau_feedback_max", godot::PROPERTY_HINT_RANGE, "0.5,0.99,0.01"), "set_taau_feedback_max", "get_taau_feedback_max");

    godot::ClassDB::bind_method(godot::D_METHOD("reset_taau_history"), &ChronosVPUCompositorEffect::reset_taau_history);

    godot::ClassDB::bind_method(godot::D_METHOD("set_output_display_mode", "mode"), &ChronosVPUCompositorEffect::set_output_display_mode);
    godot::ClassDB::bind_method(godot::D_METHOD("get_output_display_mode"), &ChronosVPUCompositorEffect::get_output_display_mode);
    ADD_PROPERTY(
        godot::PropertyInfo(godot::Variant::INT, "output_display_mode", godot::PROPERTY_HINT_ENUM, "SDR sRGB:0,Display P3 (unsupported in Godot 4.4):1,HDR10 PQ (unsupported in Godot 4.4):2"),
        "set_output_display_mode",
        "get_output_display_mode"
    );

    godot::ClassDB::bind_method(godot::D_METHOD("set_debug_view_mode", "mode"), &ChronosVPUCompositorEffect::set_debug_view_mode);
    godot::ClassDB::bind_method(godot::D_METHOD("get_debug_view_mode"), &ChronosVPUCompositorEffect::get_debug_view_mode);
    ADD_PROPERTY(
        godot::PropertyInfo(godot::Variant::INT, "debug_view_mode", godot::PROPERTY_HINT_ENUM, "Final Output:0,Dilated Velocity:1,Reactive Mask:2,FFT Bloom & Halation:3,TAA Confidence:4"),
        "set_debug_view_mode",
        "get_debug_view_mode"
    );

    godot::ClassDB::bind_method(godot::D_METHOD("set_exposure_ev", "ev"), &ChronosVPUCompositorEffect::set_exposure_ev);
    godot::ClassDB::bind_method(godot::D_METHOD("get_exposure_ev"), &ChronosVPUCompositorEffect::get_exposure_ev);
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::FLOAT, "exposure_ev", godot::PROPERTY_HINT_RANGE, "-6.0,6.0,0.05"), "set_exposure_ev", "get_exposure_ev");

    godot::ClassDB::bind_method(godot::D_METHOD("set_peak_luminance_nits", "nits"), &ChronosVPUCompositorEffect::set_peak_luminance_nits);
    godot::ClassDB::bind_method(godot::D_METHOD("get_peak_luminance_nits"), &ChronosVPUCompositorEffect::get_peak_luminance_nits);
    // Reserved for HDR output; has no effect while only SDR is supported.
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::FLOAT, "peak_luminance_nits", godot::PROPERTY_HINT_RANGE, "80.0,4000.0,10.0"), "set_peak_luminance_nits", "get_peak_luminance_nits");

    godot::ClassDB::bind_method(godot::D_METHOD("set_bloom_threshold", "threshold"), &ChronosVPUCompositorEffect::set_bloom_threshold);
    godot::ClassDB::bind_method(godot::D_METHOD("get_bloom_threshold"), &ChronosVPUCompositorEffect::get_bloom_threshold);
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::FLOAT, "bloom_threshold", godot::PROPERTY_HINT_RANGE, "0.0,10.0,0.05"), "set_bloom_threshold", "get_bloom_threshold");

    godot::ClassDB::bind_method(godot::D_METHOD("set_bloom_intensity", "intensity"), &ChronosVPUCompositorEffect::set_bloom_intensity);
    godot::ClassDB::bind_method(godot::D_METHOD("get_bloom_intensity"), &ChronosVPUCompositorEffect::get_bloom_intensity);
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::FLOAT, "bloom_intensity", godot::PROPERTY_HINT_RANGE, "0.0,2.0,0.01"), "set_bloom_intensity", "get_bloom_intensity");

    godot::ClassDB::bind_method(godot::D_METHOD("set_halation_intensity", "intensity"), &ChronosVPUCompositorEffect::set_halation_intensity);
    godot::ClassDB::bind_method(godot::D_METHOD("get_halation_intensity"), &ChronosVPUCompositorEffect::get_halation_intensity);
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::FLOAT, "halation_intensity", godot::PROPERTY_HINT_RANGE, "0.0,2.0,0.01"), "set_halation_intensity", "get_halation_intensity");

    godot::ClassDB::bind_method(godot::D_METHOD("set_aperture_blades", "blades"), &ChronosVPUCompositorEffect::set_aperture_blades);
    godot::ClassDB::bind_method(godot::D_METHOD("get_aperture_blades"), &ChronosVPUCompositorEffect::get_aperture_blades);
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::FLOAT, "aperture_blades", godot::PROPERTY_HINT_RANGE, "4.0,12.0,1.0"), "set_aperture_blades", "get_aperture_blades");

    godot::ClassDB::bind_method(godot::D_METHOD("set_reactive_sensitivity", "sensitivity"), &ChronosVPUCompositorEffect::set_reactive_sensitivity);
    godot::ClassDB::bind_method(godot::D_METHOD("get_reactive_sensitivity"), &ChronosVPUCompositorEffect::get_reactive_sensitivity);
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::FLOAT, "reactive_sensitivity", godot::PROPERTY_HINT_RANGE, "0.0,2.5,0.05"), "set_reactive_sensitivity", "get_reactive_sensitivity");

    godot::ClassDB::bind_method(godot::D_METHOD("set_transparency_reactive_scale", "scale"), &ChronosVPUCompositorEffect::set_transparency_reactive_scale);
    godot::ClassDB::bind_method(godot::D_METHOD("get_transparency_reactive_scale"), &ChronosVPUCompositorEffect::get_transparency_reactive_scale);
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::FLOAT, "transparency_reactive_scale", godot::PROPERTY_HINT_RANGE, "0.0,2.0,0.05"), "set_transparency_reactive_scale", "get_transparency_reactive_scale");

    godot::ClassDB::bind_method(godot::D_METHOD("reload_shaders"), &ChronosVPUCompositorEffect::reload_shaders);
    godot::ClassDB::bind_method(godot::D_METHOD("get_dilated_velocity_rid", "view"), &ChronosVPUCompositorEffect::get_dilated_velocity_rid, DEFVAL(0));
    godot::ClassDB::bind_method(godot::D_METHOD("get_reactive_mask_rid", "view"), &ChronosVPUCompositorEffect::get_reactive_mask_rid, DEFVAL(0));
    godot::ClassDB::bind_method(godot::D_METHOD("get_taau_output_rid", "view"), &ChronosVPUCompositorEffect::get_taau_output_rid, DEFVAL(0));
    godot::ClassDB::bind_method(godot::D_METHOD("get_bloom_texture_rid", "view"), &ChronosVPUCompositorEffect::get_bloom_texture_rid, DEFVAL(0));
}

void ChronosVPUCompositorEffect::set_dilate_velocity_enabled(bool p_enabled) { dilate_velocity_enabled = p_enabled; }
bool ChronosVPUCompositorEffect::is_dilate_velocity_enabled() const { return dilate_velocity_enabled; }

void ChronosVPUCompositorEffect::set_reactive_mask_enabled(bool p_enabled) { reactive_mask_enabled = p_enabled; }
bool ChronosVPUCompositorEffect::is_reactive_mask_enabled() const { return reactive_mask_enabled; }

void ChronosVPUCompositorEffect::set_taau_enabled(bool p_enabled) {
    if (taau_enabled != p_enabled) {
        taau_enabled = p_enabled;
        taau_reset_requested = true;
    }
}
bool ChronosVPUCompositorEffect::is_taau_enabled() const { return taau_enabled; }

void ChronosVPUCompositorEffect::set_fft_bloom_enabled(bool p_enabled) { fft_bloom_enabled = p_enabled; }
bool ChronosVPUCompositorEffect::is_fft_bloom_enabled() const { return fft_bloom_enabled; }

void ChronosVPUCompositorEffect::set_aces_odt_enabled(bool p_enabled) { aces_odt_enabled = p_enabled; }
bool ChronosVPUCompositorEffect::is_aces_odt_enabled() const { return aces_odt_enabled; }

void ChronosVPUCompositorEffect::set_taau_sharpness(float p_sharpness) { taau_sharpness = p_sharpness; }
float ChronosVPUCompositorEffect::get_taau_sharpness() const { return taau_sharpness; }

void ChronosVPUCompositorEffect::set_taau_feedback_min(float p_min) { taau_feedback_min = p_min; }
float ChronosVPUCompositorEffect::get_taau_feedback_min() const { return taau_feedback_min; }

void ChronosVPUCompositorEffect::set_taau_feedback_max(float p_max) { taau_feedback_max = p_max; }
float ChronosVPUCompositorEffect::get_taau_feedback_max() const { return taau_feedback_max; }

void ChronosVPUCompositorEffect::reset_taau_history() {
    taau_reset_requested = true;
}

void ChronosVPUCompositorEffect::set_output_display_mode(int32_t p_mode) {
    if (p_mode != DISPLAY_MODE_SDR_SRGB && p_mode != output_display_mode) {
        godot::UtilityFunctions::push_warning(
            "[Prism RP] Display P3 / HDR10 output needs an HDR swapchain, which Godot 4.4 lacks. Rendering SDR sRGB."
        );
    }
    output_display_mode = p_mode;
}
int32_t ChronosVPUCompositorEffect::get_output_display_mode() const { return output_display_mode; }

void ChronosVPUCompositorEffect::set_debug_view_mode(int32_t p_mode) { debug_view_mode = p_mode; }
int32_t ChronosVPUCompositorEffect::get_debug_view_mode() const { return debug_view_mode; }

void ChronosVPUCompositorEffect::set_exposure_ev(float p_ev) { exposure_ev = p_ev; }
float ChronosVPUCompositorEffect::get_exposure_ev() const { return exposure_ev; }

void ChronosVPUCompositorEffect::set_peak_luminance_nits(float p_nits) { peak_luminance_nits = p_nits; }
float ChronosVPUCompositorEffect::get_peak_luminance_nits() const { return peak_luminance_nits; }

void ChronosVPUCompositorEffect::set_bloom_threshold(float p_threshold) { bloom_threshold = p_threshold; }
float ChronosVPUCompositorEffect::get_bloom_threshold() const { return bloom_threshold; }

void ChronosVPUCompositorEffect::set_bloom_intensity(float p_intensity) { bloom_intensity = p_intensity; }
float ChronosVPUCompositorEffect::get_bloom_intensity() const { return bloom_intensity; }

void ChronosVPUCompositorEffect::set_halation_intensity(float p_intensity) { halation_intensity = p_intensity; }
float ChronosVPUCompositorEffect::get_halation_intensity() const { return halation_intensity; }

void ChronosVPUCompositorEffect::set_aperture_blades(float p_blades) { aperture_blades = p_blades; }
float ChronosVPUCompositorEffect::get_aperture_blades() const { return aperture_blades; }

void ChronosVPUCompositorEffect::set_reactive_sensitivity(float p_sensitivity) { reactive_sensitivity = p_sensitivity; }
float ChronosVPUCompositorEffect::get_reactive_sensitivity() const { return reactive_sensitivity; }

void ChronosVPUCompositorEffect::set_transparency_reactive_scale(float p_scale) { transparency_reactive_scale = p_scale; }
float ChronosVPUCompositorEffect::get_transparency_reactive_scale() const { return transparency_reactive_scale; }

void ChronosVPUCompositorEffect::reload_shaders() {
    pipeline_dirty = true;
    taau_reset_requested = true;
}

godot::RID ChronosVPUCompositorEffect::get_dilated_velocity_rid(int32_t p_view) const {
    if (p_view < 0 || static_cast<size_t>(p_view) >= view_targets.size()) {
        return godot::RID();
    }
    return view_targets[static_cast<size_t>(p_view)].dilated_velocity;
}

godot::RID ChronosVPUCompositorEffect::get_reactive_mask_rid(int32_t p_view) const {
    if (p_view < 0 || static_cast<size_t>(p_view) >= view_targets.size()) {
        return godot::RID();
    }
    return view_targets[static_cast<size_t>(p_view)].reactive_mask;
}

godot::RID ChronosVPUCompositorEffect::get_taau_output_rid(int32_t p_view) const {
    if (p_view < 0 || static_cast<size_t>(p_view) >= view_targets.size()) {
        return godot::RID();
    }
    // Most recently written TAA history (the ping-pong index already advanced past it).
    const PerViewTargets &vt = view_targets[static_cast<size_t>(p_view)];
    return vt.taau_history[1u - vt.taau_write_index];
}

godot::RID ChronosVPUCompositorEffect::get_bloom_texture_rid(int32_t p_view) const {
    if (p_view < 0 || static_cast<size_t>(p_view) >= view_targets.size()) {
        return godot::RID();
    }
    return view_targets[static_cast<size_t>(p_view)].bloom_spatial;
}

void ChronosVPUCompositorEffect::_render_callback(int32_t p_effect_callback_type, godot::RenderData *p_render_data) {
    if (p_render_data == nullptr) {
        return;
    }

    if (rd == nullptr) {
        rd = godot::RenderingServer::get_singleton()->get_rendering_device();
        if (rd == nullptr) {
            return;
        }
    }

    if (pipeline_dirty) {
        _release_render_resources();
        if (!_ensure_render_resources()) {
            pipeline_dirty = false;
            return;
        }
        pipeline_dirty = false;
    }

    if (!dilate_pipeline.is_valid() || !reactive_pipeline.is_valid() ||
        !taau_pipeline.is_valid() || !fft_horiz_pipeline.is_valid() ||
        !fft_vert_pipeline.is_valid() || !fft_inv_horiz_pipeline.is_valid() ||
        !aces_odt_pipeline.is_valid()) {
        return;
    }

    godot::Ref<godot::RenderSceneBuffers> base_buffers = p_render_data->get_render_scene_buffers();
    auto *scene_buffers = godot::Object::cast_to<godot::RenderSceneBuffersRD>(base_buffers.ptr());
    if (scene_buffers == nullptr) {
        return;
    }

    // Everything runs at the internal resolution: Godot upscales to the target size after us.
    const godot::Vector2i internal_size = scene_buffers->get_internal_size();
    const uint32_t view_count = scene_buffers->get_view_count();
    if (internal_size.x <= 0 || internal_size.y <= 0 || view_count == 0) {
        return;
    }

    if (!_ensure_view_targets(internal_size, view_count)) {
        return;
    }

    // Godot's own temporal paths (TAA, FSR2, MetalFX Temporal) jitter the projection and run
    // their own history after us. Two temporal resolves fight each other (double blur, and the
    // jitter would be stripped twice), so Chronos TAA stands down while one is active.
    const godot::RenderingServer::ViewportScaling3DMode scaling_mode = scene_buffers->get_scaling_3d_mode();
    const bool godot_temporal_active =
        scene_buffers->get_use_taa() ||
        scaling_mode == godot::RenderingServer::VIEWPORT_SCALING_3D_MODE_FSR2 ||
        scaling_mode == godot::RenderingServer::VIEWPORT_SCALING_3D_MODE_METALFX_TEMPORAL;
    if (godot_temporal_active && taau_enabled && !godot_temporal_warned) {
        godot::UtilityFunctions::push_warning(
            "[Prism RP] Godot TAA / FSR2 / MetalFX Temporal is active on this viewport; Chronos TAA is "
            "bypassed to avoid a double temporal resolve. Disable Viewport.use_taa and use a non-temporal "
            "scaling_3d_mode (e.g. Bilinear) to use Chronos TAA."
        );
        godot_temporal_warned = true;
    } else if (!godot_temporal_active) {
        godot_temporal_warned = false;
    }
    const bool chronos_taa_active = taau_enabled && !godot_temporal_active;
    if (!chronos_taa_active) {
        taau_reset_requested = true;  // history goes stale while bypassed; start fresh on resume
    }

    godot::RenderSceneData *scene_data = p_render_data->get_render_scene_data();
    const godot::Vector2 internal_size_f(static_cast<float>(internal_size.x), static_cast<float>(internal_size.y));

    // Particle/transparency reactivity: only when the opaque capture ran for this frame.
    const bool has_opaque_capture =
        ChronosOpaqueCaptureEffect::consume_capture(scene_buffers->get_instance_id()) &&
        scene_buffers->has_texture(ChronosOpaqueCaptureEffect::kContext, ChronosOpaqueCaptureEffect::kTextureName);

    // Full array textures for copies (texture_copy addresses layers of the underlying image,
    // so a per-view slice with layer 0 would always copy eye 0).
    const godot::RID color_full = scene_buffers->get_color_texture();

    const float exposure_scale = std::exp2(exposure_ev);
    const float aspect_ratio = internal_size_f.x / internal_size_f.y;
    const bool bloom_contributes = fft_bloom_enabled && bloom_intensity > 0.0f;
    const uint32_t groups_x = static_cast<uint32_t>((internal_size.x + 7) / 8);
    const uint32_t groups_y = static_cast<uint32_t>((internal_size.y + 7) / 8);

    const bool reset_history_this_frame = taau_reset_requested;
    taau_reset_requested = false;

    for (uint32_t view = 0; view < view_count; ++view) {
        PerViewTargets &vt = view_targets[view];
        const godot::RID color_tex = scene_buffers->get_color_layer(view);
        const godot::RID depth_tex = scene_buffers->get_depth_layer(view);
        const godot::RID velocity_tex = scene_buffers->get_velocity_layer(view);

        if (!color_tex.is_valid() || !color_full.is_valid() || !depth_tex.is_valid() || !velocity_tex.is_valid()) {
            continue;
        }

        // Per-eye, depth-corrected projection (flip-Y, reverse-Z [0,1]). Each XR eye has its own
        // asymmetric frustum, so jitter and depth linearization can't come from the shared camera.
        // An off-center frustum shifts the image by -0.5 * proj[2].xy in UV (+y down, matching
        // Godot's velocity). Under Godot's own temporal AA the projection carries Godot's jitter,
        // which Godot already strips from velocity, so it must not be treated as ours.
        godot::Vector2 jitter_uv(0.0f, 0.0f);
        float depth_param_a = 0.05f;
        float depth_param_b = 0.0f;
        if (scene_data != nullptr) {
            const godot::Projection view_proj = scene_data->get_view_projection(view);
            if (!godot_temporal_active) {
                jitter_uv = godot::Vector2(
                    -0.5f * static_cast<float>(view_proj.columns[2].x),
                    -0.5f * static_cast<float>(view_proj.columns[2].y)
                );
            }
            depth_param_a = static_cast<float>(view_proj.columns[3].z);
            depth_param_b = static_cast<float>(view_proj.columns[2].z);
        }

        // Only the sub-pixel part counts as TAA jitter, so a deliberately off-center frustum
        // (XR eyes, lens shift) isn't treated as a huge jitter.
        const godot::Vector2 jitter_full_px = jitter_uv * internal_size_f;
        const godot::Vector2 jitter_px(
            jitter_full_px.x - std::round(jitter_full_px.x),
            jitter_full_px.y - std::round(jitter_full_px.y)
        );

        // (current - previous) jitter: Godot's velocity includes camera-applied jitter.
        const godot::Vector2 jitter_delta_uv = vt.has_prev_jitter ? (jitter_uv - vt.prev_jitter_uv) : godot::Vector2(0.0f, 0.0f);
        vt.prev_jitter_uv = jitter_uv;
        vt.has_prev_jitter = true;

        const bool first_frame_for_view = !vt.has_valid_history || reset_history_this_frame;
        if (!vt.has_valid_history) {
            rd->texture_copy(
                color_full,
                vt.prev_color_history,
                godot::Vector3(0, 0, 0),
                godot::Vector3(0, 0, 0),
                godot::Vector3(internal_size.x, internal_size.y, 1),
                0, 0, view, 0
            );
            vt.has_valid_history = true;
        }

        // TAA consumes dilated velocity and the reactive mask, so it forces both passes on.
        const bool run_dilate = dilate_velocity_enabled || chronos_taa_active || reactive_mask_enabled || debug_view_mode == DEBUG_VIEW_DILATED_VELOCITY;
        const bool run_reactive = reactive_mask_enabled || chronos_taa_active || debug_view_mode == DEBUG_VIEW_REACTIVE_MASK;

        // PASS 1: 3x3 closest-depth velocity dilation (jitter removed) -> RG16F
        if (run_dilate) {
            ScopedPassTimer timer("chronos_dilate_velocity", godot::Color(0.20f, 0.75f, 0.95f, 1.0f));
            godot::TypedArray<godot::RDUniform> uniforms;
            uniforms.push_back(make_sampler_texture_uniform(0, nearest_sampler, depth_tex));
            uniforms.push_back(make_sampler_texture_uniform(1, nearest_sampler, velocity_tex));
            uniforms.push_back(make_image_uniform(2, vt.dilated_velocity));

            const godot::RID uset = godot::UniformSetCacheRD::get_cache(dilate_shader, 0, uniforms);
            if (uset.is_valid()) {
                DilateVelocityPushConstants pc{
                    internal_size.x, internal_size.y,
                    static_cast<float>(jitter_delta_uv.x), static_cast<float>(jitter_delta_uv.y)
                };
                const auto pc_bytes = make_push_constant_bytes(pc);

                const int64_t cl = rd->compute_list_begin();
                rd->compute_list_bind_compute_pipeline(cl, dilate_pipeline);
                rd->compute_list_bind_uniform_set(cl, uset, 0);
                rd->compute_list_set_push_constant(cl, pc_bytes, static_cast<uint32_t>(pc_bytes.size()));
                rd->compute_list_dispatch(cl, groups_x, groups_y, 1);
                rd->compute_list_end();
            }
        }

        // PASS 2: temporal reactive mask -> R8_UNORM
        if (run_reactive) {
            ScopedPassTimer timer("chronos_reactive_mask", godot::Color(0.95f, 0.55f, 0.15f, 1.0f));
            godot::TypedArray<godot::RDUniform> uniforms;
            uniforms.push_back(make_sampler_texture_uniform(0, nearest_sampler, color_tex));
            uniforms.push_back(make_sampler_texture_uniform(1, linear_sampler, vt.prev_color_history));
            uniforms.push_back(make_sampler_texture_uniform(2, nearest_sampler, depth_tex));
            uniforms.push_back(make_sampler_texture_uniform(3, nearest_sampler, vt.dilated_velocity));
            uniforms.push_back(make_image_uniform(4, vt.reactive_mask));
            // Binding 5 must always be valid; without a capture it's a placeholder and flag bit 0 is off.
            const godot::RID opaque_tex = has_opaque_capture
                ? scene_buffers->get_texture_slice(ChronosOpaqueCaptureEffect::kContext, ChronosOpaqueCaptureEffect::kTextureName, view, 0, 1, 1)
                : vt.prev_color_history;
            uniforms.push_back(make_sampler_texture_uniform(5, nearest_sampler, opaque_tex.is_valid() ? opaque_tex : vt.prev_color_history));

            const godot::RID uset = godot::UniformSetCacheRD::get_cache(reactive_shader, 0, uniforms);
            if (uset.is_valid()) {
                ReactiveMaskPushConstants pc{
                    internal_size.x, internal_size.y,
                    reactive_sensitivity, 2.5f,
                    static_cast<float>(jitter_delta_uv.x), static_cast<float>(jitter_delta_uv.y),
                    depth_param_a, depth_param_b,
                    transparency_reactive_scale,
                    has_opaque_capture ? 1u : 0u,
                    0.0f, 0.0f
                };
                const auto pc_bytes = make_push_constant_bytes(pc);

                const int64_t cl = rd->compute_list_begin();
                rd->compute_list_bind_compute_pipeline(cl, reactive_pipeline);
                rd->compute_list_bind_uniform_set(cl, uset, 0);
                rd->compute_list_set_push_constant(cl, pc_bytes, static_cast<uint32_t>(pc_bytes.size()));
                rd->compute_list_dispatch(cl, groups_x, groups_y, 1);
                rd->compute_list_end();
            }
        }

        // Raw frame copy: next frame's shock reference, and this frame's composite/bloom input
        // when TAA is off (so the color buffer is never sampled and written in the same pass).
        rd->texture_copy(
            color_full,
            vt.prev_color_history,
            godot::Vector3(0, 0, 0),
            godot::Vector3(0, 0, 0),
            godot::Vector3(internal_size.x, internal_size.y, 1),
            0, 0, view, 0
        );

        // PASS 3: compute TAA at internal resolution
        godot::RID resolved_linear_hdr = vt.prev_color_history;
        bool resolved_has_confidence = false;

        if (chronos_taa_active) {
            ScopedPassTimer timer("chronos_taau", godot::Color(0.25f, 0.88f, 0.45f, 1.0f));
            const uint32_t write_idx = vt.taau_write_index;
            const uint32_t read_idx = 1u - write_idx;
            const godot::RID history_read_tex = vt.taau_history[read_idx];
            const godot::RID history_write_tex = vt.taau_history[write_idx];

            godot::TypedArray<godot::RDUniform> u_taau;
            u_taau.push_back(make_sampler_texture_uniform(0, nearest_sampler, color_tex));
            u_taau.push_back(make_sampler_texture_uniform(1, linear_sampler, history_read_tex));
            u_taau.push_back(make_sampler_texture_uniform(2, nearest_sampler, depth_tex));
            u_taau.push_back(make_sampler_texture_uniform(3, nearest_sampler, vt.dilated_velocity));
            u_taau.push_back(make_sampler_texture_uniform(4, linear_sampler, vt.reactive_mask));
            u_taau.push_back(make_image_uniform(5, history_write_tex));

            const godot::RID uset = godot::UniformSetCacheRD::get_cache(taau_shader, 0, u_taau);
            if (uset.is_valid()) {
                const uint32_t taau_flags = first_frame_for_view ? 1u : 0u;

                TAAUPushConstants pc{
                    internal_size.x,
                    internal_size.y,
                    internal_size.x,
                    internal_size.y,
                    static_cast<float>(jitter_px.x),
                    static_cast<float>(jitter_px.y),
                    taau_feedback_min,
                    taau_feedback_max,
                    18.0f,
                    frame_counter,
                    taau_flags,
                    0.0f
                };
                const auto pc_bytes = make_push_constant_bytes(pc);

                const int64_t cl = rd->compute_list_begin();
                rd->compute_list_bind_compute_pipeline(cl, taau_pipeline);
                rd->compute_list_bind_uniform_set(cl, uset, 0);
                rd->compute_list_set_push_constant(cl, pc_bytes, static_cast<uint32_t>(pc_bytes.size()));
                rd->compute_list_dispatch(cl, groups_x, groups_y, 1);
                rd->compute_list_end();

                vt.taau_write_index = read_idx;
                resolved_linear_hdr = history_write_tex;
                resolved_has_confidence = true;
            }
        }

        // PASS 4: 256x256 radix-2 Stockham FFT bloom & red halation
        // Skipped entirely when it can't reach the screen. The three dispatches get their own
        // (non-nested, so stage totals aren't double-counted) timers for the budget breakdown.
        const bool run_bloom = bloom_contributes || debug_view_mode == DEBUG_VIEW_FFT_BLOOM;
        if (run_bloom) {
            {
                ScopedPassTimer timer("chronos_fft_bloom_forward_rows", godot::Color(0.85f, 0.35f, 0.90f, 1.0f));
                godot::TypedArray<godot::RDUniform> u_horiz;
                u_horiz.push_back(make_sampler_texture_uniform(0, linear_sampler, resolved_linear_hdr));
                u_horiz.push_back(make_image_uniform(1, vt.fft_ping_rg));
                u_horiz.push_back(make_image_uniform(2, vt.fft_ping_b));

                const godot::RID uset = godot::UniformSetCacheRD::get_cache(fft_horiz_shader, 0, u_horiz);
                if (uset.is_valid()) {
                    FFTForwardHorizPushConstants pc{ bloom_threshold, 0.5f, exposure_scale, 0.0f };
                    const auto pc_bytes = make_push_constant_bytes(pc);

                    const int64_t cl = rd->compute_list_begin();
                    rd->compute_list_bind_compute_pipeline(cl, fft_horiz_pipeline);
                    rd->compute_list_bind_uniform_set(cl, uset, 0);
                    rd->compute_list_set_push_constant(cl, pc_bytes, static_cast<uint32_t>(pc_bytes.size()));
                    rd->compute_list_dispatch(cl, 1, kFFTGridSize, 1);
                    rd->compute_list_end();
                }
            }

            {
                ScopedPassTimer timer("chronos_fft_bloom_columns_convolve", godot::Color(0.80f, 0.30f, 0.85f, 1.0f));
                godot::TypedArray<godot::RDUniform> u_vert;
                u_vert.push_back(make_image_uniform(0, vt.fft_ping_rg));
                u_vert.push_back(make_image_uniform(1, vt.fft_ping_b));
                u_vert.push_back(make_image_uniform(2, vt.fft_pong_rg));
                u_vert.push_back(make_image_uniform(3, vt.fft_pong_b));

                const godot::RID uset = godot::UniformSetCacheRD::get_cache(fft_vert_shader, 0, u_vert);
                if (uset.is_valid()) {
                    FFTVerticalKernelPushConstants pc{ 1.0f, 0.45f, halation_intensity, aperture_blades, aspect_ratio, 0.0f, 0.0f, 0.0f };
                    const auto pc_bytes = make_push_constant_bytes(pc);

                    const int64_t cl = rd->compute_list_begin();
                    rd->compute_list_bind_compute_pipeline(cl, fft_vert_pipeline);
                    rd->compute_list_bind_uniform_set(cl, uset, 0);
                    rd->compute_list_set_push_constant(cl, pc_bytes, static_cast<uint32_t>(pc_bytes.size()));
                    rd->compute_list_dispatch(cl, kFFTGridSize, 1, 1);
                    rd->compute_list_end();
                }
            }

            {
                ScopedPassTimer timer("chronos_fft_bloom_inverse_rows", godot::Color(0.75f, 0.25f, 0.80f, 1.0f));
                godot::TypedArray<godot::RDUniform> u_inv_horiz;
                u_inv_horiz.push_back(make_image_uniform(0, vt.fft_pong_rg));
                u_inv_horiz.push_back(make_image_uniform(1, vt.fft_pong_b));
                u_inv_horiz.push_back(make_image_uniform(2, vt.bloom_spatial));

                const godot::RID uset = godot::UniformSetCacheRD::get_cache(fft_inv_horiz_shader, 0, u_inv_horiz);
                if (uset.is_valid()) {
                    FFTInvHorizPushConstants pc{ bloom_intensity, 0.0f, 0.0f, 0.0f };
                    const auto pc_bytes = make_push_constant_bytes(pc);

                    const int64_t cl = rd->compute_list_begin();
                    rd->compute_list_bind_compute_pipeline(cl, fft_inv_horiz_pipeline);
                    rd->compute_list_bind_uniform_set(cl, uset, 0);
                    rd->compute_list_set_push_constant(cl, pc_bytes, static_cast<uint32_t>(pc_bytes.size()));
                    rd->compute_list_dispatch(cl, 1, kFFTGridSize, 1);
                    rd->compute_list_end();
                }
            }
        }

        // PASS 5: composite (sharpen, bloom, optional SDR look) -> linear Rec.709 into the color buffer.
        // Must run whenever TAA ran, or the resolved frame would never reach the screen.
        const bool sharpen = chronos_taa_active && resolved_has_confidence && taau_sharpness > 0.001f;
        if (chronos_taa_active || bloom_contributes || aces_odt_enabled || exposure_ev != 0.0f || debug_view_mode != DEBUG_VIEW_FINAL) {
            ScopedPassTimer timer("chronos_aces2_odt", godot::Color(0.95f, 0.82f, 0.22f, 1.0f));
            godot::TypedArray<godot::RDUniform> u_comp;
            u_comp.push_back(make_sampler_texture_uniform(0, nearest_sampler, resolved_linear_hdr));
            u_comp.push_back(make_sampler_texture_uniform(1, linear_sampler, vt.bloom_spatial));
            u_comp.push_back(make_sampler_texture_uniform(2, nearest_sampler, vt.dilated_velocity));
            u_comp.push_back(make_sampler_texture_uniform(3, nearest_sampler, vt.reactive_mask));
            u_comp.push_back(make_image_uniform(4, color_tex));

            const godot::RID uset = godot::UniformSetCacheRD::get_cache(aces_odt_shader, 0, u_comp);
            if (uset.is_valid()) {
                uint32_t flags = 0;
                if (bloom_contributes) flags |= 1u;
                if (aces_odt_enabled) flags |= 2u;
                if (sharpen) flags |= 4u;
                if (resolved_has_confidence) flags |= 8u;

                CompositePushConstants pc{
                    internal_size.x,
                    internal_size.y,
                    exposure_scale,
                    taau_sharpness,
                    debug_view_mode,
                    flags,
                    0.0f,
                    0.0f
                };
                const auto pc_bytes = make_push_constant_bytes(pc);

                const int64_t cl = rd->compute_list_begin();
                rd->compute_list_bind_compute_pipeline(cl, aces_odt_pipeline);
                rd->compute_list_bind_uniform_set(cl, uset, 0);
                rd->compute_list_set_push_constant(cl, pc_bytes, static_cast<uint32_t>(pc_bytes.size()));
                rd->compute_list_dispatch(cl, groups_x, groups_y, 1);
                rd->compute_list_end();
            }
        }
    }

    ++frame_counter;
}

bool ChronosVPUCompositorEffect::_ensure_render_resources() {
    if (rd == nullptr) {
        return false;
    }

    nearest_sampler = rd_utils::create_sampler(rd, false);
    linear_sampler = rd_utils::create_sampler(rd, true);

    if (!nearest_sampler.is_valid() || !linear_sampler.is_valid()) {
        godot::UtilityFunctions::push_error("[Prism RP] Failed to create Chronos VPU samplers.");
        _release_render_resources();
        return false;
    }

    auto build_stage = [&](const char *p_path, const char *p_name, godot::RID &r_shader, godot::RID &r_pipeline) -> bool {
        return rd_utils::build_compute_stage(rd, p_path, p_name, r_shader, r_pipeline);
    };

    const bool ok =
        build_stage("res://addons/prism_rp/shaders/chronos/chronos_dilate_velocity.comp.glsl", "chronos_dilate_velocity", dilate_shader, dilate_pipeline) &&
        build_stage("res://addons/prism_rp/shaders/chronos/chronos_reactive_mask.comp.glsl", "chronos_reactive_mask", reactive_shader, reactive_pipeline) &&
        build_stage("res://addons/prism_rp/shaders/chronos/chronos_taau.comp.glsl", "chronos_taau", taau_shader, taau_pipeline) &&
        build_stage("res://addons/prism_rp/shaders/chronos/chronos_fft_bloom_horizontal.comp.glsl", "chronos_fft_bloom_horiz", fft_horiz_shader, fft_horiz_pipeline) &&
        build_stage("res://addons/prism_rp/shaders/chronos/chronos_fft_bloom_vertical.comp.glsl", "chronos_fft_bloom_vert", fft_vert_shader, fft_vert_pipeline) &&
        build_stage("res://addons/prism_rp/shaders/chronos/chronos_fft_bloom_inv_horizontal.comp.glsl", "chronos_fft_bloom_inv_horiz", fft_inv_horiz_shader, fft_inv_horiz_pipeline) &&
        build_stage("res://addons/prism_rp/shaders/chronos/chronos_aces2_odt.comp.glsl", "chronos_aces2_odt", aces_odt_shader, aces_odt_pipeline);

    if (!ok) {
        _release_render_resources();
        return false;
    }

    return true;
}

godot::RID ChronosVPUCompositorEffect::_create_2d_texture(
    uint32_t p_width,
    uint32_t p_height,
    godot::RenderingDevice::DataFormat p_format,
    uint32_t p_usage_bits
) const {
    return rd_utils::create_2d_texture(rd, p_width, p_height, p_format, p_usage_bits);
}

bool ChronosVPUCompositorEffect::_ensure_view_targets(const godot::Vector2i &p_internal_size, uint32_t p_view_count) {
    if (cached_internal_size == p_internal_size && view_targets.size() == p_view_count) {
        return true;
    }

    _release_view_targets();

    const uint32_t iw = static_cast<uint32_t>(p_internal_size.x);
    const uint32_t ih = static_cast<uint32_t>(p_internal_size.y);

    const uint32_t rw_sample_copy_usage =
        godot::RenderingDevice::TEXTURE_USAGE_STORAGE_BIT |
        godot::RenderingDevice::TEXTURE_USAGE_SAMPLING_BIT |
        godot::RenderingDevice::TEXTURE_USAGE_CAN_COPY_FROM_BIT |
        godot::RenderingDevice::TEXTURE_USAGE_CAN_COPY_TO_BIT;

    const uint32_t history_usage =
        godot::RenderingDevice::TEXTURE_USAGE_SAMPLING_BIT |
        godot::RenderingDevice::TEXTURE_USAGE_CAN_COPY_TO_BIT;

    const uint32_t fft_storage_usage =
        godot::RenderingDevice::TEXTURE_USAGE_STORAGE_BIT;

    view_targets.resize(p_view_count);
    for (uint32_t v = 0; v < p_view_count; ++v) {
        PerViewTargets &vt = view_targets[v];
        vt.dilated_velocity = _create_2d_texture(iw, ih, godot::RenderingDevice::DATA_FORMAT_R16G16_SFLOAT, rw_sample_copy_usage);
        vt.reactive_mask = _create_2d_texture(iw, ih, godot::RenderingDevice::DATA_FORMAT_R8_UNORM, rw_sample_copy_usage);
        vt.prev_color_history = _create_2d_texture(iw, ih, godot::RenderingDevice::DATA_FORMAT_R16G16B16A16_SFLOAT, history_usage);
        vt.taau_history[0] = _create_2d_texture(iw, ih, godot::RenderingDevice::DATA_FORMAT_R16G16B16A16_SFLOAT, rw_sample_copy_usage);
        vt.taau_history[1] = _create_2d_texture(iw, ih, godot::RenderingDevice::DATA_FORMAT_R16G16B16A16_SFLOAT, rw_sample_copy_usage);
        vt.fft_ping_rg = _create_2d_texture(kFFTGridSize, kFFTGridSize, godot::RenderingDevice::DATA_FORMAT_R32G32B32A32_SFLOAT, fft_storage_usage);
        vt.fft_ping_b = _create_2d_texture(kFFTGridSize, kFFTGridSize, godot::RenderingDevice::DATA_FORMAT_R32G32B32A32_SFLOAT, fft_storage_usage);
        vt.fft_pong_rg = _create_2d_texture(kFFTGridSize, kFFTGridSize, godot::RenderingDevice::DATA_FORMAT_R32G32B32A32_SFLOAT, fft_storage_usage);
        vt.fft_pong_b = _create_2d_texture(kFFTGridSize, kFFTGridSize, godot::RenderingDevice::DATA_FORMAT_R32G32B32A32_SFLOAT, fft_storage_usage);
        vt.bloom_spatial = _create_2d_texture(kFFTGridSize, kFFTGridSize, godot::RenderingDevice::DATA_FORMAT_R16G16B16A16_SFLOAT, rw_sample_copy_usage);
        vt.taau_write_index = 0;
        vt.has_valid_history = false;
        vt.has_prev_jitter = false;

        if (!vt.dilated_velocity.is_valid() || !vt.reactive_mask.is_valid() ||
            !vt.prev_color_history.is_valid() || !vt.taau_history[0].is_valid() ||
            !vt.taau_history[1].is_valid() ||
            !vt.fft_ping_rg.is_valid() || !vt.fft_ping_b.is_valid() ||
            !vt.fft_pong_rg.is_valid() || !vt.fft_pong_b.is_valid() ||
            !vt.bloom_spatial.is_valid()) {
            godot::UtilityFunctions::push_error("[Prism RP] Failed to allocate Chronos VPU transient render targets.");
            _release_view_targets();
            return false;
        }
    }

    cached_internal_size = p_internal_size;
    taau_reset_requested = true;
    return true;
}

void ChronosVPUCompositorEffect::_release_render_resources() {
    if (rd == nullptr) {
        return;
    }

    auto free_pipeline_and_shader = [&](godot::RID &r_pipeline, godot::RID &r_shader) {
        rd_utils::free_compute_stage(rd, r_pipeline, r_shader);
    };

    free_pipeline_and_shader(dilate_pipeline, dilate_shader);
    free_pipeline_and_shader(reactive_pipeline, reactive_shader);
    free_pipeline_and_shader(taau_pipeline, taau_shader);
    free_pipeline_and_shader(fft_horiz_pipeline, fft_horiz_shader);
    free_pipeline_and_shader(fft_vert_pipeline, fft_vert_shader);
    free_pipeline_and_shader(fft_inv_horiz_pipeline, fft_inv_horiz_shader);
    free_pipeline_and_shader(aces_odt_pipeline, aces_odt_shader);

    if (nearest_sampler.is_valid()) {
        rd->free_rid(nearest_sampler);
        nearest_sampler = godot::RID();
    }
    if (linear_sampler.is_valid()) {
        rd->free_rid(linear_sampler);
        linear_sampler = godot::RID();
    }
}

void ChronosVPUCompositorEffect::_release_view_targets() {
    if (rd != nullptr) {
        auto free_tex = [&](godot::RID &r_tex) {
            if (r_tex.is_valid()) {
                rd->free_rid(r_tex);
                r_tex = godot::RID();
            }
        };

        for (PerViewTargets &vt : view_targets) {
            free_tex(vt.dilated_velocity);
            free_tex(vt.reactive_mask);
            free_tex(vt.prev_color_history);
            free_tex(vt.taau_history[0]);
            free_tex(vt.taau_history[1]);
            free_tex(vt.fft_ping_rg);
            free_tex(vt.fft_ping_b);
            free_tex(vt.fft_pong_rg);
            free_tex(vt.fft_pong_b);
            free_tex(vt.bloom_spatial);
        }
    }
    view_targets.clear();
    cached_internal_size = godot::Vector2i(0, 0);
}

} // namespace prism
