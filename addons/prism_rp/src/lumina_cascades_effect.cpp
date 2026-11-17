#include "prism_rp/lumina_cascades_effect.h"
#include "prism_rp/prism_rd_utils.h"
#include "prism_rp/prism_telemetry.h"

#include <algorithm>
#include <cmath>

#include <godot_cpp/classes/render_scene_buffers.hpp>
#include <godot_cpp/classes/render_scene_buffers_rd.hpp>
#include <godot_cpp/classes/render_scene_data.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/classes/uniform_set_cache_rd.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/projection.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

namespace prism {

using rd_utils::dispatch_compute;
using rd_utils::make_image_uniform;
using rd_utils::make_push_constant_bytes;
using rd_utils::make_sampler_texture_uniform;

namespace {

constexpr const char *kShaderDir = "res://addons/prism_rp/shaders/lumina/";
constexpr float kMaxHistory = 32.0f;

uint32_t groups_for(int32_t p_size) {
    return static_cast<uint32_t>((p_size + 7) / 8);
}

} // namespace

LuminaCascadesCompositorEffect::LuminaCascadesCompositorEffect() {
    set_effect_callback_type(EFFECT_CALLBACK_TYPE_POST_OPAQUE);
    set_access_resolved_color(true);
    set_access_resolved_depth(true);
    set_needs_motion_vectors(true);
    set_needs_normal_roughness(true);

    if (godot::RenderingServer::get_singleton() != nullptr) {
        rd = godot::RenderingServer::get_singleton()->get_rendering_device();
    }
}

LuminaCascadesCompositorEffect::~LuminaCascadesCompositorEffect() {
    _release_view_targets();
    _release_render_resources();
}

void LuminaCascadesCompositorEffect::_bind_methods() {
    using godot::ClassDB;
    using godot::D_METHOD;
    using godot::PropertyInfo;
    using godot::Variant;

    ClassDB::bind_method(D_METHOD("set_gi_enabled", "enabled"), &LuminaCascadesCompositorEffect::set_gi_enabled);
    ClassDB::bind_method(D_METHOD("is_gi_enabled"), &LuminaCascadesCompositorEffect::is_gi_enabled);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "gi_enabled"), "set_gi_enabled", "is_gi_enabled");

    ClassDB::bind_method(D_METHOD("set_temporal_enabled", "enabled"), &LuminaCascadesCompositorEffect::set_temporal_enabled);
    ClassDB::bind_method(D_METHOD("is_temporal_enabled"), &LuminaCascadesCompositorEffect::is_temporal_enabled);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "temporal_enabled"), "set_temporal_enabled", "is_temporal_enabled");

    ClassDB::bind_method(D_METHOD("set_atrous_iterations", "iterations"), &LuminaCascadesCompositorEffect::set_atrous_iterations);
    ClassDB::bind_method(D_METHOD("get_atrous_iterations"), &LuminaCascadesCompositorEffect::get_atrous_iterations);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "atrous_iterations", godot::PROPERTY_HINT_RANGE, "0,5,1"), "set_atrous_iterations", "get_atrous_iterations");

    ClassDB::bind_method(D_METHOD("set_cascade_count", "count"), &LuminaCascadesCompositorEffect::set_cascade_count);
    ClassDB::bind_method(D_METHOD("get_cascade_count"), &LuminaCascadesCompositorEffect::get_cascade_count);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "cascade_count", godot::PROPERTY_HINT_RANGE, "1,4,1"), "set_cascade_count", "get_cascade_count");

    ClassDB::bind_method(D_METHOD("set_probe_spacing", "pixels"), &LuminaCascadesCompositorEffect::set_probe_spacing);
    ClassDB::bind_method(D_METHOD("get_probe_spacing"), &LuminaCascadesCompositorEffect::get_probe_spacing);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "probe_spacing", godot::PROPERTY_HINT_ENUM, "4 px:4,8 px:8,16 px:16"), "set_probe_spacing", "get_probe_spacing");

    ClassDB::bind_method(D_METHOD("set_steps_per_interval", "steps"), &LuminaCascadesCompositorEffect::set_steps_per_interval);
    ClassDB::bind_method(D_METHOD("get_steps_per_interval"), &LuminaCascadesCompositorEffect::get_steps_per_interval);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "steps_per_interval", godot::PROPERTY_HINT_RANGE, "2,32,1"), "set_steps_per_interval", "get_steps_per_interval");

    ClassDB::bind_method(D_METHOD("set_thickness", "thickness"), &LuminaCascadesCompositorEffect::set_thickness);
    ClassDB::bind_method(D_METHOD("get_thickness"), &LuminaCascadesCompositorEffect::get_thickness);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "thickness", godot::PROPERTY_HINT_RANGE, "0.01,0.5,0.01"), "set_thickness", "get_thickness");

    ClassDB::bind_method(D_METHOD("set_intensity", "intensity"), &LuminaCascadesCompositorEffect::set_intensity);
    ClassDB::bind_method(D_METHOD("get_intensity"), &LuminaCascadesCompositorEffect::get_intensity);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "intensity", godot::PROPERTY_HINT_RANGE, "0.0,4.0,0.05"), "set_intensity", "get_intensity");

    ClassDB::bind_method(D_METHOD("set_denoise_strength", "strength"), &LuminaCascadesCompositorEffect::set_denoise_strength);
    ClassDB::bind_method(D_METHOD("get_denoise_strength"), &LuminaCascadesCompositorEffect::get_denoise_strength);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "denoise_strength", godot::PROPERTY_HINT_RANGE, "0.5,16.0,0.5"), "set_denoise_strength", "get_denoise_strength");

    ClassDB::bind_method(D_METHOD("set_albedo_mode", "mode"), &LuminaCascadesCompositorEffect::set_albedo_mode);
    ClassDB::bind_method(D_METHOD("get_albedo_mode"), &LuminaCascadesCompositorEffect::get_albedo_mode);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "albedo_mode", godot::PROPERTY_HINT_ENUM, "Albedo Proxy:0,White (lighting only):1"), "set_albedo_mode", "get_albedo_mode");

    ClassDB::bind_method(D_METHOD("set_debug_view_mode", "mode"), &LuminaCascadesCompositorEffect::set_debug_view_mode);
    ClassDB::bind_method(D_METHOD("get_debug_view_mode"), &LuminaCascadesCompositorEffect::get_debug_view_mode);
    ADD_PROPERTY(
        PropertyInfo(Variant::INT, "debug_view_mode", godot::PROPERTY_HINT_ENUM,
            "Final Output:0,Raw Irradiance:1,Denoised Irradiance:2,History Length:3,Variance:4,Normals:5,Albedo Proxy:6"),
        "set_debug_view_mode", "get_debug_view_mode");

    ClassDB::bind_method(D_METHOD("reset_history"), &LuminaCascadesCompositorEffect::reset_history);
    ClassDB::bind_method(D_METHOD("reload_shaders"), &LuminaCascadesCompositorEffect::reload_shaders);
    ClassDB::bind_method(D_METHOD("get_irradiance_rid", "view"), &LuminaCascadesCompositorEffect::get_irradiance_rid, DEFVAL(0));
}

void LuminaCascadesCompositorEffect::set_gi_enabled(bool p_enabled) {
    if (gi_enabled != p_enabled) {
        gi_enabled = p_enabled;
        history_reset_requested = true;
    }
}
bool LuminaCascadesCompositorEffect::is_gi_enabled() const { return gi_enabled; }

void LuminaCascadesCompositorEffect::set_temporal_enabled(bool p_enabled) { temporal_enabled = p_enabled; }
bool LuminaCascadesCompositorEffect::is_temporal_enabled() const { return temporal_enabled; }

void LuminaCascadesCompositorEffect::set_atrous_iterations(int32_t p_iterations) { atrous_iterations = std::clamp(p_iterations, 0, 5); }
int32_t LuminaCascadesCompositorEffect::get_atrous_iterations() const { return atrous_iterations; }

void LuminaCascadesCompositorEffect::set_cascade_count(int32_t p_count) { cascade_count = std::clamp(p_count, 1, kMaxCascades); }
int32_t LuminaCascadesCompositorEffect::get_cascade_count() const { return cascade_count; }

void LuminaCascadesCompositorEffect::set_probe_spacing(int32_t p_spacing) {
    probe_spacing = (p_spacing <= 4) ? 4 : (p_spacing <= 8 ? 8 : 16);
}
int32_t LuminaCascadesCompositorEffect::get_probe_spacing() const { return probe_spacing; }

void LuminaCascadesCompositorEffect::set_steps_per_interval(int32_t p_steps) { steps_per_interval = std::clamp(p_steps, 2, 32); }
int32_t LuminaCascadesCompositorEffect::get_steps_per_interval() const { return steps_per_interval; }

void LuminaCascadesCompositorEffect::set_thickness(float p_thickness) { thickness = p_thickness; }
float LuminaCascadesCompositorEffect::get_thickness() const { return thickness; }

void LuminaCascadesCompositorEffect::set_intensity(float p_intensity) { intensity = p_intensity; }
float LuminaCascadesCompositorEffect::get_intensity() const { return intensity; }

void LuminaCascadesCompositorEffect::set_denoise_strength(float p_strength) { denoise_strength = p_strength; }
float LuminaCascadesCompositorEffect::get_denoise_strength() const { return denoise_strength; }

void LuminaCascadesCompositorEffect::set_albedo_mode(int32_t p_mode) { albedo_mode = p_mode; }
int32_t LuminaCascadesCompositorEffect::get_albedo_mode() const { return albedo_mode; }

void LuminaCascadesCompositorEffect::set_debug_view_mode(int32_t p_mode) { debug_view_mode = p_mode; }
int32_t LuminaCascadesCompositorEffect::get_debug_view_mode() const { return debug_view_mode; }

void LuminaCascadesCompositorEffect::reset_history() { history_reset_requested = true; }

void LuminaCascadesCompositorEffect::reload_shaders() {
    pipeline_dirty = true;
    history_reset_requested = true;
}

godot::RID LuminaCascadesCompositorEffect::get_irradiance_rid(int32_t p_view) const {
    if (p_view < 0 || static_cast<size_t>(p_view) >= view_targets.size()) {
        return godot::RID();
    }
    return view_targets[static_cast<size_t>(p_view)].irradiance_raw;
}

void LuminaCascadesCompositorEffect::_render_callback(int32_t p_effect_callback_type, godot::RenderData *p_render_data) {
    if (p_render_data == nullptr || !gi_enabled) {
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
        _ensure_render_resources();
        pipeline_dirty = false;
    }
    if (!trace_pipeline.is_valid() || !merge_pipeline.is_valid() || !integrate_pipeline.is_valid() ||
        !temporal_pipeline.is_valid() || !atrous_pipeline.is_valid() || !composite_pipeline.is_valid()) {
        return;
    }

    godot::Ref<godot::RenderSceneBuffers> base_buffers = p_render_data->get_render_scene_buffers();
    auto *scene_buffers = godot::Object::cast_to<godot::RenderSceneBuffersRD>(base_buffers.ptr());
    godot::RenderSceneData *scene_data = p_render_data->get_render_scene_data();
    if (scene_buffers == nullptr || scene_data == nullptr) {
        return;
    }

    const godot::Vector2i size = scene_buffers->get_internal_size();
    const uint32_t view_count = scene_buffers->get_view_count();
    if (size.x <= 0 || size.y <= 0 || view_count == 0) {
        return;
    }
    if (!scene_buffers->has_texture("forward_clustered", "normal_roughness")) {
        return;  // only Forward+ provides normal_roughness
    }
    if (!_ensure_view_targets(size, view_count)) {
        return;
    }

    const bool reset_history = history_reset_requested || !temporal_enabled;
    history_reset_requested = false;
    const uint32_t gx = groups_for(size.x);
    const uint32_t gy = groups_for(size.y);

    for (uint32_t view = 0; view < view_count; ++view) {
        PerViewTargets &vt = view_targets[view];
        const godot::RID color_tex = scene_buffers->get_color_layer(view);
        const godot::RID depth_tex = scene_buffers->get_depth_layer(view);
        const godot::RID velocity_tex = scene_buffers->get_velocity_layer(view);
        const godot::RID normal_tex = scene_buffers->get_texture_slice("forward_clustered", "normal_roughness", view, 0, 1, 1);
        if (!color_tex.is_valid() || !depth_tex.is_valid() || !velocity_tex.is_valid() || !normal_tex.is_valid()) {
            continue;
        }

        // Per-eye depth-corrected projection (see lumina_view_position).
        const godot::Projection proj = scene_data->get_view_projection(view);
        const float proj4[4] = {
            static_cast<float>(proj.columns[0].x), static_cast<float>(proj.columns[1].y),
            static_cast<float>(proj.columns[2].x), static_cast<float>(proj.columns[2].y)
        };
        const float depth_params[2] = { static_cast<float>(proj.columns[3].z), static_cast<float>(proj.columns[2].z) };

        // 1. Trace every level.
        {
            ScopedPassTimer timer("lumina_rc_trace", godot::Color(0.30f, 0.65f, 1.00f, 1.0f));
            for (int32_t level = 0; level < cascade_count; ++level) {
                const CascadeLevel &cl = vt.cascades[static_cast<size_t>(level)];
                godot::TypedArray<godot::RDUniform> u;
                u.push_back(make_sampler_texture_uniform(0, nearest_sampler, depth_tex));
                u.push_back(make_sampler_texture_uniform(1, nearest_sampler, normal_tex));
                u.push_back(make_sampler_texture_uniform(2, linear_sampler, color_tex));
                u.push_back(make_image_uniform(3, cl.texture));
                const godot::RID uset = godot::UniformSetCacheRD::get_cache(trace_shader, 0, u);
                if (!uset.is_valid()) {
                    continue;
                }

                // Interval i covers [l0 * (4^i - 1) / 3, l0 * (4^(i+1) - 1) / 3) screen pixels, l0 = probe spacing.
                const float l0 = static_cast<float>(probe_spacing);
                const float scale = std::pow(4.0f, static_cast<float>(level));
                TracePushConstants pc;
                pc.internal_res_x = size.x;
                pc.internal_res_y = size.y;
                pc.probe_count_x = cl.probe_count.x;
                pc.probe_count_y = cl.probe_count.y;
                pc.block = cl.block;
                pc.probe_spacing_px = cl.spacing_px;
                pc.interval_start_px = l0 * (scale - 1.0f) / 3.0f;
                pc.interval_length_px = l0 * scale;
                pc.steps = steps_per_interval;
                pc.thickness = thickness;
                pc.frame_index = frame_counter;
                pc.normal_bias = 0.01f;
                std::copy(std::begin(proj4), std::end(proj4), pc.proj);
                std::copy(std::begin(depth_params), std::end(depth_params), pc.depth_params);

                dispatch_compute(rd, trace_pipeline, uset, make_push_constant_bytes(pc),
                    groups_for(cl.probe_count.x * cl.block), groups_for(cl.probe_count.y * cl.block));
            }
        }

        // 2. Merge top-down.
        if (cascade_count > 1) {
            ScopedPassTimer timer("lumina_rc_merge", godot::Color(0.25f, 0.55f, 0.95f, 1.0f));
            for (int32_t level = cascade_count - 2; level >= 0; --level) {
                const CascadeLevel &cl = vt.cascades[static_cast<size_t>(level)];
                const CascadeLevel &parent = vt.cascades[static_cast<size_t>(level + 1)];
                godot::TypedArray<godot::RDUniform> u;
                u.push_back(make_sampler_texture_uniform(0, nearest_sampler, parent.texture));
                u.push_back(make_image_uniform(1, cl.texture));
                const godot::RID uset = godot::UniformSetCacheRD::get_cache(merge_shader, 0, u);
                if (!uset.is_valid()) {
                    continue;
                }
                MergePushConstants pc{
                    cl.probe_count.x, cl.probe_count.y,
                    parent.probe_count.x, parent.probe_count.y,
                    cl.block, parent.block,
                    cl.spacing_px, parent.spacing_px
                };
                dispatch_compute(rd, merge_pipeline, uset, make_push_constant_bytes(pc),
                    groups_for(cl.probe_count.x * cl.block), groups_for(cl.probe_count.y * cl.block));
            }
        }

        // 3. Integrate irradiance per pixel.
        {
            ScopedPassTimer timer("lumina_rc_integrate", godot::Color(0.20f, 0.50f, 0.90f, 1.0f));
            const CascadeLevel &c0 = vt.cascades[0];
            godot::TypedArray<godot::RDUniform> u;
            u.push_back(make_sampler_texture_uniform(0, nearest_sampler, c0.texture));
            u.push_back(make_sampler_texture_uniform(1, nearest_sampler, depth_tex));
            u.push_back(make_sampler_texture_uniform(2, nearest_sampler, normal_tex));
            u.push_back(make_image_uniform(3, vt.irradiance_raw));
            const godot::RID uset = godot::UniformSetCacheRD::get_cache(integrate_shader, 0, u);
            if (uset.is_valid()) {
                IntegratePushConstants pc;
                pc.internal_res_x = size.x;
                pc.internal_res_y = size.y;
                pc.probe_count_x = c0.probe_count.x;
                pc.probe_count_y = c0.probe_count.y;
                pc.block = c0.block;
                pc.probe_spacing_px = c0.spacing_px;
                std::copy(std::begin(depth_params), std::end(depth_params), pc.depth_params);
                dispatch_compute(rd, integrate_pipeline, uset, make_push_constant_bytes(pc), gx, gy);
            }
        }

        // 4a. SVGF temporal (also runs with temporal off, as a forced reset, to produce variance).
        const uint32_t prev = vt.history_index;
        const uint32_t curr = 1u - prev;
        {
            ScopedPassTimer timer("lumina_svgf_temporal", godot::Color(0.55f, 0.40f, 0.95f, 1.0f));
            godot::TypedArray<godot::RDUniform> u;
            u.push_back(make_sampler_texture_uniform(0, nearest_sampler, vt.irradiance_raw));
            u.push_back(make_sampler_texture_uniform(1, linear_sampler, vt.history[prev]));
            u.push_back(make_sampler_texture_uniform(2, linear_sampler, vt.moments[prev]));
            u.push_back(make_sampler_texture_uniform(3, nearest_sampler, vt.geometry[prev]));
            u.push_back(make_sampler_texture_uniform(4, nearest_sampler, velocity_tex));
            u.push_back(make_sampler_texture_uniform(5, nearest_sampler, depth_tex));
            u.push_back(make_sampler_texture_uniform(6, nearest_sampler, normal_tex));
            u.push_back(make_image_uniform(7, vt.history[curr]));
            u.push_back(make_image_uniform(8, vt.moments[curr]));
            u.push_back(make_image_uniform(9, vt.geometry[curr]));
            u.push_back(make_image_uniform(10, vt.filter[0]));
            const godot::RID uset = godot::UniformSetCacheRD::get_cache(temporal_shader, 0, u);
            if (uset.is_valid()) {
                TemporalPushConstants pc;
                pc.internal_res_x = size.x;
                pc.internal_res_y = size.y;
                std::copy(std::begin(depth_params), std::end(depth_params), pc.depth_params);
                pc.flags = (reset_history || !vt.has_history) ? 1u : 0u;
                pc.max_history = kMaxHistory;
                dispatch_compute(rd, temporal_pipeline, uset, make_push_constant_bytes(pc), gx, gy);
                vt.history_index = curr;
                vt.has_history = true;
            }
        }

        // 4b. SVGF a-trous iterations, ping-ponging filter[0] <-> filter[1].
        uint32_t filtered = 0;
        if (atrous_iterations > 0) {
            ScopedPassTimer timer("lumina_svgf_atrous", godot::Color(0.50f, 0.35f, 0.90f, 1.0f));
            for (int32_t it = 0; it < atrous_iterations; ++it) {
                const uint32_t src = filtered;
                const uint32_t dst = 1u - src;
                godot::TypedArray<godot::RDUniform> u;
                u.push_back(make_sampler_texture_uniform(0, nearest_sampler, vt.filter[src]));
                u.push_back(make_sampler_texture_uniform(1, nearest_sampler, depth_tex));
                u.push_back(make_sampler_texture_uniform(2, nearest_sampler, normal_tex));
                u.push_back(make_image_uniform(3, vt.filter[dst]));
                const godot::RID uset = godot::UniformSetCacheRD::get_cache(atrous_shader, 0, u);
                if (!uset.is_valid()) {
                    break;
                }
                AtrousPushConstants pc;
                pc.internal_res_x = size.x;
                pc.internal_res_y = size.y;
                pc.step_px = 1 << it;
                pc.phi_color = denoise_strength;
                std::copy(std::begin(depth_params), std::end(depth_params), pc.depth_params);
                dispatch_compute(rd, atrous_pipeline, uset, make_push_constant_bytes(pc), gx, gy);
                filtered = dst;
            }
        }

        // 5. Composite into the opaque color buffer.
        {
            ScopedPassTimer timer("lumina_composite", godot::Color(0.45f, 0.75f, 1.00f, 1.0f));
            godot::TypedArray<godot::RDUniform> u;
            u.push_back(make_image_uniform(0, color_tex));
            u.push_back(make_sampler_texture_uniform(1, nearest_sampler, vt.filter[filtered]));
            u.push_back(make_sampler_texture_uniform(2, nearest_sampler, vt.irradiance_raw));
            u.push_back(make_sampler_texture_uniform(3, nearest_sampler, vt.history[vt.history_index]));
            u.push_back(make_sampler_texture_uniform(4, nearest_sampler, depth_tex));
            u.push_back(make_sampler_texture_uniform(5, nearest_sampler, normal_tex));
            const godot::RID uset = godot::UniformSetCacheRD::get_cache(composite_shader, 0, u);
            if (uset.is_valid()) {
                CompositePushConstants pc;
                pc.internal_res_x = size.x;
                pc.internal_res_y = size.y;
                pc.intensity = intensity;
                pc.debug_view_mode = debug_view_mode;
                pc.albedo_mode = albedo_mode;
                pc.max_history = kMaxHistory;
                dispatch_compute(rd, composite_pipeline, uset, make_push_constant_bytes(pc), gx, gy);
            }
        }
    }

    ++frame_counter;
}

bool LuminaCascadesCompositorEffect::_ensure_render_resources() {
    if (rd == nullptr) {
        return false;
    }
    nearest_sampler = rd_utils::create_sampler(rd, false);
    linear_sampler = rd_utils::create_sampler(rd, true);

    auto build = [&](const char *p_file, const char *p_name, godot::RID &r_shader, godot::RID &r_pipeline) {
        return rd_utils::build_compute_stage(rd, godot::String(kShaderDir) + p_file, p_name, r_shader, r_pipeline);
    };
    const bool ok =
        build("lumina_rc_trace.comp.glsl", "lumina_rc_trace", trace_shader, trace_pipeline) &&
        build("lumina_rc_merge.comp.glsl", "lumina_rc_merge", merge_shader, merge_pipeline) &&
        build("lumina_rc_integrate.comp.glsl", "lumina_rc_integrate", integrate_shader, integrate_pipeline) &&
        build("lumina_svgf_temporal.comp.glsl", "lumina_svgf_temporal", temporal_shader, temporal_pipeline) &&
        build("lumina_svgf_atrous.comp.glsl", "lumina_svgf_atrous", atrous_shader, atrous_pipeline) &&
        build("lumina_composite.comp.glsl", "lumina_composite", composite_shader, composite_pipeline);
    if (!ok) {
        _release_render_resources();
    }
    return ok;
}

void LuminaCascadesCompositorEffect::_release_render_resources() {
    if (rd == nullptr) {
        return;
    }
    rd_utils::free_compute_stage(rd, trace_pipeline, trace_shader);
    rd_utils::free_compute_stage(rd, merge_pipeline, merge_shader);
    rd_utils::free_compute_stage(rd, integrate_pipeline, integrate_shader);
    rd_utils::free_compute_stage(rd, temporal_pipeline, temporal_shader);
    rd_utils::free_compute_stage(rd, atrous_pipeline, atrous_shader);
    rd_utils::free_compute_stage(rd, composite_pipeline, composite_shader);
    rd_utils::free_rid(rd, nearest_sampler);
    rd_utils::free_rid(rd, linear_sampler);
}

bool LuminaCascadesCompositorEffect::_ensure_view_targets(const godot::Vector2i &p_size, uint32_t p_view_count) {
    if (cached_internal_size == p_size && view_targets.size() == p_view_count &&
        cached_cascade_count == cascade_count && cached_probe_spacing == probe_spacing) {
        return true;
    }
    _release_view_targets();

    using RD = godot::RenderingDevice;
    const uint32_t rw = RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT;
    const uint32_t w = static_cast<uint32_t>(p_size.x);
    const uint32_t h = static_cast<uint32_t>(p_size.y);

    view_targets.resize(p_view_count);
    for (PerViewTargets &vt : view_targets) {
        bool ok = true;
        for (int32_t level = 0; level < cascade_count; ++level) {
            CascadeLevel &cl = vt.cascades[static_cast<size_t>(level)];
            const int32_t spacing = probe_spacing << level;
            cl.spacing_px = static_cast<float>(spacing);
            cl.block = kBaseBlock << level;
            cl.probe_count = godot::Vector2i((p_size.x + spacing - 1) / spacing, (p_size.y + spacing - 1) / spacing);
            cl.texture = rd_utils::create_2d_texture(rd,
                static_cast<uint32_t>(cl.probe_count.x * cl.block), static_cast<uint32_t>(cl.probe_count.y * cl.block),
                RD::DATA_FORMAT_R16G16B16A16_SFLOAT, rw);
            ok = ok && cl.texture.is_valid();
        }
        vt.irradiance_raw = rd_utils::create_2d_texture(rd, w, h, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, rw);
        ok = ok && vt.irradiance_raw.is_valid();
        for (size_t i = 0; i < 2; ++i) {
            vt.history[i] = rd_utils::create_2d_texture(rd, w, h, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, rw);
            vt.moments[i] = rd_utils::create_2d_texture(rd, w, h, RD::DATA_FORMAT_R16G16_SFLOAT, rw);
            vt.geometry[i] = rd_utils::create_2d_texture(rd, w, h, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, rw);
            vt.filter[i] = rd_utils::create_2d_texture(rd, w, h, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, rw);
            ok = ok && vt.history[i].is_valid() && vt.moments[i].is_valid() && vt.geometry[i].is_valid() && vt.filter[i].is_valid();
        }
        vt.history_index = 0;
        vt.has_history = false;
        if (!ok) {
            godot::UtilityFunctions::push_error("[Prism RP] Failed to allocate Lumina-Cascades render targets.");
            _release_view_targets();
            return false;
        }
    }

    cached_internal_size = p_size;
    cached_cascade_count = cascade_count;
    cached_probe_spacing = probe_spacing;
    history_reset_requested = true;
    return true;
}

void LuminaCascadesCompositorEffect::_release_view_targets() {
    for (PerViewTargets &vt : view_targets) {
        for (CascadeLevel &cl : vt.cascades) {
            rd_utils::free_rid(rd, cl.texture);
        }
        rd_utils::free_rid(rd, vt.irradiance_raw);
        for (size_t i = 0; i < 2; ++i) {
            rd_utils::free_rid(rd, vt.history[i]);
            rd_utils::free_rid(rd, vt.moments[i]);
            rd_utils::free_rid(rd, vt.geometry[i]);
            rd_utils::free_rid(rd, vt.filter[i]);
        }
    }
    view_targets.clear();
    cached_internal_size = godot::Vector2i(0, 0);
    cached_cascade_count = 0;
    cached_probe_spacing = 0;
}

} // namespace prism
