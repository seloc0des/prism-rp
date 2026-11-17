#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <godot_cpp/classes/object.hpp>
#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/string.hpp>

namespace prism {

class PrismTelemetry : public godot::Object {
    GDCLASS(PrismTelemetry, godot::Object)

public:
    struct PassSample {
        double latest_gpu_ms = 0.0;
        double ema_gpu_ms = 0.0;
        double latest_cpu_ms = 0.0;
        uint64_t sample_count = 0;
    };

    PrismTelemetry();
    ~PrismTelemetry() override;

    // The singleton is owned by register_types.cpp. The class is registered as abstract so
    // scripts and the editor can't create extra instances that would replace it.
    static PrismTelemetry *get_singleton();
    static void set_singleton(PrismTelemetry *p_instance);

    void configure_from_cmdline();

    // Once-per-frame collection. connect_frame_hook() wires RenderingServer.frame_pre_draw
    // (emitted on the main thread) to a call_on_render_thread() hop, so collection runs on the
    // render thread exactly once per frame, before that frame's draw, however many viewports
    // or Chronos instances record passes.
    void connect_frame_hook();
    void disconnect_frame_hook();

    void begin_pass(
        const godot::String &p_pass_name,
        const godot::Color &p_debug_color = godot::Color(0.25f, 0.72f, 0.95f, 1.0f)
    );
    void end_pass();


    void set_enabled(bool p_enabled);
    [[nodiscard]] bool is_enabled() const;

    [[nodiscard]] godot::String get_active_tier() const;

    // RenderDoc in-app API: only works when the process was launched under RenderDoc.
    [[nodiscard]] bool is_renderdoc_attached();
    bool trigger_renderdoc_capture();

    [[nodiscard]] double get_pass_gpu_ms(const godot::String &p_pass_name) const;
    [[nodiscard]] godot::Dictionary get_telemetry_dictionary() const;
    bool export_json(const godot::String &p_output_path) const;

protected:
    static void _bind_methods();

private:
    static godot::String _map_to_canonical_stage(const godot::String &p_subpass_name);
    void _ensure_rendering_device();
    void _on_frame_pre_draw();
    void _collect_on_render_thread();
    void *_get_renderdoc_api();

    static PrismTelemetry *singleton;

    godot::RenderingDevice *rd = nullptr;
#if defined(DEBUG_ENABLED)
    bool enabled = true;     // debug/editor builds: on by default
#else
    bool enabled = false;    // template_release: no GPU timestamps unless asked (--prism-telemetry=on)
#endif
    bool frame_hook_connected = false;
    uint32_t open_pass_depth = 0;   // render thread only

    // Guards everything below: written on the render thread, read from scripts on the main thread.
    mutable std::mutex stats_mutex;

    uint64_t current_cpu_frame = 0;
    uint64_t last_collected_gpu_frame = UINT64_MAX;

    std::unordered_map<std::string, PassSample> subpass_stats;
    std::unordered_map<std::string, PassSample> canonical_stage_stats;
    std::vector<std::string> insertion_order;

    godot::String active_tier = "tier2";

    // Cached on the render thread so main-thread readers never touch the RenderingDevice.
    bool has_device_info = false;
    godot::String device_name;
    godot::String device_vendor;
    double vram_textures_mb = 0.0;
    double vram_buffers_mb = 0.0;
    double vram_total_mb = 0.0;

    void *renderdoc_api = nullptr;
    bool renderdoc_probed = false;
};

class ScopedPassTimer {
public:
    explicit ScopedPassTimer(
        const godot::String &p_pass_name,
        const godot::Color &p_debug_color = godot::Color(0.25f, 0.72f, 0.95f, 1.0f)
    ) {
        if (PrismTelemetry *telemetry = PrismTelemetry::get_singleton(); telemetry != nullptr) {
            telemetry->begin_pass(p_pass_name, p_debug_color);
            active = true;
        }
    }

    ~ScopedPassTimer() {
        if (active) {
            if (PrismTelemetry *telemetry = PrismTelemetry::get_singleton(); telemetry != nullptr) {
                telemetry->end_pass();
            }
        }
    }

    ScopedPassTimer(const ScopedPassTimer &) = delete;
    ScopedPassTimer &operator=(const ScopedPassTimer &) = delete;

private:
    bool active = false;
};

} // namespace prism
