#pragma once

#include <cstdint>
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
    void begin_frame(uint64_t p_frame_index);

    void begin_pass(
        const godot::String &p_pass_name,
        const godot::Color &p_debug_color = godot::Color(0.25f, 0.72f, 0.95f, 1.0f)
    );
    void end_pass();

    void collect_timestamps();

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
    void *_get_renderdoc_api();

    static PrismTelemetry *singleton;

    godot::RenderingDevice *rd = nullptr;
    bool enabled = true;
    uint32_t open_pass_depth = 0;

    uint64_t current_cpu_frame = 0;
    uint64_t last_collected_gpu_frame = UINT64_MAX;

    std::unordered_map<std::string, PassSample> subpass_stats;
    std::unordered_map<std::string, PassSample> canonical_stage_stats;
    std::vector<std::string> insertion_order;

    godot::String active_tier = "tier2";

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
