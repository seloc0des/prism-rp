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

    static PrismTelemetry *get_singleton();

    void configure_from_cmdline();
    void begin_frame(uint64_t p_frame_index);

    void begin_pass(
        const godot::String &p_pass_name,
        const godot::Color &p_debug_color = godot::Color(0.25f, 0.72f, 0.95f, 1.0f)
    );
    void end_pass();

    void collect_and_maybe_export();

    void set_enabled(bool p_enabled);
    [[nodiscard]] bool is_enabled() const;

    void set_qa_capture_frame(int64_t p_frame);
    [[nodiscard]] int64_t get_qa_capture_frame() const;

    void set_telemetry_output_path(const godot::String &p_path);
    [[nodiscard]] godot::String get_telemetry_output_path() const;

    [[nodiscard]] double get_pass_gpu_ms(const godot::String &p_pass_name) const;
    [[nodiscard]] godot::Dictionary get_telemetry_dictionary() const;
    bool export_json(const godot::String &p_output_path) const;

protected:
    static void _bind_methods();

private:
    static godot::String _map_to_canonical_stage(const godot::String &p_subpass_name);
    void _ensure_rendering_device();

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
    int64_t qa_capture_frame = -1;
    godot::String qa_telemetry_out_path;
    bool qa_exported = false;
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
