#include "prism_rp/prism_telemetry.h"

#include <algorithm>
#include <utility>

#include <godot_cpp/classes/dir_access.hpp>
#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/json.hpp>
#include <godot_cpp/classes/os.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

namespace prism {

namespace {

constexpr const char *kBeginPrefix = "prism_begin:";
constexpr const char *kEndPrefix = "prism_end:";
constexpr double kNanosToMillis = 1.0 / 1'000'000.0;
constexpr double kEmaAlpha = 0.20;

} // namespace

PrismTelemetry *PrismTelemetry::singleton = nullptr;

PrismTelemetry::PrismTelemetry() {
    singleton = this;
    _ensure_rendering_device();
    configure_from_cmdline();
}

PrismTelemetry::~PrismTelemetry() {
    if (!qa_exported && !qa_telemetry_out_path.is_empty() && !canonical_stage_stats.empty()) {
        export_json(qa_telemetry_out_path);
        qa_exported = true;
    }

    if (singleton == this) {
        singleton = nullptr;
    }
}

PrismTelemetry *PrismTelemetry::get_singleton() {
    return singleton;
}

void PrismTelemetry::_bind_methods() {
    godot::ClassDB::bind_method(godot::D_METHOD("set_enabled", "enabled"), &PrismTelemetry::set_enabled);
    godot::ClassDB::bind_method(godot::D_METHOD("is_enabled"), &PrismTelemetry::is_enabled);
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::BOOL, "enabled"), "set_enabled", "is_enabled");

    godot::ClassDB::bind_method(godot::D_METHOD("set_qa_capture_frame", "frame"), &PrismTelemetry::set_qa_capture_frame);
    godot::ClassDB::bind_method(godot::D_METHOD("get_qa_capture_frame"), &PrismTelemetry::get_qa_capture_frame);

    godot::ClassDB::bind_method(godot::D_METHOD("set_telemetry_output_path", "path"), &PrismTelemetry::set_telemetry_output_path);
    godot::ClassDB::bind_method(godot::D_METHOD("get_telemetry_output_path"), &PrismTelemetry::get_telemetry_output_path);

    godot::ClassDB::bind_method(godot::D_METHOD("begin_pass", "pass_name", "debug_color"), &PrismTelemetry::begin_pass, DEFVAL(godot::Color(0.25f, 0.72f, 0.95f, 1.0f)));
    godot::ClassDB::bind_method(godot::D_METHOD("end_pass"), &PrismTelemetry::end_pass);
    godot::ClassDB::bind_method(godot::D_METHOD("get_pass_gpu_ms", "pass_name"), &PrismTelemetry::get_pass_gpu_ms);
    godot::ClassDB::bind_method(godot::D_METHOD("get_telemetry_dictionary"), &PrismTelemetry::get_telemetry_dictionary);
    godot::ClassDB::bind_method(godot::D_METHOD("export_json", "output_path"), &PrismTelemetry::export_json);
}

void PrismTelemetry::_ensure_rendering_device() {
    if (rd == nullptr && godot::RenderingServer::get_singleton() != nullptr) {
        rd = godot::RenderingServer::get_singleton()->get_rendering_device();
    }
}

void PrismTelemetry::configure_from_cmdline() {
    if (godot::OS::get_singleton() == nullptr) {
        return;
    }

    const godot::PackedStringArray user_args = godot::OS::get_singleton()->get_cmdline_user_args();
    for (int64_t i = 0; i < user_args.size(); ++i) {
        const godot::String arg = user_args[i];
        if (arg.begins_with("--prism-tier=")) {
            active_tier = arg.get_slice("=", 1);
        } else if (arg.begins_with("--qa-capture-frame=")) {
            qa_capture_frame = arg.get_slice("=", 1).to_int();
        } else if (arg.begins_with("--qa-telemetry-out=")) {
            qa_telemetry_out_path = arg.get_slice("=", 1);
        }
    }
}

void PrismTelemetry::set_enabled(bool p_enabled) {
    enabled = p_enabled;
}

bool PrismTelemetry::is_enabled() const {
    return enabled;
}

void PrismTelemetry::set_qa_capture_frame(int64_t p_frame) {
    qa_capture_frame = p_frame;
    qa_exported = false;
}

int64_t PrismTelemetry::get_qa_capture_frame() const {
    return qa_capture_frame;
}

void PrismTelemetry::set_telemetry_output_path(const godot::String &p_path) {
    qa_telemetry_out_path = p_path;
    qa_exported = false;
}

godot::String PrismTelemetry::get_telemetry_output_path() const {
    return qa_telemetry_out_path;
}

void PrismTelemetry::begin_frame(uint64_t p_frame_index) {
    current_cpu_frame = p_frame_index;
    open_pass_depth = 0;
    collect_and_maybe_export();
}

void PrismTelemetry::begin_pass(const godot::String &p_pass_name, const godot::Color &p_debug_color) {
    if (!enabled) {
        return;
    }
    _ensure_rendering_device();
    if (rd == nullptr) {
        return;
    }

    rd->draw_command_begin_label(p_pass_name, p_debug_color);
    rd->capture_timestamp(godot::String(kBeginPrefix) + p_pass_name);
    ++open_pass_depth;
}

void PrismTelemetry::end_pass() {
    if (!enabled || open_pass_depth == 0) {
        return;
    }
    _ensure_rendering_device();
    if (rd == nullptr) {
        return;
    }

    rd->capture_timestamp(godot::String(kEndPrefix));
    rd->draw_command_end_label();
    --open_pass_depth;
}

godot::String PrismTelemetry::_map_to_canonical_stage(const godot::String &p_subpass_name) {
    if (p_subpass_name == "chronos_aces2_odt" || p_subpass_name.begins_with("aces2_")) {
        return "aces2_optics";
    }
    if (p_subpass_name.begins_with("chronos_")) {
        return "chronos_vpu";
    }
    if (p_subpass_name.begins_with("lumina_") || p_subpass_name.begins_with("svgf_")) {
        return "lumina_cascades_gi";
    }
    if (p_subpass_name.begins_with("meshlet_") || p_subpass_name.begins_with("visbuf_")) {
        return "gpu_meshlet_cull";
    }
    if (p_subpass_name.begins_with("tile_shade_") || p_subpass_name.begins_with("material_")) {
        return "tile_material_shade";
    }
    if (p_subpass_name.begins_with("vcsp_") || p_subpass_name.begins_with("shadow_")) {
        return "virtual_shadow_pages";
    }
    if (p_subpass_name.begins_with("froxel_") || p_subpass_name.begins_with("wboit_")) {
        return "froxel_volumetrics";
    }
    return p_subpass_name;
}

void PrismTelemetry::collect_and_maybe_export() {
    if (!enabled) {
        return;
    }
    _ensure_rendering_device();
    if (rd == nullptr) {
        return;
    }

    const uint32_t ts_count = rd->get_captured_timestamps_count();
    if (ts_count == 0) {
        return;
    }

    const uint64_t retired_gpu_frame = rd->get_captured_timestamps_frame();
    if (retired_gpu_frame == last_collected_gpu_frame) {
        return;
    }
    last_collected_gpu_frame = retired_gpu_frame;

    struct OpenStamp {
        std::string name;
        uint64_t gpu_ns = 0;
        uint64_t cpu_ns = 0;
    };

    std::vector<OpenStamp> stamp_stack;
    stamp_stack.reserve(16);

    std::unordered_map<std::string, double> frame_subpass_gpu_ms;
    std::unordered_map<std::string, double> frame_subpass_cpu_ms;
    std::unordered_map<std::string, double> frame_stage_gpu_ms;
    std::unordered_map<std::string, double> frame_stage_cpu_ms;

    for (uint32_t i = 0; i < ts_count; ++i) {
        const godot::String label = rd->get_captured_timestamp_name(i);
        const uint64_t gpu_ns = rd->get_captured_timestamp_gpu_time(i);
        const uint64_t cpu_ns = rd->get_captured_timestamp_cpu_time(i);

        if (label.begins_with(kBeginPrefix)) {
            const godot::String pass_name = label.substr(12);
            const std::string key = pass_name.utf8().get_data();
            if (!subpass_stats.contains(key)) {
                insertion_order.push_back(key);
            }
            stamp_stack.push_back(OpenStamp{ key, gpu_ns, cpu_ns });
        } else if (label.begins_with(kEndPrefix) && !stamp_stack.empty()) {
            const OpenStamp start = std::move(stamp_stack.back());
            stamp_stack.pop_back();

            const double delta_gpu_ms = (gpu_ns >= start.gpu_ns)
                ? static_cast<double>(gpu_ns - start.gpu_ns) * kNanosToMillis
                : 0.0;
            const double delta_cpu_ms = (cpu_ns >= start.cpu_ns)
                ? static_cast<double>(cpu_ns - start.cpu_ns) * kNanosToMillis
                : 0.0;

            frame_subpass_gpu_ms[start.name] += delta_gpu_ms;
            frame_subpass_cpu_ms[start.name] += delta_cpu_ms;

            const godot::String canonical_gd = _map_to_canonical_stage(godot::String::utf8(start.name.c_str()));
            const std::string canonical_key = canonical_gd.utf8().get_data();
            frame_stage_gpu_ms[canonical_key] += delta_gpu_ms;
            frame_stage_cpu_ms[canonical_key] += delta_cpu_ms;
        }
    }

    auto update_sample_map = [](
        std::unordered_map<std::string, PassSample> &r_target_map,
        const std::unordered_map<std::string, double> &p_frame_gpu,
        const std::unordered_map<std::string, double> &p_frame_cpu
    ) {
        for (const auto &[key, gpu_ms] : p_frame_gpu) {
            PassSample &sample = r_target_map[key];
            const double cpu_ms = p_frame_cpu.contains(key) ? p_frame_cpu.at(key) : 0.0;
            sample.latest_gpu_ms = gpu_ms;
            sample.latest_cpu_ms = cpu_ms;
            if (sample.sample_count == 0) {
                sample.ema_gpu_ms = gpu_ms;
            } else {
                sample.ema_gpu_ms = (kEmaAlpha * gpu_ms) + ((1.0 - kEmaAlpha) * sample.ema_gpu_ms);
            }
            ++sample.sample_count;
        }
    };

    update_sample_map(subpass_stats, frame_subpass_gpu_ms, frame_subpass_cpu_ms);
    update_sample_map(canonical_stage_stats, frame_stage_gpu_ms, frame_stage_cpu_ms);

    if (!qa_exported &&
        qa_capture_frame >= 0 &&
        !qa_telemetry_out_path.is_empty() &&
        retired_gpu_frame >= static_cast<uint64_t>(qa_capture_frame) &&
        !canonical_stage_stats.empty()) {
        if (export_json(qa_telemetry_out_path)) {
            qa_exported = true;
            godot::UtilityFunctions::print(
                "[Prism RP] Exported GPU telemetry for frame ",
                static_cast<int64_t>(retired_gpu_frame),
                " -> ",
                qa_telemetry_out_path
            );
        }
    }
}

double PrismTelemetry::get_pass_gpu_ms(const godot::String &p_pass_name) const {
    const std::string key = p_pass_name.utf8().get_data();
    if (const auto it = canonical_stage_stats.find(key); it != canonical_stage_stats.end()) {
        return it->second.latest_gpu_ms;
    }
    if (const auto it = subpass_stats.find(key); it != subpass_stats.end()) {
        return it->second.latest_gpu_ms;
    }
    return 0.0;
}

godot::Dictionary PrismTelemetry::get_telemetry_dictionary() const {
    godot::Dictionary root;
    root["schema_version"] = 1;
    root["tier"] = active_tier;
    root["captured_gpu_frame"] = static_cast<int64_t>(last_collected_gpu_frame == UINT64_MAX ? 0 : last_collected_gpu_frame);
    root["submitted_cpu_frame"] = static_cast<int64_t>(current_cpu_frame);

    if (rd != nullptr) {
        godot::Dictionary device_info;
        device_info["name"] = rd->get_device_name();
        device_info["vendor"] = rd->get_device_vendor_name();
        root["device"] = device_info;

        constexpr double kBytesToMegabytes = 1.0 / (1024.0 * 1024.0);
        godot::Dictionary vram_mb;
        vram_mb["textures"] = static_cast<double>(rd->get_memory_usage(godot::RenderingDevice::MEMORY_TEXTURES)) * kBytesToMegabytes;
        vram_mb["buffers"] = static_cast<double>(rd->get_memory_usage(godot::RenderingDevice::MEMORY_BUFFERS)) * kBytesToMegabytes;
        vram_mb["total"] = static_cast<double>(rd->get_memory_usage(godot::RenderingDevice::MEMORY_TOTAL)) * kBytesToMegabytes;
        root["vram_usage_mb"] = vram_mb;
    }

    godot::Dictionary pass_timings_ms;
    double total_gpu_ms = 0.0;
    for (const auto &[stage_name, sample] : canonical_stage_stats) {
        pass_timings_ms[godot::String::utf8(stage_name.c_str())] = sample.latest_gpu_ms;
        total_gpu_ms += sample.latest_gpu_ms;
    }
    root["pass_timings_ms"] = pass_timings_ms;
    root["total_prism_gpu_ms"] = total_gpu_ms;

    godot::Dictionary subpass_timings;
    for (const std::string &sub_name : insertion_order) {
        const auto it = subpass_stats.find(sub_name);
        if (it == subpass_stats.end()) {
            continue;
        }
        godot::Dictionary entry;
        entry["gpu_ms"] = it->second.latest_gpu_ms;
        entry["ema_gpu_ms"] = it->second.ema_gpu_ms;
        entry["cpu_encode_ms"] = it->second.latest_cpu_ms;
        entry["samples"] = static_cast<int64_t>(it->second.sample_count);
        subpass_timings[godot::String::utf8(sub_name.c_str())] = entry;
    }
    root["subpass_timings"] = subpass_timings;

    return root;
}

bool PrismTelemetry::export_json(const godot::String &p_output_path) const {
    if (p_output_path.is_empty()) {
        return false;
    }

    const godot::String parent_dir = p_output_path.get_base_dir();
    if (!parent_dir.is_empty()) {
        godot::DirAccess::make_dir_recursive_absolute(parent_dir);
    }

    const godot::String json_text = godot::JSON::stringify(get_telemetry_dictionary(), "\t", true);
    godot::Ref<godot::FileAccess> file = godot::FileAccess::open(p_output_path, godot::FileAccess::WRITE);
    if (file.is_null()) {
        godot::UtilityFunctions::push_error("[Prism RP] Failed to open telemetry output path for writing: ", p_output_path);
        return false;
    }

    file->store_string(json_text + "\n");
    file->flush();
    return true;
}

} // namespace prism
