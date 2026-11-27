#include "prism_rp/prism_telemetry.h"

#include <algorithm>
#include <cstring>
#include <utility>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#elif defined(__linux__)
#include <dlfcn.h>
#endif

#include <renderdoc_app.h>

#include <godot_cpp/classes/dir_access.hpp>
#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/json.hpp>
#include <godot_cpp/classes/os.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/callable_method_pointer.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

namespace prism {

namespace {

constexpr const char *kBeginPrefix = "prism_begin:";
constexpr const char *kEndPrefix = "prism_end:";
const int64_t kBeginPrefixLength = static_cast<int64_t>(std::strlen(kBeginPrefix));
constexpr double kNanosToMillis = 1.0 / 1'000'000.0;
constexpr double kEmaAlpha = 0.20;

} // namespace

PrismTelemetry *PrismTelemetry::singleton = nullptr;

double PrismTelemetry::PassSample::median_gpu_ms() const {
    const size_t count = static_cast<size_t>(std::min<uint64_t>(sample_count, kMedianWindow));
    if (count == 0) {
        return 0.0;
    }
    std::array<double, kMedianWindow> sorted = recent_gpu_ms;
    std::sort(sorted.begin(), sorted.begin() + count);
    const size_t mid = count / 2;
    return (count % 2 == 1) ? sorted[mid] : 0.5 * (sorted[mid - 1] + sorted[mid]);
}

// The RenderingServer is not registered yet when extensions initialize, so nothing here (or in
// connect_frame_hook, which register_types defers to the first main-loop iteration) touches it.
PrismTelemetry::PrismTelemetry() {
    configure_from_cmdline();
}

namespace {
godot::RenderingServer *rendering_server_if_ready() {
    godot::Engine *engine = godot::Engine::get_singleton();
    if (engine == nullptr || !engine->has_singleton("RenderingServer")) {
        return nullptr;
    }
    return godot::RenderingServer::get_singleton();
}
} // namespace

// No export or RenderingDevice access here: at extension shutdown the RenderingServer may
// already be gone. The QA rig exports explicitly while the engine is still running.
PrismTelemetry::~PrismTelemetry() {
    if (singleton == this) {
        singleton = nullptr;
    }
    rd = nullptr;
}

PrismTelemetry *PrismTelemetry::get_singleton() {
    return singleton;
}

void PrismTelemetry::set_singleton(PrismTelemetry *p_instance) {
    singleton = p_instance;
}

void PrismTelemetry::_bind_methods() {
    godot::ClassDB::bind_method(godot::D_METHOD("set_enabled", "enabled"), &PrismTelemetry::set_enabled);
    godot::ClassDB::bind_method(godot::D_METHOD("is_enabled"), &PrismTelemetry::is_enabled);
    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::BOOL, "enabled"), "set_enabled", "is_enabled");

    godot::ClassDB::bind_method(godot::D_METHOD("get_active_tier"), &PrismTelemetry::get_active_tier);
    godot::ClassDB::bind_method(godot::D_METHOD("is_renderdoc_attached"), &PrismTelemetry::is_renderdoc_attached);
    godot::ClassDB::bind_method(godot::D_METHOD("trigger_renderdoc_capture"), &PrismTelemetry::trigger_renderdoc_capture);

    godot::ClassDB::bind_method(godot::D_METHOD("begin_pass", "pass_name", "debug_color"), &PrismTelemetry::begin_pass, DEFVAL(godot::Color(0.25f, 0.72f, 0.95f, 1.0f)));
    godot::ClassDB::bind_method(godot::D_METHOD("end_pass"), &PrismTelemetry::end_pass);
    godot::ClassDB::bind_method(godot::D_METHOD("get_pass_gpu_ms", "pass_name"), &PrismTelemetry::get_pass_gpu_ms);
    godot::ClassDB::bind_method(godot::D_METHOD("get_telemetry_dictionary"), &PrismTelemetry::get_telemetry_dictionary);
    godot::ClassDB::bind_method(godot::D_METHOD("export_json", "output_path"), &PrismTelemetry::export_json);
}

void PrismTelemetry::_ensure_rendering_device() {
    if (rd == nullptr) {
        if (godot::RenderingServer *rs = rendering_server_if_ready(); rs != nullptr) {
            rd = rs->get_rendering_device();
        }
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
        } else if (arg.begins_with("--prism-telemetry=")) {
            const godot::String value = arg.get_slice("=", 1).to_lower();
            enabled = (value == "on" || value == "1" || value == "true");
        }
    }
}

void PrismTelemetry::set_enabled(bool p_enabled) {
    enabled = p_enabled;
}

bool PrismTelemetry::is_enabled() const {
    return enabled;
}

godot::String PrismTelemetry::get_active_tier() const {
    return active_tier;
}

void *PrismTelemetry::_get_renderdoc_api() {
    if (renderdoc_probed) {
        return renderdoc_api;
    }
    renderdoc_probed = true;

    pRENDERDOC_GetAPI get_api = nullptr;
#if defined(_WIN32)
    if (HMODULE mod = GetModuleHandleA("renderdoc.dll"); mod != nullptr) {
        get_api = reinterpret_cast<pRENDERDOC_GetAPI>(GetProcAddress(mod, "RENDERDOC_GetAPI"));
    }
#elif defined(__linux__)
    // RTLD_NOLOAD: only succeed if RenderDoc already injected itself; never load it ourselves.
    if (void *mod = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD); mod != nullptr) {
        get_api = reinterpret_cast<pRENDERDOC_GetAPI>(dlsym(mod, "RENDERDOC_GetAPI"));
    }
#endif
    if (get_api != nullptr) {
        void *api = nullptr;
        if (get_api(eRENDERDOC_API_Version_1_1_2, &api) == 1) {
            renderdoc_api = api;
        }
    }
    return renderdoc_api;
}

bool PrismTelemetry::is_renderdoc_attached() {
    return _get_renderdoc_api() != nullptr;
}

bool PrismTelemetry::trigger_renderdoc_capture() {
    auto *api = static_cast<RENDERDOC_API_1_1_2 *>(_get_renderdoc_api());
    if (api == nullptr) {
        return false;
    }
    api->TriggerCapture();
    return true;
}

void PrismTelemetry::connect_frame_hook() {
    godot::RenderingServer *rs = rendering_server_if_ready();
    if (rs == nullptr || frame_hook_connected) {
        return;
    }
    rs->connect("frame_pre_draw", callable_mp(this, &PrismTelemetry::_on_frame_pre_draw));
    frame_hook_connected = true;
}

void PrismTelemetry::disconnect_frame_hook() {
    godot::RenderingServer *rs = rendering_server_if_ready();
    if (rs == nullptr || !frame_hook_connected) {
        return;
    }
    const godot::Callable cb = callable_mp(this, &PrismTelemetry::_on_frame_pre_draw);
    if (rs->is_connected("frame_pre_draw", cb)) {
        rs->disconnect("frame_pre_draw", cb);
    }
    frame_hook_connected = false;
}

// Main thread. Queued onto the render thread ahead of this frame's draw.
void PrismTelemetry::_on_frame_pre_draw() {
    if (!enabled) {
        return;
    }
    if (godot::RenderingServer *rs = rendering_server_if_ready(); rs != nullptr) {
        rs->call_on_render_thread(callable_mp(this, &PrismTelemetry::_collect_on_render_thread));
    }
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

void PrismTelemetry::_collect_on_render_thread() {
    open_pass_depth = 0;
    _ensure_rendering_device();
    if (rd == nullptr) {
        return;
    }

    {
        constexpr double kBytesToMegabytes = 1.0 / (1024.0 * 1024.0);
        std::lock_guard<std::mutex> lock(stats_mutex);
        current_cpu_frame = godot::Engine::get_singleton()->get_frames_drawn();
        if (!has_device_info) {
            device_name = rd->get_device_name();
            device_vendor = rd->get_device_vendor_name();
            has_device_info = true;
        }
        vram_textures_mb = static_cast<double>(rd->get_memory_usage(godot::RenderingDevice::MEMORY_TEXTURES)) * kBytesToMegabytes;
        vram_buffers_mb = static_cast<double>(rd->get_memory_usage(godot::RenderingDevice::MEMORY_BUFFERS)) * kBytesToMegabytes;
        vram_total_mb = static_cast<double>(rd->get_memory_usage(godot::RenderingDevice::MEMORY_TOTAL)) * kBytesToMegabytes;
    }

    const uint32_t ts_count = rd->get_captured_timestamps_count();
    if (ts_count == 0) {
        return;
    }

    const uint64_t retired_gpu_frame = rd->get_captured_timestamps_frame();

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
            const godot::String pass_name = label.substr(kBeginPrefixLength);
            const std::string key = pass_name.utf8().get_data();
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
        const std::unordered_map<std::string, double> &p_frame_cpu,
        uint64_t p_gpu_frame
    ) {
        for (const auto &[key, gpu_ms] : p_frame_gpu) {
            PassSample &sample = r_target_map[key];
            const double cpu_ms = p_frame_cpu.contains(key) ? p_frame_cpu.at(key) : 0.0;
            sample.latest_gpu_ms = gpu_ms;
            sample.latest_cpu_ms = cpu_ms;
            sample.last_gpu_frame = p_gpu_frame;
            if (sample.sample_count == 0) {
                sample.ema_gpu_ms = gpu_ms;
            } else {
                sample.ema_gpu_ms = (kEmaAlpha * gpu_ms) + ((1.0 - kEmaAlpha) * sample.ema_gpu_ms);
            }
            sample.recent_gpu_ms[sample.sample_count % kMedianWindow] = gpu_ms;
            ++sample.sample_count;
        }
    };

    std::lock_guard<std::mutex> lock(stats_mutex);
    if (retired_gpu_frame == last_collected_gpu_frame) {
        return;
    }
    last_collected_gpu_frame = retired_gpu_frame;
    for (const auto &[key, _ms] : frame_subpass_gpu_ms) {
        if (!subpass_stats.contains(key)) {
            insertion_order.push_back(key);
        }
    }
    update_sample_map(subpass_stats, frame_subpass_gpu_ms, frame_subpass_cpu_ms, retired_gpu_frame);
    update_sample_map(canonical_stage_stats, frame_stage_gpu_ms, frame_stage_cpu_ms, retired_gpu_frame);
}

double PrismTelemetry::get_pass_gpu_ms(const godot::String &p_pass_name) const {
    const std::string key = p_pass_name.utf8().get_data();
    std::lock_guard<std::mutex> lock(stats_mutex);
    if (const auto it = canonical_stage_stats.find(key); it != canonical_stage_stats.end()) {
        return it->second.latest_gpu_ms;
    }
    if (const auto it = subpass_stats.find(key); it != subpass_stats.end()) {
        return it->second.latest_gpu_ms;
    }
    return 0.0;
}

godot::Dictionary PrismTelemetry::get_telemetry_dictionary() const {
    std::lock_guard<std::mutex> lock(stats_mutex);
    godot::Dictionary root;
    root["schema_version"] = 1;
    root["tier"] = active_tier;
    root["captured_gpu_frame"] = static_cast<int64_t>(last_collected_gpu_frame == UINT64_MAX ? 0 : last_collected_gpu_frame);
    root["submitted_cpu_frame"] = static_cast<int64_t>(current_cpu_frame);

    root["telemetry_enabled"] = enabled;
    if (has_device_info) {
        godot::Dictionary device_info;
        device_info["name"] = device_name;
        device_info["vendor"] = device_vendor;
        root["device"] = device_info;

        godot::Dictionary vram_mb;
        vram_mb["textures"] = vram_textures_mb;
        vram_mb["buffers"] = vram_buffers_mb;
        vram_mb["total"] = vram_total_mb;
        root["vram_usage_mb"] = vram_mb;
    }

    // A stage's median is the sum of its passes' medians, not the median of its per-frame totals. In most
    // frames another GPU client takes the GPU for about 0.45 ms in the middle of whichever pass is running
    // (on the Tier 2 machine a 0.03 ms pass came out at 0.5 ms), so nearly every frame's total holds one
    // such gap while each pass's own median leaves it out. Passes not timed within the window are gone.
    std::unordered_map<std::string, double> stage_median_ms;
    for (const auto &[sub_name, sample] : subpass_stats) {
        if (last_collected_gpu_frame - sample.last_gpu_frame < kMedianWindow) {
            const godot::String stage = _map_to_canonical_stage(godot::String::utf8(sub_name.c_str()));
            stage_median_ms[stage.utf8().get_data()] += sample.median_gpu_ms();
        }
    }

    godot::Dictionary pass_timings_ms;
    godot::Dictionary pass_timings_median_ms;
    double total_gpu_ms = 0.0;
    for (const auto &[stage_name, sample] : canonical_stage_stats) {
        const godot::String name = godot::String::utf8(stage_name.c_str());
        pass_timings_ms[name] = sample.latest_gpu_ms;
        pass_timings_median_ms[name] = stage_median_ms[stage_name];
        total_gpu_ms += sample.latest_gpu_ms;
    }
    root["pass_timings_ms"] = pass_timings_ms;
    root["pass_timings_median_ms"] = pass_timings_median_ms;
    root["median_window_frames"] = static_cast<int64_t>(kMedianWindow);
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
        entry["median_gpu_ms"] = it->second.median_gpu_ms();
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

    file->store_string(json_text + godot::String("\n"));
    file->flush();
    return true;
}

} // namespace prism
