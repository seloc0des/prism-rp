#include "prism_rp/chronos_opaque_capture_effect.h"

#include <mutex>
#include <unordered_set>

#include <godot_cpp/classes/render_scene_buffers.hpp>
#include <godot_cpp/classes/render_scene_buffers_rd.hpp>
#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/classes/rendering_server.hpp>

namespace prism {

namespace {
std::mutex g_capture_mutex;
std::unordered_set<uint64_t> g_fresh_captures;
} // namespace

ChronosOpaqueCaptureEffect::ChronosOpaqueCaptureEffect() {
    set_effect_callback_type(EFFECT_CALLBACK_TYPE_PRE_TRANSPARENT);
    set_access_resolved_color(true);
}

ChronosOpaqueCaptureEffect::~ChronosOpaqueCaptureEffect() {
    copy.release(rd);
    rd_utils::free_rid(rd, nearest_sampler);
}

void ChronosOpaqueCaptureEffect::mark_captured(uint64_t p_scene_buffers_id) {
    std::lock_guard<std::mutex> lock(g_capture_mutex);
    g_fresh_captures.insert(p_scene_buffers_id);
}

bool ChronosOpaqueCaptureEffect::consume_capture(uint64_t p_scene_buffers_id) {
    std::lock_guard<std::mutex> lock(g_capture_mutex);
    return g_fresh_captures.erase(p_scene_buffers_id) > 0;
}

void ChronosOpaqueCaptureEffect::_render_callback(int32_t p_effect_callback_type, godot::RenderData *p_render_data) {
    if (p_render_data == nullptr || godot::RenderingServer::get_singleton() == nullptr) {
        return;
    }
    if (rd == nullptr) {
        rd = godot::RenderingServer::get_singleton()->get_rendering_device();
    }
    if (rd != nullptr && !init_attempted) {
        init_attempted = true;  // a failed compile reports once, not every frame
        nearest_sampler = rd_utils::create_sampler(rd, false);
        copy.init(rd);
    }
    if (!copy.is_valid() || !nearest_sampler.is_valid()) {
        return;
    }
    godot::Ref<godot::RenderSceneBuffers> base_buffers = p_render_data->get_render_scene_buffers();
    auto *scene_buffers = godot::Object::cast_to<godot::RenderSceneBuffersRD>(base_buffers.ptr());
    if (rd == nullptr || scene_buffers == nullptr) {
        return;
    }

    const godot::Vector2i size = scene_buffers->get_internal_size();
    const uint32_t view_count = scene_buffers->get_view_count();
    if (size.x <= 0 || size.y <= 0 || view_count == 0) {
        return;
    }

    // Scene buffers drop named textures whenever they are reconfigured (resize, view count),
    // so creating on demand keeps the size in sync.
    if (!scene_buffers->has_texture(kContext, kTextureName)) {
        scene_buffers->create_texture(
            kContext, kTextureName,
            godot::RenderingDevice::DATA_FORMAT_R16G16B16A16_SFLOAT,
            godot::RenderingDevice::TEXTURE_USAGE_SAMPLING_BIT | godot::RenderingDevice::TEXTURE_USAGE_STORAGE_BIT,
            godot::RenderingDevice::TEXTURE_SAMPLES_1,
            size, view_count, 1, true, false
        );
    }
    // Compute copy per view: Godot 4.4's color buffer can be sampled but not used as a
    // texture_copy source. Per-view layer views also pick the right XR eye.
    for (uint32_t view = 0; view < view_count; ++view) {
        const godot::RID src = scene_buffers->get_color_layer(view);
        const godot::RID dst = scene_buffers->get_texture_slice(kContext, kTextureName, view, 0, 1, 1);
        if (!src.is_valid() || !dst.is_valid()) {
            return;
        }
        copy.copy(rd, nearest_sampler, src, dst, size);
    }
    mark_captured(scene_buffers->get_instance_id());
}

} // namespace prism
