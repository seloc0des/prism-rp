#include "prism_rp/chronos_opaque_capture_effect.h"

#include <mutex>
#include <unordered_set>

#include <godot_cpp/classes/render_scene_buffers.hpp>
#include <godot_cpp/classes/render_scene_buffers_rd.hpp>
#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/variant/vector3.hpp>

namespace prism {

namespace {
std::mutex g_capture_mutex;
std::unordered_set<uint64_t> g_fresh_captures;
} // namespace

ChronosOpaqueCaptureEffect::ChronosOpaqueCaptureEffect() {
    set_effect_callback_type(EFFECT_CALLBACK_TYPE_PRE_TRANSPARENT);
    set_access_resolved_color(true);
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
    godot::RenderingDevice *rd = godot::RenderingServer::get_singleton()->get_rendering_device();
    godot::Ref<godot::RenderSceneBuffers> base_buffers = p_render_data->get_render_scene_buffers();
    auto *scene_buffers = godot::Object::cast_to<godot::RenderSceneBuffersRD>(base_buffers.ptr());
    if (rd == nullptr || scene_buffers == nullptr) {
        return;
    }

    const godot::Vector2i size = scene_buffers->get_internal_size();
    const uint32_t view_count = scene_buffers->get_view_count();
    const godot::RID color_full = scene_buffers->get_color_texture();
    if (size.x <= 0 || size.y <= 0 || view_count == 0 || !color_full.is_valid()) {
        return;
    }

    // Scene buffers drop named textures whenever they are reconfigured (resize, view count),
    // so creating on demand keeps the size in sync.
    if (!scene_buffers->has_texture(kContext, kTextureName)) {
        scene_buffers->create_texture(
            kContext, kTextureName,
            godot::RenderingDevice::DATA_FORMAT_R16G16B16A16_SFLOAT,
            godot::RenderingDevice::TEXTURE_USAGE_SAMPLING_BIT | godot::RenderingDevice::TEXTURE_USAGE_CAN_COPY_TO_BIT,
            godot::RenderingDevice::TEXTURE_SAMPLES_1,
            size, view_count, 1, true, false
        );
    }
    const godot::RID opaque_full = scene_buffers->get_texture(kContext, kTextureName);
    if (!opaque_full.is_valid()) {
        return;
    }

    // Copy from the full array textures with explicit layers: texture_copy addresses layers of
    // the underlying image, so copying from a per-view slice with layer 0 would always read eye 0.
    for (uint32_t view = 0; view < view_count; ++view) {
        rd->texture_copy(
            color_full, opaque_full,
            godot::Vector3(0, 0, 0), godot::Vector3(0, 0, 0),
            godot::Vector3(size.x, size.y, 1),
            0, 0, view, view
        );
    }
    mark_captured(scene_buffers->get_instance_id());
}

} // namespace prism
