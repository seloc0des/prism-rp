#pragma once

#include <cstdint>

#include <godot_cpp/classes/compositor_effect.hpp>
#include <godot_cpp/classes/render_data.hpp>
#include <godot_cpp/variant/string_name.hpp>

namespace prism {

// Copies the opaque-only color buffer at PRE_TRANSPARENT into a scene-buffer texture shared
// per viewport. ChronosVPUCompositorEffect (POST_TRANSPARENT) compares it with the final frame
// to flag pixels touched by particles and transparent surfaces in the reactive mask. Those
// surfaces don't write depth or motion vectors, so without this they ghost in the TAA history.
// This is the same idea as FSR 2's ffxFsr2ContextGenerateReactiveMask (opaque-only vs. final).
//
// Usage: add it to the same Compositor as Chronos (order doesn't matter; callback types do).
class ChronosOpaqueCaptureEffect : public godot::CompositorEffect {
    GDCLASS(ChronosOpaqueCaptureEffect, godot::CompositorEffect)

public:
    static constexpr const char *kContext = "prism_chronos";
    static constexpr const char *kTextureName = "opaque_color";

    ChronosOpaqueCaptureEffect();

    void _render_callback(int32_t p_effect_callback_type, godot::RenderData *p_render_data) override;

    // Render-thread handshake keyed by the RenderSceneBuffersRD instance id. mark_captured()
    // after copying; consume_capture() returns true once per capture, so a removed capture
    // effect can never leave Chronos reading a stale opaque frame.
    static void mark_captured(uint64_t p_scene_buffers_id);
    static bool consume_capture(uint64_t p_scene_buffers_id);

protected:
    static void _bind_methods() {}
};

} // namespace prism
