#pragma once

// RenderingDevice helpers shared by the Prism RP compositor effects (Chronos, Lumina).

#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_set>

#include <godot_cpp/classes/rd_uniform.hpp>
#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/vector2i.hpp>

namespace prism::rd_utils {

template <typename T>
godot::PackedByteArray make_push_constant_bytes(const T &p_data) {
    godot::PackedByteArray bytes;
    bytes.resize(sizeof(T));
    std::memcpy(bytes.ptrw(), &p_data, sizeof(T));
    return bytes;
}

godot::Ref<godot::RDUniform> make_sampler_texture_uniform(int32_t p_binding, const godot::RID &p_sampler, const godot::RID &p_texture);
godot::Ref<godot::RDUniform> make_image_uniform(int32_t p_binding, const godot::RID &p_texture);

godot::RID create_sampler(godot::RenderingDevice *p_rd, bool p_linear);
godot::RID create_2d_texture(godot::RenderingDevice *p_rd, uint32_t p_width, uint32_t p_height,
        godot::RenderingDevice::DataFormat p_format, uint32_t p_usage_bits, uint32_t p_mipmaps = 1);

// Frees a valid RID and resets it.
void free_rid(godot::RenderingDevice *p_rd, godot::RID &r_rid);

// Inlines #include "..." (local directory first, then res://addons/prism_rp/shaders/include)
// and strips the GL_GOOGLE_include_directive line, which RD's GLSL compiler doesn't need.
godot::String load_and_preprocess_glsl(const godot::String &p_file_path, std::unordered_set<std::string> &r_visited_paths);

// Loads, preprocesses and compiles a compute shader. Returns an invalid RID and pushes an error on failure.
godot::RID compile_compute_shader(godot::RenderingDevice *p_rd, const godot::String &p_shader_path, const godot::String &p_debug_name);

// Compiles a shader and its compute pipeline. On failure both RIDs stay invalid.
bool build_compute_stage(godot::RenderingDevice *p_rd, const godot::String &p_shader_path, const godot::String &p_debug_name,
        godot::RID &r_shader, godot::RID &r_pipeline);

// Frees a pipeline (if still valid) and its shader.
void free_compute_stage(godot::RenderingDevice *p_rd, godot::RID &r_pipeline, godot::RID &r_shader);

// Records one compute dispatch: bind pipeline + set 0 + push constants.
void dispatch_compute(godot::RenderingDevice *p_rd, const godot::RID &p_pipeline, const godot::RID &p_uniform_set,
        const godot::PackedByteArray &p_push_constants, uint32_t p_groups_x, uint32_t p_groups_y, uint32_t p_groups_z = 1);

// The same with sets 0 and 1.
void dispatch_compute(godot::RenderingDevice *p_rd, const godot::RID &p_pipeline, const godot::RID &p_set0, const godot::RID &p_set1,
        const godot::PackedByteArray &p_push_constants, uint32_t p_groups_x, uint32_t p_groups_y, uint32_t p_groups_z = 1);

// Compute-shader copy (shaders/util/prism_copy.comp.glsl) for sources that can be sampled but
// not copied from, such as Godot 4.4's scene color buffer. Destination must be RGBA16F storage.
class ComputeCopy {
public:
    struct alignas(16) PushConstants {
        int32_t size_x = 0;
        int32_t size_y = 0;
        float _pad[2] = { 0.0f, 0.0f };
    };
    static_assert(sizeof(PushConstants) == 16);

    bool init(godot::RenderingDevice *p_rd);
    void release(godot::RenderingDevice *p_rd);
    [[nodiscard]] bool is_valid() const { return pipeline.is_valid(); }
    // Copies the top-left p_size texels of p_src (sampled with p_nearest_sampler) into p_dst.
    void copy(godot::RenderingDevice *p_rd, const godot::RID &p_nearest_sampler, const godot::RID &p_src,
            const godot::RID &p_dst, const godot::Vector2i &p_size) const;

private:
    godot::RID shader;
    godot::RID pipeline;
};

} // namespace prism::rd_utils
