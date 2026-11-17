#include "prism_rp/prism_rd_utils.h"

#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/uniform_set_cache_rd.hpp>
#include <godot_cpp/classes/rd_sampler_state.hpp>
#include <godot_cpp/classes/rd_shader_source.hpp>
#include <godot_cpp/classes/rd_shader_spirv.hpp>
#include <godot_cpp/classes/rd_texture_format.hpp>
#include <godot_cpp/classes/rd_texture_view.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

namespace prism::rd_utils {

godot::Ref<godot::RDUniform> make_sampler_texture_uniform(int32_t p_binding, const godot::RID &p_sampler, const godot::RID &p_texture) {
    godot::Ref<godot::RDUniform> u;
    u.instantiate();
    u->set_uniform_type(godot::RenderingDevice::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE);
    u->set_binding(p_binding);
    u->add_id(p_sampler);
    u->add_id(p_texture);
    return u;
}

godot::Ref<godot::RDUniform> make_image_uniform(int32_t p_binding, const godot::RID &p_texture) {
    godot::Ref<godot::RDUniform> u;
    u.instantiate();
    u->set_uniform_type(godot::RenderingDevice::UNIFORM_TYPE_IMAGE);
    u->set_binding(p_binding);
    u->add_id(p_texture);
    return u;
}

godot::RID create_sampler(godot::RenderingDevice *p_rd, bool p_linear) {
    const auto filter = p_linear ? godot::RenderingDevice::SAMPLER_FILTER_LINEAR : godot::RenderingDevice::SAMPLER_FILTER_NEAREST;
    godot::Ref<godot::RDSamplerState> state;
    state.instantiate();
    state->set_mag_filter(filter);
    state->set_min_filter(filter);
    state->set_mip_filter(filter);
    state->set_repeat_u(godot::RenderingDevice::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE);
    state->set_repeat_v(godot::RenderingDevice::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE);
    state->set_repeat_w(godot::RenderingDevice::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE);
    return p_rd->sampler_create(state);
}

godot::RID create_2d_texture(godot::RenderingDevice *p_rd, uint32_t p_width, uint32_t p_height,
        godot::RenderingDevice::DataFormat p_format, uint32_t p_usage_bits) {
    godot::Ref<godot::RDTextureFormat> fmt;
    fmt.instantiate();
    fmt->set_width(p_width);
    fmt->set_height(p_height);
    fmt->set_depth(1);
    fmt->set_array_layers(1);
    fmt->set_mipmaps(1);
    fmt->set_texture_type(godot::RenderingDevice::TEXTURE_TYPE_2D);
    fmt->set_format(p_format);
    fmt->set_samples(godot::RenderingDevice::TEXTURE_SAMPLES_1);
    fmt->set_usage_bits(p_usage_bits);

    godot::Ref<godot::RDTextureView> view;
    view.instantiate();
    return p_rd->texture_create(fmt, view);
}

void free_rid(godot::RenderingDevice *p_rd, godot::RID &r_rid) {
    if (p_rd != nullptr && r_rid.is_valid()) {
        p_rd->free_rid(r_rid);
    }
    r_rid = godot::RID();
}

godot::String load_and_preprocess_glsl(const godot::String &p_file_path, std::unordered_set<std::string> &r_visited_paths) {
    const std::string canonical_key = p_file_path.utf8().get_data();
    if (r_visited_paths.contains(canonical_key)) {
        return "";
    }
    r_visited_paths.insert(canonical_key);

    if (!godot::FileAccess::file_exists(p_file_path)) {
        godot::UtilityFunctions::push_error("[Prism RP] Shader file not found: ", p_file_path);
        return "";
    }

    const godot::String raw_source = godot::FileAccess::get_file_as_string(p_file_path);
    const godot::PackedStringArray lines = raw_source.split("\n");
    const godot::String parent_dir = p_file_path.get_base_dir();

    godot::String resolved_source;
    for (int64_t i = 0; i < lines.size(); ++i) {
        const godot::String line = lines[i];
        const godot::String trimmed = line.strip_edges();

        if (trimmed.begins_with("#extension GL_GOOGLE_include_directive")) {
            continue;
        }

        if (trimmed.begins_with("#include")) {
            const int64_t first_quote = trimmed.find("\"");
            const int64_t last_quote = trimmed.rfind("\"");
            if (first_quote != -1 && last_quote > first_quote) {
                const godot::String rel_include = trimmed.substr(first_quote + 1, last_quote - first_quote - 1);
                const godot::String candidate_local = parent_dir.path_join(rel_include);
                const godot::String candidate_shared = godot::String("res://addons/prism_rp/shaders/include").path_join(rel_include);
                const godot::String target_include = godot::FileAccess::file_exists(candidate_local) ? candidate_local : candidate_shared;

                resolved_source += load_and_preprocess_glsl(target_include, r_visited_paths) + godot::String("\n");
                continue;
            }
        }

        resolved_source += line + godot::String("\n");
    }

    return resolved_source;
}

godot::RID compile_compute_shader(godot::RenderingDevice *p_rd, const godot::String &p_shader_path, const godot::String &p_debug_name) {
    std::unordered_set<std::string> visited;
    const godot::String glsl_source = load_and_preprocess_glsl(p_shader_path, visited);
    if (glsl_source.is_empty()) {
        return godot::RID();
    }

    godot::Ref<godot::RDShaderSource> rd_source;
    rd_source.instantiate();
    rd_source->set_language(godot::RenderingDevice::SHADER_LANGUAGE_GLSL);
    rd_source->set_stage_source(godot::RenderingDevice::SHADER_STAGE_COMPUTE, glsl_source);

    const godot::Ref<godot::RDShaderSPIRV> spirv = p_rd->shader_compile_spirv_from_source(rd_source);
    if (spirv.is_null()) {
        godot::UtilityFunctions::push_error("[Prism RP] Null SPIR-V returned for shader: ", p_shader_path);
        return godot::RID();
    }

    const godot::String compile_error = spirv->get_stage_compile_error(godot::RenderingDevice::SHADER_STAGE_COMPUTE);
    if (!compile_error.is_empty()) {
        godot::UtilityFunctions::push_error("[Prism RP] SPIR-V compilation failed for ", p_shader_path, ":\n", compile_error);
        return godot::RID();
    }

    return p_rd->shader_create_from_spirv(spirv, p_debug_name);
}

bool build_compute_stage(godot::RenderingDevice *p_rd, const godot::String &p_shader_path, const godot::String &p_debug_name,
        godot::RID &r_shader, godot::RID &r_pipeline) {
    r_shader = compile_compute_shader(p_rd, p_shader_path, p_debug_name);
    if (!r_shader.is_valid()) {
        return false;
    }
    r_pipeline = p_rd->compute_pipeline_create(r_shader);
    if (!r_pipeline.is_valid()) {
        godot::UtilityFunctions::push_error("[Prism RP] Failed to create compute pipeline: ", p_debug_name);
        free_rid(p_rd, r_shader);
        return false;
    }
    return true;
}

void free_compute_stage(godot::RenderingDevice *p_rd, godot::RID &r_pipeline, godot::RID &r_shader) {
    if (p_rd != nullptr && r_pipeline.is_valid() && p_rd->compute_pipeline_is_valid(r_pipeline)) {
        p_rd->free_rid(r_pipeline);
    }
    r_pipeline = godot::RID();
    free_rid(p_rd, r_shader);
}

void dispatch_compute(godot::RenderingDevice *p_rd, const godot::RID &p_pipeline, const godot::RID &p_uniform_set,
        const godot::PackedByteArray &p_push_constants, uint32_t p_groups_x, uint32_t p_groups_y, uint32_t p_groups_z) {
    const int64_t cl = p_rd->compute_list_begin();
    p_rd->compute_list_bind_compute_pipeline(cl, p_pipeline);
    p_rd->compute_list_bind_uniform_set(cl, p_uniform_set, 0);
    p_rd->compute_list_set_push_constant(cl, p_push_constants, static_cast<uint32_t>(p_push_constants.size()));
    p_rd->compute_list_dispatch(cl, p_groups_x, p_groups_y, p_groups_z);
    p_rd->compute_list_end();
}

bool ComputeCopy::init(godot::RenderingDevice *p_rd) {
    if (is_valid()) {
        return true;
    }
    return build_compute_stage(p_rd, "res://addons/prism_rp/shaders/util/prism_copy.comp.glsl", "prism_copy", shader, pipeline);
}

void ComputeCopy::release(godot::RenderingDevice *p_rd) {
    free_compute_stage(p_rd, pipeline, shader);
}

void ComputeCopy::copy(godot::RenderingDevice *p_rd, const godot::RID &p_nearest_sampler, const godot::RID &p_src,
        const godot::RID &p_dst, const godot::Vector2i &p_size) const {
    if (!is_valid() || p_size.x <= 0 || p_size.y <= 0) {
        return;
    }
    godot::TypedArray<godot::RDUniform> u;
    u.push_back(make_sampler_texture_uniform(0, p_nearest_sampler, p_src));
    u.push_back(make_image_uniform(1, p_dst));
    const godot::RID uset = godot::UniformSetCacheRD::get_cache(shader, 0, u);
    if (!uset.is_valid()) {
        return;
    }
    PushConstants pc;
    pc.size_x = p_size.x;
    pc.size_y = p_size.y;
    dispatch_compute(p_rd, pipeline, uset, make_push_constant_bytes(pc),
        static_cast<uint32_t>((p_size.x + 7) / 8), static_cast<uint32_t>((p_size.y + 7) / 8));
}

} // namespace prism::rd_utils
