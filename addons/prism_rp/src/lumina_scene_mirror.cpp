#include "prism_rp/lumina_scene_mirror.h"
#include "prism_rp/prism_rd_utils.h"
#include "prism_rp/prism_telemetry.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <godot_cpp/classes/array_mesh.hpp>
#include <godot_cpp/classes/base_material3d.hpp>
#include <godot_cpp/classes/editor_interface.hpp>
#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/classes/geometry_instance3d.hpp>
#include <godot_cpp/classes/mesh.hpp>
#include <godot_cpp/classes/mesh_instance3d.hpp>
#include <godot_cpp/classes/os.hpp>
#include <godot_cpp/classes/primitive_mesh.hpp>
#include <godot_cpp/classes/rd_acceleration_structure_geometry.hpp>
#include <godot_cpp/classes/rd_sampler_state.hpp>
#include <godot_cpp/classes/rd_texture_format.hpp>
#include <godot_cpp/classes/rd_texture_view.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/classes/scene_tree.hpp>
#include <godot_cpp/classes/texture2d.hpp>
#include <godot_cpp/classes/viewport.hpp>
#include <godot_cpp/classes/window.hpp>
#include <godot_cpp/classes/world3d.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/callable_method_pointer.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

namespace prism {

using godot::RenderingDevice;

LuminaSceneMirror *LuminaSceneMirror::singleton = nullptr;

namespace {

constexpr uint32_t kCornerBytes = 3 * sizeof(float);
constexpr uint32_t kMinPoolElements = 4096;
constexpr uint32_t kMinTableElements = 64;

// Same guard as PrismTelemetry: the RenderingServer is registered after extensions initialize.
godot::RenderingServer *rendering_server_if_ready() {
    godot::Engine *engine = godot::Engine::get_singleton();
    if (engine == nullptr || !engine->has_singleton("RenderingServer")) {
        return nullptr;
    }
    return godot::RenderingServer::get_singleton();
}

godot::SceneTree *scene_tree() {
    return godot::Object::cast_to<godot::SceneTree>(godot::Engine::get_singleton()->get_main_loop());
}

godot::PackedByteArray to_bytes(const void *p_data, size_t p_size) {
    godot::PackedByteArray bytes;
    bytes.resize(static_cast<int64_t>(p_size));
    if (p_size > 0) {
        std::memcpy(bytes.ptrw(), p_data, p_size);
    }
    return bytes;
}

template <typename T>
godot::PackedByteArray vector_bytes(const std::vector<T> &p_values) {
    return to_bytes(p_values.data(), p_values.size() * sizeof(T));
}

uint32_t float_bits(float p_value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &p_value, sizeof(bits));
    return bits;
}

// unpackUnorm4x8 order: r in the low byte.
uint32_t pack_unorm4x8(const godot::Color &p_color) {
    auto channel = [](float p_v) { return static_cast<uint32_t>(std::lround(std::clamp(p_v, 0.0f, 1.0f) * 255.0f)); };
    return channel(p_color.r) | (channel(p_color.g) << 8) | (channel(p_color.b) << 16) | (channel(p_color.a) << 24);
}

// Octahedral encoding (lumina_oct_decode's inverse) as unpackSnorm2x16, x in the low half.
uint32_t pack_oct_normal(const godot::Vector3 &p_normal) {
    const float sum = std::abs(p_normal.x) + std::abs(p_normal.y) + std::abs(p_normal.z);
    if (sum <= 0.0f) {
        return 0;
    }
    float x = p_normal.x / sum;
    float y = p_normal.y / sum;
    if (p_normal.z < 0.0f) {
        const float ox = (1.0f - std::abs(y)) * (x >= 0.0f ? 1.0f : -1.0f);
        const float oy = (1.0f - std::abs(x)) * (y >= 0.0f ? 1.0f : -1.0f);
        x = ox;
        y = oy;
    }
    auto snorm16 = [](float p_v) {
        return static_cast<uint32_t>(static_cast<uint16_t>(static_cast<int16_t>(std::lround(std::clamp(p_v, -1.0f, 1.0f) * 32767.0f))));
    };
    return snorm16(x) | (snorm16(y) << 16);
}

// Ray-query facing is decided on the raw vertex numbers as DXR does (front = clockwise in a left-handed frame),
// which makes Godot's clockwise front faces (right-handed) back faces. Flipping keeps "front" meaning the side
// Godot draws: its front for CULL_BACK and CULL_DISABLED, its back for CULL_FRONT.
uint32_t instance_flags_for(godot::BaseMaterial3D::CullMode p_cull, bool p_opaque) {
    uint32_t flags = p_opaque ? static_cast<uint32_t>(RenderingDevice::ACCELERATION_STRUCTURE_INSTANCE_FORCE_OPAQUE_BIT) : 0u;
    switch (p_cull) {
        case godot::BaseMaterial3D::CULL_DISABLED:
            flags |= RenderingDevice::ACCELERATION_STRUCTURE_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT |
                RenderingDevice::ACCELERATION_STRUCTURE_INSTANCE_TRIANGLE_FLIP_FACING_BIT;
            break;
        case godot::BaseMaterial3D::CULL_FRONT:
            break;
        default:
            flags |= RenderingDevice::ACCELERATION_STRUCTURE_INSTANCE_TRIANGLE_FLIP_FACING_BIT;
            break;
    }
    return flags;
}

godot::BaseMaterial3D::CullMode cull_mode_of(const godot::Ref<godot::Material> &p_material) {
    const auto *base = godot::Object::cast_to<godot::BaseMaterial3D>(p_material.ptr());
    return base != nullptr ? base->get_cull_mode() : godot::BaseMaterial3D::CULL_BACK;
}

uint32_t mask_for(godot::GeometryInstance3D::ShadowCastingSetting p_setting) {
    switch (p_setting) {
        case godot::GeometryInstance3D::SHADOW_CASTING_SETTING_OFF:
            return LuminaSceneMirror::kMaskCamera;
        case godot::GeometryInstance3D::SHADOW_CASTING_SETTING_SHADOWS_ONLY:
            return LuminaSceneMirror::kMaskShadow;
        default:
            return LuminaSceneMirror::kMaskCamera | LuminaSceneMirror::kMaskShadow;
    }
}

bool is_triangle_surface(const godot::Ref<godot::Mesh> &p_mesh, int32_t p_surface) {
    if (const auto *array_mesh = godot::Object::cast_to<godot::ArrayMesh>(p_mesh.ptr()); array_mesh != nullptr) {
        return array_mesh->surface_get_primitive_type(p_surface) == godot::Mesh::PRIMITIVE_TRIANGLES;
    }
    return godot::Object::cast_to<godot::PrimitiveMesh>(p_mesh.ptr()) != nullptr;
}

godot::Ref<godot::RDUniform> make_uniform(godot::RenderingDevice::UniformType p_type, int32_t p_binding, const godot::RID &p_id) {
    godot::Ref<godot::RDUniform> uniform;
    uniform.instantiate();
    uniform->set_uniform_type(p_type);
    uniform->set_binding(p_binding);
    uniform->add_id(p_id);
    return uniform;
}

} // namespace

LuminaSceneMirror *LuminaSceneMirror::get_singleton() { return singleton; }
void LuminaSceneMirror::set_singleton(LuminaSceneMirror *p_mirror) { singleton = p_mirror; }

void LuminaSceneMirror::connect_frame_hook() {
    godot::RenderingServer *rs = rendering_server_if_ready();
    if (rs == nullptr || frame_hook_connected) {
        return;
    }
    rs->connect("frame_pre_draw", callable_mp(this, &LuminaSceneMirror::_on_frame_pre_draw));
    frame_hook_connected = true;
    log_lifecycle = godot::OS::get_singleton()->get_cmdline_user_args().has("--prism-mirror-log");
}

void LuminaSceneMirror::_log(const godot::String &p_message) const {
    if (log_lifecycle) {
        godot::UtilityFunctions::print(p_message);
    } else {
        godot::UtilityFunctions::print_verbose(p_message);
    }
}

// Extension shutdown. The GPU side was released when the scene tree's root left the tree.
void LuminaSceneMirror::disconnect_frame_hook() {
    godot::RenderingServer *rs = rendering_server_if_ready();
    if (rs == nullptr || !frame_hook_connected) {
        return;
    }
    const godot::Callable cb = callable_mp(this, &LuminaSceneMirror::_on_frame_pre_draw);
    if (rs->is_connected("frame_pre_draw", cb)) {
        rs->disconnect("frame_pre_draw", cb);
    }
    frame_hook_connected = false;
}

godot::RID LuminaSceneMirror::scene_uniform_set(const godot::RID &p_shader) {
    request_serial.fetch_add(1, std::memory_order_relaxed);
    if (!scene_ready) {
        return godot::RID();
    }
    godot::RID &set = uniform_sets[p_shader.get_id()];
    // RenderingDevice frees a set along with any resource it uses, the shader included.
    if (!set.is_valid() || !rd->uniform_set_is_valid(set)) {
        set = rd->uniform_set_create(uniforms, p_shader, 1);
    }
    return set;
}

// --- Main thread -----------------------------------------------------------------------------------------------

void LuminaSceneMirror::_on_frame_pre_draw() {
    ++frame_index;
    const uint32_t serial = request_serial.load(std::memory_order_relaxed);
    if (serial != last_seen_request) {
        last_seen_request = serial;
        idle_frames = 0;
    } else if (idle_frames < kIdleFramesBeforeRelease) {
        ++idle_frames;
    }

    if (idle_frames >= kIdleFramesBeforeRelease) {
        if (tracking) {
            _stop_tracking();
        }
        return;
    }
    if (!tracking) {
        _start_tracking();
    }
    if (!tracking) {
        return;
    }
    _gather_frame();
    if (godot::RenderingServer *rs = rendering_server_if_ready(); rs != nullptr) {
        rs->call_on_render_thread(callable_mp(this, &LuminaSceneMirror::_apply_on_render_thread));
    }
}

void LuminaSceneMirror::_start_tracking() {
    godot::SceneTree *tree = scene_tree();
    if (tree == nullptr || tree->get_root() == nullptr) {
        return;
    }
    tree->connect("node_added", callable_mp(this, &LuminaSceneMirror::_on_node_added));
    tree->connect("node_removed", callable_mp(this, &LuminaSceneMirror::_on_node_removed));
    // Releases the GPU side while the RenderingServer still runs: the root leaves the tree when the main loop ends.
    tree->get_root()->connect("tree_exiting", callable_mp(this, &LuminaSceneMirror::_stop_tracking));
    tracking = true;
    _log("[Prism RP] Lumina scene mirror: following the scene tree.");

    std::vector<godot::Node *> stack = { tree->get_root() };
    while (!stack.empty()) {
        godot::Node *node = stack.back();
        stack.pop_back();
        _on_node_added(node);
        for (int32_t i = 0; i < node->get_child_count(true); ++i) {
            stack.push_back(node->get_child(i, true));
        }
    }
}

void LuminaSceneMirror::_stop_tracking() {
    if (!tracking) {
        return;
    }
    tracking = false;
    if (godot::SceneTree *tree = scene_tree(); tree != nullptr) {
        tree->disconnect("node_added", callable_mp(this, &LuminaSceneMirror::_on_node_added));
        tree->disconnect("node_removed", callable_mp(this, &LuminaSceneMirror::_on_node_removed));
        const godot::Callable on_exit = callable_mp(this, &LuminaSceneMirror::_stop_tracking);
        if (tree->get_root() != nullptr && tree->get_root()->is_connected("tree_exiting", on_exit)) {
            tree->get_root()->disconnect("tree_exiting", on_exit);
        }
    }
    const godot::Callable on_changed = callable_mp(this, &LuminaSceneMirror::_on_mesh_changed);
    for (const auto &[mesh_object_id, mesh] : main_meshes) {
        godot::Object *object = godot::ObjectDB::get_instance(mesh_object_id);
        if (object != nullptr && object->is_connected("changed", on_changed)) {
            object->disconnect("changed", on_changed);
        }
    }
    tracked_nodes.clear();
    main_meshes.clear();
    {
        std::lock_guard lock(pending_mutex);
        pending = Pending{};
        pending.release = true;
    }
    if (godot::RenderingServer *rs = rendering_server_if_ready(); rs != nullptr) {
        rs->call_on_render_thread(callable_mp(this, &LuminaSceneMirror::_apply_on_render_thread));
    }
}

void LuminaSceneMirror::_on_node_added(godot::Node *p_node) {
    if (godot::Object::cast_to<godot::MeshInstance3D>(p_node) != nullptr) {
        tracked_nodes.insert(p_node->get_instance_id());
    }
}

void LuminaSceneMirror::_on_node_removed(godot::Node *p_node) {
    tracked_nodes.erase(p_node->get_instance_id());
}

void LuminaSceneMirror::_on_mesh_changed(uint64_t p_mesh_object_id) {
    if (auto it = main_meshes.find(p_mesh_object_id); it != main_meshes.end()) {
        it->second.dirty = true;
    }
}

godot::World3D *LuminaSceneMirror::_mirrored_world() const {
    godot::Viewport *viewport = nullptr;
    if (godot::Engine::get_singleton()->is_editor_hint()) {
        godot::EditorInterface *editor = godot::EditorInterface::get_singleton();
        godot::Node *edited = editor != nullptr ? editor->get_edited_scene_root() : nullptr;
        viewport = edited != nullptr ? edited->get_viewport() : nullptr;
    } else if (godot::SceneTree *tree = scene_tree(); tree != nullptr) {
        viewport = tree->get_root();
    }
    return viewport != nullptr ? viewport->find_world_3d().ptr() : nullptr;
}

void LuminaSceneMirror::_gather_frame() {
    godot::World3D *world = _mirrored_world();
    FrameState frame;
    frame.materials.emplace_back();   // index 0: Godot's default material, white and opaque
    std::unordered_map<uint64_t, uint32_t> material_seen;

    for (auto it = tracked_nodes.begin(); it != tracked_nodes.end();) {
        auto *instance = godot::Object::cast_to<godot::MeshInstance3D>(godot::ObjectDB::get_instance(*it));
        if (instance == nullptr) {
            it = tracked_nodes.erase(it);
            continue;
        }
        ++it;
        _gather_instance(instance, world, frame, material_seen);
    }
    _drop_unused_meshes();

    std::lock_guard lock(pending_mutex);
    for (MeshUpload &upload : gathered_uploads) {
        pending.uploads.push_back(std::move(upload));
    }
    pending.removals.insert(pending.removals.end(), gathered_removals.begin(), gathered_removals.end());
    gathered_uploads.clear();
    gathered_removals.clear();
    pending.frame = std::move(frame);
    pending.has_frame = true;
}

void LuminaSceneMirror::_gather_instance(godot::MeshInstance3D *p_instance, godot::World3D *p_world, FrameState &r_frame,
        std::unordered_map<uint64_t, uint32_t> &r_material_seen) {
    if (p_world == nullptr || !p_instance->is_visible_in_tree() ||
            p_instance->get_gi_mode() == godot::GeometryInstance3D::GI_MODE_DISABLED ||
            p_instance->get_world_3d().ptr() != p_world) {
        return;
    }
    const godot::Ref<godot::Mesh> mesh = p_instance->get_mesh();
    if (mesh.is_null()) {
        return;
    }
    const MainMesh *main_mesh = _main_mesh_for(mesh);
    if (main_mesh == nullptr || main_mesh->mesh_id == 0) {
        return;
    }

    FrameInstance instance;
    instance.mesh_id = main_mesh->mesh_id;
    instance.transform = p_instance->get_global_transform();
    instance.mask = mask_for(p_instance->get_cast_shadows_setting());
    instance.material_base = static_cast<uint32_t>(r_frame.instance_materials.size());
    bool opaque = true;
    for (uint32_t s = 0; s < main_mesh->surface_count; ++s) {
        const uint32_t index = _material_index(p_instance->get_active_material(static_cast<int32_t>(s)), r_frame, r_material_seen);
        opaque = opaque && (r_frame.materials[index].record.flags & (kMaterialAlphaTest | kMaterialSkip)) == 0;
        r_frame.instance_materials.push_back(index);
    }
    // Ray flags cull per instance, so the first surface's cull mode stands for the whole mesh.
    instance.flags = instance_flags_for(cull_mode_of(p_instance->get_active_material(0)), opaque);
    r_frame.instances.push_back(instance);
}

const LuminaSceneMirror::MainMesh *LuminaSceneMirror::_main_mesh_for(const godot::Ref<godot::Mesh> &p_mesh) {
    const uint64_t key = p_mesh->get_instance_id();
    auto [it, inserted] = main_meshes.try_emplace(key);
    MainMesh &mesh = it->second;
    mesh.last_used_frame = frame_index;
    if (inserted) {
        p_mesh->connect("changed", callable_mp(this, &LuminaSceneMirror::_on_mesh_changed).bind(key));
    }
    if (mesh.dirty) {
        if (mesh.mesh_id != 0) {
            gathered_removals.push_back(mesh.mesh_id);
        }
        mesh = MainMesh{};
        mesh.last_used_frame = frame_index;
    }
    if (mesh.mesh_id == 0 && mesh.mirrorable) {
        MeshUpload upload;
        mesh.surface_count = static_cast<uint32_t>(p_mesh->get_surface_count());
        mesh.mirrorable = _read_mesh(p_mesh, upload);
        if (mesh.mirrorable) {
            upload.mesh_id = next_mesh_id++;
            mesh.mesh_id = upload.mesh_id;
            gathered_uploads.push_back(std::move(upload));
        }
    }
    return &mesh;
}

// Meshes nobody has drawn for a while leave the GPU; entries go once their Mesh is freed.
void LuminaSceneMirror::_drop_unused_meshes() {
    for (auto it = main_meshes.begin(); it != main_meshes.end();) {
        MainMesh &mesh = it->second;
        const bool freed = godot::ObjectDB::get_instance(it->first) == nullptr;
        if (mesh.mesh_id != 0 && (freed || frame_index - mesh.last_used_frame > kUnusedFramesBeforeDrop)) {
            gathered_removals.push_back(mesh.mesh_id);
            mesh.mesh_id = 0;
        }
        it = freed ? main_meshes.erase(it) : std::next(it);
    }
}

uint32_t LuminaSceneMirror::_material_index(const godot::Ref<godot::Material> &p_material, FrameState &r_frame,
        std::unordered_map<uint64_t, uint32_t> &r_seen) {
    const auto *base = godot::Object::cast_to<godot::BaseMaterial3D>(p_material.ptr());
    if (base == nullptr) {
        return 0;   // no material, or a ShaderMaterial: drawn here as Godot's default material
    }
    const uint64_t key = base->get_instance_id();
    if (auto it = r_seen.find(key); it != r_seen.end()) {
        return it->second;
    }

    FrameMaterial material;
    MaterialRecord &record = material.record;
    const godot::Color albedo = base->get_albedo();
    const godot::Color linear = albedo.srgb_to_linear();
    record.albedo[0] = linear.r;
    record.albedo[1] = linear.g;
    record.albedo[2] = linear.b;
    record.albedo[3] = albedo.a;
    const godot::Vector3 scale = base->get_uv1_scale();
    const godot::Vector3 offset = base->get_uv1_offset();
    record.uv_transform[0] = scale.x;
    record.uv_transform[1] = scale.y;
    record.uv_transform[2] = offset.x;
    record.uv_transform[3] = offset.y;
    if (const godot::Ref<godot::Texture2D> texture = base->get_texture(godot::BaseMaterial3D::TEXTURE_ALBEDO); texture.is_valid()) {
        material.albedo_texture = texture->get_rid();
    }
    switch (base->get_transparency()) {
        case godot::BaseMaterial3D::TRANSPARENCY_ALPHA_SCISSOR:
            record.flags |= kMaterialAlphaTest;
            record.alpha_scissor = base->get_alpha_scissor_threshold();
            break;
        case godot::BaseMaterial3D::TRANSPARENCY_ALPHA_HASH:
            record.flags |= kMaterialAlphaTest;   // Godot's hash is a per-pixel dither; a ray takes its mean, a 0.5 cut
            break;
        case godot::BaseMaterial3D::TRANSPARENCY_ALPHA:
        case godot::BaseMaterial3D::TRANSPARENCY_ALPHA_DEPTH_PRE_PASS:
            record.flags |= kMaterialSkip;
            break;
        default:
            break;
    }
    if (base->get_flag(godot::BaseMaterial3D::FLAG_ALBEDO_FROM_VERTEX_COLOR)) {
        record.flags |= kMaterialVertexColor;
        if (base->get_flag(godot::BaseMaterial3D::FLAG_SRGB_VERTEX_COLOR)) {
            record.flags |= kMaterialVertexColorSrgb;
        }
    }

    const uint32_t index = static_cast<uint32_t>(r_frame.materials.size());
    r_frame.materials.push_back(material);
    r_seen.emplace(key, index);
    return index;
}

// Reads every triangle surface. Returns false when there is nothing to mirror: no triangles, or a skinned or
// blend-shape mesh (not mirrored yet).
// An ArrayMesh's arrays come back from the GPU, stalling it, so its format is checked before anything is read.
bool LuminaSceneMirror::_read_mesh(const godot::Ref<godot::Mesh> &p_mesh, MeshUpload &r_upload) {
    if (const auto *array_mesh = godot::Object::cast_to<godot::ArrayMesh>(p_mesh.ptr()); array_mesh != nullptr) {
        if (array_mesh->get_blend_shape_count() > 0) {
            return false;
        }
        for (int32_t s = 0; s < array_mesh->get_surface_count(); ++s) {
            if ((array_mesh->surface_get_format(s) & godot::Mesh::ARRAY_FORMAT_BONES) != 0) {
                return false;
            }
        }
    }
    for (int32_t s = 0; s < p_mesh->get_surface_count(); ++s) {
        if (!is_triangle_surface(p_mesh, s)) {
            continue;
        }
        const godot::Array arrays = p_mesh->surface_get_arrays(s);
        if (arrays.size() != godot::Mesh::ARRAY_MAX) {
            continue;
        }
        if (arrays[godot::Mesh::ARRAY_BONES].get_type() != godot::Variant::NIL) {
            return false;   // a PrimitiveMesh with bones, such as the trail meshes
        }
        const godot::PackedVector3Array vertices = arrays[godot::Mesh::ARRAY_VERTEX];
        const godot::PackedVector3Array normals = arrays[godot::Mesh::ARRAY_NORMAL];
        const godot::PackedVector2Array uvs = arrays[godot::Mesh::ARRAY_TEX_UV];
        const godot::PackedColorArray colors = arrays[godot::Mesh::ARRAY_COLOR];
        godot::PackedInt32Array index_array = arrays[godot::Mesh::ARRAY_INDEX];
        const int64_t vertex_count = vertices.size();
        if (index_array.is_empty()) {
            index_array.resize(vertex_count);
            for (int64_t i = 0; i < vertex_count; ++i) {
                index_array.set(i, static_cast<int32_t>(i));
            }
        }
        const int64_t corner_count = index_array.size() - index_array.size() % 3;
        if (corner_count == 0) {
            continue;
        }

        SurfaceGeometry geometry;
        geometry.material_slot = static_cast<uint32_t>(s);
        geometry.corners.reserve(static_cast<size_t>(corner_count) * 3);
        geometry.indices.reserve(static_cast<size_t>(corner_count));
        bool valid = true;
        for (int64_t c = 0; c < corner_count && valid; ++c) {
            const int32_t index = index_array[c];
            valid = index >= 0 && index < vertex_count;
            if (valid) {
                const godot::Vector3 &p = vertices[index];
                geometry.corners.insert(geometry.corners.end(), { p.x, p.y, p.z });
                geometry.indices.push_back(static_cast<uint32_t>(index));
            }
        }
        if (!valid) {
            continue;
        }
        geometry.attributes.reserve(static_cast<size_t>(vertex_count) * kAttributeWords);
        for (int64_t v = 0; v < vertex_count; ++v) {
            const godot::Vector2 uv = v < uvs.size() ? uvs[v] : godot::Vector2();
            geometry.attributes.push_back(float_bits(uv.x));
            geometry.attributes.push_back(float_bits(uv.y));
            geometry.attributes.push_back(v < colors.size() ? pack_unorm4x8(colors[v]) : 0xFFFFFFFFu);
            geometry.attributes.push_back(v < normals.size() ? pack_oct_normal(normals[v]) : 0u);
        }
        r_upload.surfaces.push_back(std::move(geometry));
    }
    return !r_upload.surfaces.empty();
}

// --- Render thread ---------------------------------------------------------------------------------------------

void LuminaSceneMirror::_apply_on_render_thread() {
    Pending work;
    {
        std::lock_guard lock(pending_mutex);
        std::swap(work, pending);
    }
    if (work.release) {
        _release_on_render_thread();
    }
    if ((work.uploads.empty() && !work.has_frame) || !_ensure_render_resources()) {
        return;
    }

    ScopedPassTimer timer("lumina_scene_mirror", godot::Color(0.40f, 0.75f, 0.45f, 1.0f));
    for (const uint32_t mesh_id : work.removals) {
        _remove_mesh(mesh_id);
    }
    uint32_t corners = 0;
    uint32_t vertices = 0;
    uint32_t geometries = 0;
    for (const MeshUpload &upload : work.uploads) {
        for (const SurfaceGeometry &surface : upload.surfaces) {
            corners += static_cast<uint32_t>(surface.indices.size());
            vertices += static_cast<uint32_t>(surface.attributes.size() / kAttributeWords);
        }
        geometries += static_cast<uint32_t>(upload.surfaces.size());
    }
    if (!_ensure_pool_space(corners, vertices, geometries)) {
        scene_ready = false;
        return;
    }
    for (MeshUpload &upload : work.uploads) {
        _upload_mesh(upload);
    }
    if (work.has_frame) {
        scene_ready = _write_frame(work.frame) && _build_tlas(work.frame);
    }
    if (scene_ready && uniforms_dirty) {
        _rebuild_uniforms();
    }
}

bool LuminaSceneMirror::_ensure_render_resources() {
    if (rd != nullptr) {
        return true;
    }
    godot::RenderingDevice *device = godot::RenderingServer::get_singleton()->get_rendering_device();
    if (device == nullptr || !device->has_feature(RenderingDevice::SUPPORTS_RAY_QUERY)) {
        return false;
    }

    godot::Ref<godot::RDSamplerState> state;
    state.instantiate();
    state->set_mag_filter(RenderingDevice::SAMPLER_FILTER_LINEAR);
    state->set_min_filter(RenderingDevice::SAMPLER_FILTER_LINEAR);
    state->set_mip_filter(RenderingDevice::SAMPLER_FILTER_LINEAR);
    state->set_repeat_u(RenderingDevice::SAMPLER_REPEAT_MODE_REPEAT);
    state->set_repeat_v(RenderingDevice::SAMPLER_REPEAT_MODE_REPEAT);
    godot::RID new_sampler = device->sampler_create(state);

    godot::Ref<godot::RDTextureFormat> format;
    format.instantiate();
    format->set_format(RenderingDevice::DATA_FORMAT_R8G8B8A8_UNORM);
    format->set_width(1);
    format->set_height(1);
    format->set_usage_bits(RenderingDevice::TEXTURE_USAGE_SAMPLING_BIT | RenderingDevice::TEXTURE_USAGE_CAN_UPDATE_BIT);
    godot::PackedByteArray white;
    white.resize(4);
    white.fill(255);
    godot::TypedArray<godot::PackedByteArray> data;
    data.push_back(white);
    godot::RID new_white = device->texture_create(format, memnew(godot::RDTextureView), data);
    if (!new_sampler.is_valid() || !new_white.is_valid()) {
        rd_utils::free_rid(device, new_sampler);
        rd_utils::free_rid(device, new_white);
        return false;
    }
    rd = device;
    sampler = new_sampler;
    white_texture = new_white;
    texture_slots = { TextureSlot{ 0, white_texture, 0 } };
    uniforms_dirty = true;
    return true;
}

void LuminaSceneMirror::_release_on_render_thread() {
    if (rd == nullptr) {
        return;
    }
    _log(godot::String("[Prism RP] Lumina scene mirror: released (") + godot::itos(static_cast<int64_t>(render_meshes.size())) + " meshes).");
    _free_uniform_sets();
    rd_utils::free_rid(rd, tlas);
    for (auto &[mesh_id, mesh] : render_meshes) {
        rd_utils::free_rid(rd, mesh.blas);
    }
    render_meshes.clear();
    for (Pool *pool : { &positions, &indices, &attributes, &surfaces }) {
        rd_utils::free_rid(rd, pool->buffer);
        *pool = Pool{};
    }
    rd_utils::free_rid(rd, instance_buffer);
    rd_utils::free_rid(rd, instance_material_buffer);
    rd_utils::free_rid(rd, material_buffer);
    rd_utils::free_rid(rd, sampler);
    rd_utils::free_rid(rd, white_texture);
    tlas_capacity = instance_capacity = instance_material_capacity = material_capacity = 0;
    texture_slots.clear();
    slot_by_texture.clear();
    free_texture_slots.clear();
    tlas_instance_pool.clear();
    uniforms.clear();
    uniforms_dirty = true;
    scene_ready = false;
    rd = nullptr;
}

void LuminaSceneMirror::_remove_mesh(uint32_t p_mesh_id) {
    auto it = render_meshes.find(p_mesh_id);
    if (it == render_meshes.end()) {
        return;
    }
    rd_utils::free_rid(rd, it->second.blas);   // its pool ranges stay as holes until the next repack
    render_meshes.erase(it);
}

// A mesh whose BLAS fails stays out of the scene until it changes: the main thread keeps its id.
void LuminaSceneMirror::_upload_mesh(MeshUpload &p_upload) {
    RenderMesh mesh;
    mesh.corner_base = positions.used;
    mesh.vertex_base = attributes.used;
    mesh.surface_base = surfaces.used;
    std::vector<float> corner_data;
    std::vector<uint32_t> index_data;
    std::vector<uint32_t> attribute_data;
    for (const SurfaceGeometry &surface : p_upload.surfaces) {
        GeometryRange range;
        range.corner_offset = mesh.corner_count;
        range.vertex_offset = mesh.vertex_count;
        range.triangle_count = static_cast<uint32_t>(surface.indices.size() / 3);
        range.material_slot = surface.material_slot;
        mesh.geometries.push_back(range);
        mesh.corner_count += static_cast<uint32_t>(surface.indices.size());
        mesh.vertex_count += static_cast<uint32_t>(surface.attributes.size() / kAttributeWords);
        corner_data.insert(corner_data.end(), surface.corners.begin(), surface.corners.end());
        index_data.insert(index_data.end(), surface.indices.begin(), surface.indices.end());
        attribute_data.insert(attribute_data.end(), surface.attributes.begin(), surface.attributes.end());
    }
    rd->buffer_update(positions.buffer, mesh.corner_base * kCornerBytes, mesh.corner_count * kCornerBytes, vector_bytes(corner_data));
    rd->buffer_update(indices.buffer, mesh.corner_base * sizeof(uint32_t), mesh.corner_count * sizeof(uint32_t), vector_bytes(index_data));
    rd->buffer_update(attributes.buffer, mesh.vertex_base * kAttributeWords * sizeof(uint32_t),
        mesh.vertex_count * kAttributeWords * sizeof(uint32_t), vector_bytes(attribute_data));
    positions.used += mesh.corner_count;
    indices.used += mesh.corner_count;
    attributes.used += mesh.vertex_count;
    surfaces.used += static_cast<uint32_t>(mesh.geometries.size());

    _write_surface_records(mesh);
    if (!_build_blas(mesh)) {
        godot::UtilityFunctions::push_error("[Prism RP] Lumina scene mirror: a mesh's BLAS failed to build; it is left out.");
        return;
    }
    render_meshes.emplace(p_upload.mesh_id, std::move(mesh));
}

bool LuminaSceneMirror::_ensure_pool_space(uint32_t p_corners, uint32_t p_vertices, uint32_t p_surfaces) {
    if (positions.buffer.is_valid() && positions.used + p_corners <= positions.capacity &&
            attributes.used + p_vertices <= attributes.capacity && surfaces.used + p_surfaces <= surfaces.capacity) {
        return true;
    }
    uint32_t live_corners = p_corners;
    uint32_t live_vertices = p_vertices;
    uint32_t live_surfaces = p_surfaces;
    for (const auto &[mesh_id, mesh] : render_meshes) {
        live_corners += mesh.corner_count;
        live_vertices += mesh.vertex_count;
        live_surfaces += static_cast<uint32_t>(mesh.geometries.size());
    }
    auto grown = [](uint32_t p_live) { return std::max(p_live + p_live / 2, kMinPoolElements); };
    return _repack(grown(live_corners), grown(live_vertices), grown(live_surfaces));
}

// Moves every live mesh into new pools of the given capacities, dropping holes, and rebuilds the BLASes against them.
bool LuminaSceneMirror::_repack(uint32_t p_corner_capacity, uint32_t p_vertex_capacity, uint32_t p_surface_capacity) {
    const uint32_t as_input = RenderingDevice::BUFFER_CREATION_DEVICE_ADDRESS_BIT |
        RenderingDevice::BUFFER_CREATION_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT | RenderingDevice::BUFFER_CREATION_AS_STORAGE_BIT;
    Pool new_positions{ rd->vertex_buffer_create(p_corner_capacity * kCornerBytes, godot::PackedByteArray(), as_input), p_corner_capacity, 0 };
    Pool new_indices{ rd->storage_buffer_create(p_corner_capacity * sizeof(uint32_t)), p_corner_capacity, 0 };
    Pool new_attributes{ rd->storage_buffer_create(p_vertex_capacity * kAttributeWords * sizeof(uint32_t)), p_vertex_capacity, 0 };
    Pool new_surfaces{ rd->storage_buffer_create(p_surface_capacity * sizeof(SurfaceRecord)), p_surface_capacity, 0 };
    if (!new_positions.buffer.is_valid() || !new_indices.buffer.is_valid() || !new_attributes.buffer.is_valid() || !new_surfaces.buffer.is_valid()) {
        for (Pool *pool : { &new_positions, &new_indices, &new_attributes, &new_surfaces }) {
            rd_utils::free_rid(rd, pool->buffer);
        }
        godot::UtilityFunctions::push_error("[Prism RP] Lumina scene mirror: could not allocate its geometry buffers.");
        return false;
    }

    for (auto &[mesh_id, mesh] : render_meshes) {
        if (mesh.corner_count > 0) {
            rd->buffer_copy(positions.buffer, new_positions.buffer, mesh.corner_base * kCornerBytes, new_positions.used * kCornerBytes, mesh.corner_count * kCornerBytes);
            rd->buffer_copy(indices.buffer, new_indices.buffer, mesh.corner_base * sizeof(uint32_t), new_indices.used * sizeof(uint32_t), mesh.corner_count * sizeof(uint32_t));
        }
        if (mesh.vertex_count > 0) {
            const uint32_t stride = kAttributeWords * sizeof(uint32_t);
            rd->buffer_copy(attributes.buffer, new_attributes.buffer, mesh.vertex_base * stride, new_attributes.used * stride, mesh.vertex_count * stride);
        }
        rd_utils::free_rid(rd, mesh.blas);   // it reads the old position pool
        mesh.corner_base = new_positions.used;
        mesh.vertex_base = new_attributes.used;
        mesh.surface_base = new_surfaces.used;
        new_positions.used += mesh.corner_count;
        new_indices.used += mesh.corner_count;
        new_attributes.used += mesh.vertex_count;
        new_surfaces.used += static_cast<uint32_t>(mesh.geometries.size());
    }
    for (Pool *pool : { &positions, &indices, &attributes, &surfaces }) {
        rd_utils::free_rid(rd, pool->buffer);
    }
    positions = new_positions;
    indices = new_indices;
    attributes = new_attributes;
    surfaces = new_surfaces;
    uniforms_dirty = true;

    for (auto it = render_meshes.begin(); it != render_meshes.end();) {
        _write_surface_records(it->second);
        if (_build_blas(it->second)) {
            ++it;
        } else {
            godot::UtilityFunctions::push_error("[Prism RP] Lumina scene mirror: a mesh's BLAS failed to rebuild; it is left out.");
            it = render_meshes.erase(it);
        }
    }
    return true;
}

bool LuminaSceneMirror::_build_blas(RenderMesh &r_mesh) {
    // Non-indexed: Godot 4.7.2's blas_create checks an index buffer's largest index, which release builds never
    // record (it stays 0xFFFFFFFF), so an indexed BLAS fails in every release export.
    godot::TypedArray<godot::RDAccelerationStructureGeometry> geometries;
    for (const GeometryRange &range : r_mesh.geometries) {
        godot::Ref<godot::RDAccelerationStructureGeometry> geometry;
        geometry.instantiate();
        geometry->set_flags(0);   // not opaque: instances whose materials all are say so with FORCE_OPAQUE
        geometry->set_vertex_buffer(positions.buffer);
        geometry->set_vertex_offset((r_mesh.corner_base + range.corner_offset) * kCornerBytes);
        geometry->set_vertex_stride(kCornerBytes);
        geometry->set_vertex_count(range.triangle_count * 3);
        geometry->set_vertex_format(RenderingDevice::DATA_FORMAT_R32G32B32_SFLOAT);
        geometries.push_back(geometry);
    }
    r_mesh.blas = rd->blas_create(geometries, RenderingDevice::ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT);
    if (!r_mesh.blas.is_valid()) {
        return false;
    }
    if (rd->blas_build(r_mesh.blas) != godot::OK) {
        rd_utils::free_rid(rd, r_mesh.blas);
        return false;
    }
    return true;
}

void LuminaSceneMirror::_write_surface_records(const RenderMesh &p_mesh) {
    std::vector<SurfaceRecord> records;
    records.reserve(p_mesh.geometries.size());
    for (const GeometryRange &range : p_mesh.geometries) {
        records.push_back({ p_mesh.corner_base + range.corner_offset, p_mesh.vertex_base + range.vertex_offset, range.triangle_count, range.material_slot });
    }
    rd->buffer_update(surfaces.buffer, p_mesh.surface_base * sizeof(SurfaceRecord),
        static_cast<uint32_t>(records.size() * sizeof(SurfaceRecord)), vector_bytes(records));
}

bool LuminaSceneMirror::_write_frame(const FrameState &p_frame) {
    ++render_frame;
    std::vector<MaterialRecord> records;
    records.reserve(p_frame.materials.size());
    for (const FrameMaterial &material : p_frame.materials) {
        MaterialRecord record = material.record;
        record.albedo_texture = _texture_slot(material.albedo_texture);
        records.push_back(record);
    }
    _age_texture_slots();

    std::vector<uint32_t> instance_materials = p_frame.instance_materials;
    if (instance_materials.empty()) {
        instance_materials.push_back(0);
    }
    if (!_ensure_buffer(material_buffer, material_capacity, static_cast<uint32_t>(records.size()), sizeof(MaterialRecord)) ||
            !_ensure_buffer(instance_material_buffer, instance_material_capacity, static_cast<uint32_t>(instance_materials.size()), sizeof(uint32_t))) {
        return false;
    }
    rd->buffer_update(material_buffer, 0, static_cast<uint32_t>(records.size() * sizeof(MaterialRecord)), vector_bytes(records));
    rd->buffer_update(instance_material_buffer, 0, static_cast<uint32_t>(instance_materials.size() * sizeof(uint32_t)), vector_bytes(instance_materials));
    return true;
}

uint32_t LuminaSceneMirror::_texture_slot(const godot::RID &p_texture) {
    if (!p_texture.is_valid()) {
        return 0;
    }
    const int64_t key = p_texture.get_id();
    const auto found = slot_by_texture.find(key);
    if (found != slot_by_texture.end() && texture_slots[found->second].last_used_frame == render_frame) {
        return found->second;   // already resolved this frame
    }
    // The sRGB view decodes albedo the way Godot's source_color sampling does. Formats without one get the plain view.
    const godot::RID rd_texture = godot::RenderingServer::get_singleton()->texture_get_rd_texture(p_texture, true);
    if (!rd_texture.is_valid() || !rd->texture_is_valid(rd_texture)) {
        return 0;
    }
    uint32_t slot = 0;
    if (found != slot_by_texture.end()) {
        slot = found->second;
    } else if (!free_texture_slots.empty()) {
        slot = free_texture_slots.back();
        free_texture_slots.pop_back();
    } else if (texture_slots.size() < kMaxTextures) {
        slot = static_cast<uint32_t>(texture_slots.size());
        texture_slots.emplace_back();
    } else {
        WARN_PRINT_ONCE("[Prism RP] Lumina scene mirror: more albedo textures than the texture array holds; the rest are white.");
        return 0;
    }
    slot_by_texture[key] = slot;
    TextureSlot &entry = texture_slots[slot];
    entry.rs_texture = key;
    entry.last_used_frame = render_frame;
    if (entry.rd_texture != rd_texture) {
        entry.rd_texture = rd_texture;   // new, or Godot replaced the texture's GPU copy
        uniforms_dirty = true;
    }
    return slot;
}

// Frees the slots of textures nobody has used for a while, and at once those whose GPU texture is gone: a uniform
// set can't be made with a freed texture.
void LuminaSceneMirror::_age_texture_slots() {
    for (uint32_t slot = 1; slot < texture_slots.size(); ++slot) {
        TextureSlot &entry = texture_slots[slot];
        if (entry.rs_texture == 0) {
            continue;
        }
        if (render_frame - entry.last_used_frame > kUnusedFramesBeforeDrop || !rd->texture_is_valid(entry.rd_texture)) {
            slot_by_texture.erase(entry.rs_texture);
            entry = TextureSlot{};
            free_texture_slots.push_back(slot);
            uniforms_dirty = true;
        }
    }
}

bool LuminaSceneMirror::_ensure_buffer(godot::RID &r_buffer, uint32_t &r_capacity, uint32_t p_needed, uint32_t p_element_size) {
    if (r_buffer.is_valid() && p_needed <= r_capacity) {
        return true;
    }
    rd_utils::free_rid(rd, r_buffer);
    r_capacity = std::max(p_needed + p_needed / 2, kMinTableElements);
    r_buffer = rd->storage_buffer_create(r_capacity * p_element_size);
    uniforms_dirty = true;
    return r_buffer.is_valid();
}

bool LuminaSceneMirror::_build_tlas(const FrameState &p_frame) {
    std::vector<InstanceRecord> records;
    records.reserve(p_frame.instances.size());
    for (const FrameInstance &instance : p_frame.instances) {
        const auto mesh = render_meshes.find(instance.mesh_id);
        if (mesh == render_meshes.end()) {
            continue;   // its upload failed
        }
        const size_t n = records.size();
        if (n == tlas_instance_pool.size()) {
            tlas_instance_pool.emplace_back();
            tlas_instance_pool.back().instantiate();
        }
        const godot::Ref<godot::RDAccelerationStructureInstance> &entry = tlas_instance_pool[n];
        entry->set_blas(mesh->second.blas);
        entry->set_transform(instance.transform);
        entry->set_id(static_cast<uint32_t>(n));   // the instance's custom index: its InstanceRecord
        entry->set_mask(instance.mask);
        entry->set_flags(instance.flags);
        InstanceRecord record;
        record.surface_base = mesh->second.surface_base;
        record.material_base = instance.material_base;
        records.push_back(record);
    }
    if (records.empty()) {
        return false;   // nothing to trace: passes leave their input as it is
    }

    const uint32_t count = static_cast<uint32_t>(records.size());
    if (!tlas.is_valid() || count > tlas_capacity) {
        rd_utils::free_rid(rd, tlas);
        tlas_capacity = std::max(count + count / 2, kMinTableElements);
        tlas = rd->tlas_create(tlas_capacity, RenderingDevice::ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT);
        uniforms_dirty = true;
        if (!tlas.is_valid()) {
            return false;
        }
    }
    if (!_ensure_buffer(instance_buffer, instance_capacity, count, sizeof(InstanceRecord))) {
        return false;
    }
    rd->buffer_update(instance_buffer, 0, count * sizeof(InstanceRecord), vector_bytes(records));

    godot::TypedArray<godot::RDAccelerationStructureInstance> instances;
    instances.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        instances[i] = tlas_instance_pool[i];
    }
    return rd->tlas_build(tlas, instances) == godot::OK;
}

void LuminaSceneMirror::_rebuild_uniforms() {
    using U = RenderingDevice::UniformType;
    uniforms.clear();
    uniforms.push_back(make_uniform(RenderingDevice::UNIFORM_TYPE_ACCELERATION_STRUCTURE, 0, tlas));
    uniforms.push_back(make_uniform(RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER, 1, positions.buffer));
    uniforms.push_back(make_uniform(RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER, 2, indices.buffer));
    uniforms.push_back(make_uniform(RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER, 3, attributes.buffer));
    uniforms.push_back(make_uniform(RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER, 4, surfaces.buffer));
    uniforms.push_back(make_uniform(RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER, 5, instance_buffer));
    uniforms.push_back(make_uniform(RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER, 6, instance_material_buffer));
    uniforms.push_back(make_uniform(RenderingDevice::UNIFORM_TYPE_STORAGE_BUFFER, 7, material_buffer));
    uniforms.push_back(make_uniform(U::UNIFORM_TYPE_SAMPLER, 8, sampler));
    godot::Ref<godot::RDUniform> textures = make_uniform(U::UNIFORM_TYPE_TEXTURE, 9, white_texture);
    for (uint32_t slot = 1; slot < kMaxTextures; ++slot) {
        const bool used = slot < texture_slots.size() && texture_slots[slot].rs_texture != 0;
        textures->add_id(used ? texture_slots[slot].rd_texture : white_texture);
    }
    uniforms.push_back(textures);
    _free_uniform_sets();
    uniforms_dirty = false;
}

void LuminaSceneMirror::_free_uniform_sets() {
    for (auto &[shader_id, set] : uniform_sets) {
        if (set.is_valid() && rd->uniform_set_is_valid(set)) {
            rd->free_rid(set);
        }
    }
    uniform_sets.clear();
}

} // namespace prism
