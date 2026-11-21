#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <godot_cpp/classes/object.hpp>
#include <godot_cpp/classes/rd_acceleration_structure_instance.hpp>
#include <godot_cpp/classes/rd_uniform.hpp>
#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/transform3d.hpp>
#include <godot_cpp/variant/typed_array.hpp>

namespace godot {
class Material;
class Mesh;
class MeshInstance3D;
class Node;
class World3D;
} // namespace godot

namespace prism {

// Lumina's copy of the scene for ray queries. Needs SUPPORTS_RAY_QUERY, so the custom Godot build.
//
// Main thread, at frame_pre_draw: follows the scene tree's MeshInstance3D nodes, reads each new mesh's triangles
// once, and gathers every instance's transform, mask and materials. Render thread, queued ahead of that frame's
// draw: uploads new meshes into pooled buffers, builds one BLAS per mesh (one geometry per surface), rebuilds the
// TLAS and rewrites the instance and material tables. Lumina's passes bind the result as one uniform set, laid out
// as lumina_scene_common.glsl declares it.
//
// The mirror idles until a pass asks for the scene (scene_uniform_set()) and releases everything once nothing has
// asked for kIdleFramesBeforeRelease frames, so stock Godot and scenes without ray-query passes pay nothing.
//
// Not mirrored yet: skinned and blend-shape meshes, MultiMesh, CSG, particles, ImmediateMesh and other Mesh types
// that are neither ArrayMesh nor PrimitiveMesh, and surfaces with alpha-blended materials (in the TLAS, but never
// hit). Only the main viewport's World3D (the edited scene's in the editor).
class LuminaSceneMirror : public godot::Object {
    GDCLASS(LuminaSceneMirror, godot::Object)

public:
    // Shared with lumina_scene_common.glsl.
    static constexpr uint32_t kMaxTextures = 1024;
    static constexpr uint32_t kMaskCamera = 0x01;   // seen by the camera
    static constexpr uint32_t kMaskShadow = 0x02;   // casts shadows
    static constexpr uint32_t kMaterialAlphaTest = 1u;
    static constexpr uint32_t kMaterialSkip = 2u;    // alpha-blended: rays pass through
    static constexpr uint32_t kMaterialVertexColor = 4u;
    static constexpr uint32_t kMaterialVertexColorSrgb = 8u;

    static LuminaSceneMirror *get_singleton();
    static void set_singleton(LuminaSceneMirror *p_mirror);

    void connect_frame_hook();
    void disconnect_frame_hook();

    // Render thread. Asks for the scene this frame and returns its uniform set for p_shader, as set 1 in the layout
    // lumina_scene_common.glsl declares, or an invalid RID until the first frame's scene is on the GPU. The mirror
    // owns the set and frees it whenever its resources change.
    godot::RID scene_uniform_set(const godot::RID &p_shader);

protected:
    static void _bind_methods() {}

private:
    static constexpr uint32_t kIdleFramesBeforeRelease = 120;
    static constexpr uint32_t kUnusedFramesBeforeDrop = 120;
    static constexpr uint32_t kAttributeWords = 4;   // uv.x, uv.y, rgba8 colour, octahedral snorm16x2 normal

    // Matches LuminaMaterial in lumina_scene_common.glsl.
    struct MaterialRecord {
        float albedo[4] = { 1.0f, 1.0f, 1.0f, 1.0f };        // linear rgb, alpha
        float uv_transform[4] = { 1.0f, 1.0f, 0.0f, 0.0f };  // uv1 scale.xy, offset.xy
        uint32_t albedo_texture = 0;                         // slot in the texture array, 0 = none
        uint32_t flags = 0;
        float alpha_scissor = 0.5f;
        uint32_t pad = 0;
    };
    static_assert(sizeof(MaterialRecord) == 48);

    // Matches LuminaSurface.
    struct SurfaceRecord {
        uint32_t corner_base = 0;     // first triangle corner in the position and index pools
        uint32_t vertex_base = 0;     // first vertex in the attribute pool
        uint32_t triangle_count = 0;
        uint32_t material_slot = 0;   // the mesh surface this geometry came from
    };
    static_assert(sizeof(SurfaceRecord) == 16);

    // Matches LuminaInstance.
    struct InstanceRecord {
        uint32_t surface_base = 0;
        uint32_t material_base = 0;
        uint32_t pad[2] = { 0, 0 };
    };
    static_assert(sizeof(InstanceRecord) == 16);

    // One triangle surface of a mesh, read on the main thread.
    struct SurfaceGeometry {
        uint32_t material_slot = 0;
        std::vector<float> corners;        // xyz per triangle corner, model space (Godot winding)
        std::vector<uint32_t> indices;     // per corner, into this surface's vertices
        std::vector<uint32_t> attributes;  // kAttributeWords per vertex
    };

    struct MeshUpload {
        uint32_t mesh_id = 0;
        std::vector<SurfaceGeometry> surfaces;
    };

    struct FrameMaterial {
        MaterialRecord record;
        godot::RID albedo_texture;   // RenderingServer texture; its RD texture is looked up on the render thread
    };

    struct FrameInstance {
        uint32_t mesh_id = 0;
        godot::Transform3D transform;
        uint32_t material_base = 0;
        uint32_t mask = 0;
        uint32_t flags = 0;          // RenderingDevice::AccelerationStructureInstanceFlagBits
    };

    struct FrameState {
        std::vector<FrameInstance> instances;
        std::vector<uint32_t> instance_materials;   // per instance, one per mesh surface
        std::vector<FrameMaterial> materials;
    };

    // Main thread -> render thread, under pending_mutex. Uploads and removals accumulate until the render thread
    // takes them; the frame state is replaced each frame.
    struct Pending {
        std::vector<MeshUpload> uploads;
        std::vector<uint32_t> removals;
        FrameState frame;
        bool has_frame = false;
        bool release = false;
    };

    // Main thread.
    struct MainMesh {
        uint32_t mesh_id = 0;          // 0: not on the GPU
        uint32_t last_used_frame = 0;
        uint32_t surface_count = 0;
        bool mirrorable = true;        // false: no triangles, or skinned / blend shapes (re-checked when it changes)
        bool dirty = false;            // its "changed" signal fired: read it again
    };

    // Render thread.
    struct GeometryRange {
        uint32_t corner_offset = 0;   // within the mesh's corner range
        uint32_t vertex_offset = 0;   // within the mesh's vertex range
        uint32_t triangle_count = 0;
        uint32_t material_slot = 0;
    };

    struct TextureSlot {
        int64_t rs_texture = 0;   // RenderingServer texture id; 0: the slot is free
        godot::RID rd_texture;
        uint32_t last_used_frame = 0;
    };

    struct RenderMesh {
        godot::RID blas;
        uint32_t corner_base = 0;
        uint32_t corner_count = 0;
        uint32_t vertex_base = 0;
        uint32_t vertex_count = 0;
        uint32_t surface_base = 0;
        std::vector<GeometryRange> geometries;
    };

    // A growable GPU buffer that meshes are appended to. Removed meshes leave holes until the next repack.
    struct Pool {
        godot::RID buffer;
        uint32_t capacity = 0;   // elements
        uint32_t used = 0;
    };

    void _log(const godot::String &p_message) const;
    void _on_frame_pre_draw();
    void _on_node_added(godot::Node *p_node);
    void _on_node_removed(godot::Node *p_node);
    void _on_mesh_changed(uint64_t p_mesh_object_id);
    void _start_tracking();
    void _stop_tracking();
    void _gather_frame();
    void _gather_instance(godot::MeshInstance3D *p_instance, godot::World3D *p_world, FrameState &r_frame,
            std::unordered_map<uint64_t, uint32_t> &r_material_seen);
    [[nodiscard]] godot::World3D *_mirrored_world() const;
    const MainMesh *_main_mesh_for(const godot::Ref<godot::Mesh> &p_mesh);
    void _drop_unused_meshes();
    static uint32_t _material_index(const godot::Ref<godot::Material> &p_material, FrameState &r_frame,
            std::unordered_map<uint64_t, uint32_t> &r_seen);
    [[nodiscard]] static bool _read_mesh(const godot::Ref<godot::Mesh> &p_mesh, MeshUpload &r_upload);

    void _apply_on_render_thread();
    void _release_on_render_thread();
    bool _ensure_render_resources();
    void _remove_mesh(uint32_t p_mesh_id);
    void _upload_mesh(MeshUpload &p_upload);
    bool _ensure_pool_space(uint32_t p_corners, uint32_t p_vertices, uint32_t p_surfaces);
    bool _repack(uint32_t p_corner_capacity, uint32_t p_vertex_capacity, uint32_t p_surface_capacity);
    bool _build_blas(RenderMesh &r_mesh);
    void _write_surface_records(const RenderMesh &p_mesh);
    bool _write_frame(const FrameState &p_frame);
    uint32_t _texture_slot(const godot::RID &p_texture);
    void _age_texture_slots();
    bool _ensure_buffer(godot::RID &r_buffer, uint32_t &r_capacity, uint32_t p_needed, uint32_t p_element_size);
    bool _build_tlas(const FrameState &p_frame);
    void _rebuild_uniforms();
    void _free_uniform_sets();

    static LuminaSceneMirror *singleton;

    // Main thread.
    bool frame_hook_connected = false;
    bool log_lifecycle = false;   // the user argument --prism-mirror-log prints starts and releases; else verbose only
    bool tracking = false;
    uint32_t frame_index = 0;
    uint32_t idle_frames = kIdleFramesBeforeRelease;   // idle until the first request
    uint32_t last_seen_request = 0;
    uint32_t next_mesh_id = 1;
    std::unordered_set<uint64_t> tracked_nodes;                // MeshInstance3D object ids
    std::unordered_map<uint64_t, MainMesh> main_meshes;        // by Mesh object id
    std::vector<MeshUpload> gathered_uploads;                  // this frame's, handed over at the end of _gather_frame
    std::vector<uint32_t> gathered_removals;

    std::mutex pending_mutex;
    Pending pending;

    std::atomic<uint32_t> request_serial{ 0 };   // bumped by scene_uniform_set() on the render thread

    // Render thread.
    godot::RenderingDevice *rd = nullptr;
    godot::RID sampler;
    godot::RID white_texture;
    Pool positions;      // vertex buffer, float xyz per corner (BLAS input)
    Pool indices;        // storage buffer, uint per corner
    Pool attributes;     // storage buffer, kAttributeWords per vertex
    Pool surfaces;       // storage buffer, SurfaceRecord
    std::unordered_map<uint32_t, RenderMesh> render_meshes;
    godot::RID tlas;
    uint32_t tlas_capacity = 0;
    godot::RID instance_buffer;
    uint32_t instance_capacity = 0;
    godot::RID instance_material_buffer;
    uint32_t instance_material_capacity = 0;
    godot::RID material_buffer;
    uint32_t material_capacity = 0;
    // Slots stay put while their texture is in use, so the set of textures, and with it the uniform set, changes
    // only when textures come and go. Slot 0 is white and never reused.
    std::vector<TextureSlot> texture_slots;
    std::unordered_map<int64_t, uint32_t> slot_by_texture;     // RenderingServer texture id -> slot
    std::vector<uint32_t> free_texture_slots;
    uint32_t render_frame = 0;
    std::vector<godot::Ref<godot::RDAccelerationStructureInstance>> tlas_instance_pool;
    godot::TypedArray<godot::RDUniform> uniforms;
    std::unordered_map<int64_t, godot::RID> uniform_sets;      // by shader RID id
    bool uniforms_dirty = true;
    bool scene_ready = false;
};

} // namespace prism
