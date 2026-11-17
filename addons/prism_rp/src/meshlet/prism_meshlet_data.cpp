#include "prism_rp/meshlet/prism_meshlet_data.h"
#include "prism_rp/meshlet/meshlet_builder.h"

#include <cstring>
#include <vector>

#include <godot_cpp/classes/array_mesh.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

namespace prism {

namespace {

struct alignas(16) GpuMeshlet {
    uint32_t vertex_offset, triangle_offset, vertex_count, triangle_count;
    float center[3], radius;
    float cone_apex[3], _pad;
    float cone_axis[3], cone_cutoff;
};
static_assert(sizeof(GpuMeshlet) == PrismMeshletData::kMeshletStride);

} // namespace

void PrismMeshletData::_bind_methods() {
    using godot::ClassDB;
    using godot::D_METHOD;
    using godot::PropertyInfo;
    using godot::Variant;
    ClassDB::bind_method(D_METHOD("set_meshlets", "data"), &PrismMeshletData::set_meshlets);
    ClassDB::bind_method(D_METHOD("get_meshlets"), &PrismMeshletData::get_meshlets);
    ClassDB::bind_method(D_METHOD("set_vertices", "data"), &PrismMeshletData::set_vertices);
    ClassDB::bind_method(D_METHOD("get_vertices"), &PrismMeshletData::get_vertices);
    ClassDB::bind_method(D_METHOD("set_triangles", "data"), &PrismMeshletData::set_triangles);
    ClassDB::bind_method(D_METHOD("get_triangles"), &PrismMeshletData::get_triangles);
    ClassDB::bind_method(D_METHOD("set_source_vertex_count", "count"), &PrismMeshletData::set_source_vertex_count);
    ClassDB::bind_method(D_METHOD("get_source_vertex_count"), &PrismMeshletData::get_source_vertex_count);
    ClassDB::bind_method(D_METHOD("get_meshlet_count"), &PrismMeshletData::get_meshlet_count);
    ClassDB::bind_method(D_METHOD("get_triangle_count"), &PrismMeshletData::get_triangle_count);
    ClassDB::bind_method(D_METHOD("get_meshlet", "index"), &PrismMeshletData::get_meshlet);
    ADD_PROPERTY(PropertyInfo(Variant::PACKED_BYTE_ARRAY, "meshlets"), "set_meshlets", "get_meshlets");
    ADD_PROPERTY(PropertyInfo(Variant::PACKED_INT32_ARRAY, "vertices"), "set_vertices", "get_vertices");
    ADD_PROPERTY(PropertyInfo(Variant::PACKED_INT32_ARRAY, "triangles"), "set_triangles", "get_triangles");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "source_vertex_count"), "set_source_vertex_count", "get_source_vertex_count");
}

void PrismMeshletData::set_meshlets(const godot::PackedByteArray &p_data) { meshlets = p_data; }
godot::PackedByteArray PrismMeshletData::get_meshlets() const { return meshlets; }
void PrismMeshletData::set_vertices(const godot::PackedInt32Array &p_data) { vertices = p_data; }
godot::PackedInt32Array PrismMeshletData::get_vertices() const { return vertices; }
void PrismMeshletData::set_triangles(const godot::PackedInt32Array &p_data) { triangles = p_data; }
godot::PackedInt32Array PrismMeshletData::get_triangles() const { return triangles; }
void PrismMeshletData::set_source_vertex_count(int32_t p_count) { source_vertex_count = p_count; }
int32_t PrismMeshletData::get_source_vertex_count() const { return source_vertex_count; }

int32_t PrismMeshletData::get_meshlet_count() const {
    return static_cast<int32_t>(meshlets.size() / kMeshletStride);
}

int32_t PrismMeshletData::get_triangle_count() const {
    return static_cast<int32_t>(triangles.size());
}

godot::Dictionary PrismMeshletData::get_meshlet(int32_t p_index) const {
    godot::Dictionary d;
    if (p_index < 0 || p_index >= get_meshlet_count()) {
        return d;
    }
    GpuMeshlet m;
    std::memcpy(&m, meshlets.ptr() + static_cast<int64_t>(p_index) * kMeshletStride, sizeof(m));
    d["vertex_offset"] = static_cast<int64_t>(m.vertex_offset);
    d["triangle_offset"] = static_cast<int64_t>(m.triangle_offset);
    d["vertex_count"] = static_cast<int64_t>(m.vertex_count);
    d["triangle_count"] = static_cast<int64_t>(m.triangle_count);
    d["center"] = godot::Vector3(m.center[0], m.center[1], m.center[2]);
    d["radius"] = m.radius;
    d["cone_apex"] = godot::Vector3(m.cone_apex[0], m.cone_apex[1], m.cone_apex[2]);
    d["cone_axis"] = godot::Vector3(m.cone_axis[0], m.cone_axis[1], m.cone_axis[2]);
    d["cone_cutoff"] = m.cone_cutoff;
    return d;
}

void PrismMeshletBuilder::_bind_methods() {
    godot::ClassDB::bind_static_method("PrismMeshletBuilder",
        godot::D_METHOD("build_surface", "mesh", "surface", "max_vertices", "max_triangles", "cone_weight"),
        &PrismMeshletBuilder::build_surface, DEFVAL(64), DEFVAL(124), DEFVAL(0.25f));
}

godot::Ref<PrismMeshletData> PrismMeshletBuilder::build_surface(const godot::Ref<godot::Mesh> &p_mesh, int32_t p_surface,
        int32_t p_max_vertices, int32_t p_max_triangles, float p_cone_weight) {
    godot::Ref<PrismMeshletData> result;
    if (p_mesh.is_null() || p_surface < 0 || p_surface >= p_mesh->get_surface_count()) {
        godot::UtilityFunctions::push_error("[Prism RP] build_surface: invalid mesh or surface index.");
        return result;
    }
    if (p_max_vertices < 3 || p_max_vertices > 255 || p_max_triangles < 1 || p_max_triangles > 512) {
        godot::UtilityFunctions::push_error("[Prism RP] build_surface: max_vertices must be 3-255 and max_triangles 1-512.");
        return result;
    }
    if (const godot::ArrayMesh *array_mesh = godot::Object::cast_to<godot::ArrayMesh>(p_mesh.ptr()); array_mesh != nullptr) {
        if (array_mesh->surface_get_primitive_type(p_surface) != godot::Mesh::PRIMITIVE_TRIANGLES) {
            godot::UtilityFunctions::push_error("[Prism RP] build_surface: only triangle surfaces can be split into meshlets.");
            return result;
        }
    }

    const godot::Array arrays = p_mesh->surface_get_arrays(p_surface);
    const godot::PackedVector3Array positions = arrays[godot::Mesh::ARRAY_VERTEX];
    const godot::PackedInt32Array index_array = arrays[godot::Mesh::ARRAY_INDEX];

    std::vector<float> pos(static_cast<size_t>(positions.size()) * 3);
    for (int64_t i = 0; i < positions.size(); ++i) {
        const godot::Vector3 v = positions[i];
        pos[3 * i + 0] = static_cast<float>(v.x);
        pos[3 * i + 1] = static_cast<float>(v.y);
        pos[3 * i + 2] = static_cast<float>(v.z);
    }
    std::vector<uint32_t> idx;
    if (index_array.is_empty()) {   // non-indexed surface: every 3 vertices form a triangle
        idx.resize(static_cast<size_t>(positions.size()));
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = static_cast<uint32_t>(i);
    } else {
        idx.resize(static_cast<size_t>(index_array.size()));
        for (int64_t i = 0; i < index_array.size(); ++i) idx[static_cast<size_t>(i)] = static_cast<uint32_t>(index_array[i]);
    }

    meshlet::Settings settings;
    settings.max_vertices = static_cast<uint32_t>(p_max_vertices);
    settings.max_triangles = static_cast<uint32_t>(p_max_triangles);
    settings.cone_weight = p_cone_weight;
    settings.clockwise_front_faces = true;   // Godot
    const meshlet::MeshletData built = meshlet::build(pos.data(), pos.size() / 3, idx.data(), idx.size(), settings);
    if (built.meshlets.empty()) {
        godot::UtilityFunctions::push_error("[Prism RP] build_surface: surface has no valid triangles.");
        return result;
    }

    godot::PackedByteArray meshlet_bytes;
    meshlet_bytes.resize(static_cast<int64_t>(built.meshlets.size()) * PrismMeshletData::kMeshletStride);
    for (size_t i = 0; i < built.meshlets.size(); ++i) {
        const meshlet::Meshlet &m = built.meshlets[i];
        GpuMeshlet g{};
        g.vertex_offset = m.vertex_offset;
        g.triangle_offset = m.triangle_offset;
        g.vertex_count = m.vertex_count;
        g.triangle_count = m.triangle_count;
        std::memcpy(g.center, m.center, sizeof(g.center));
        g.radius = m.radius;
        std::memcpy(g.cone_apex, m.cone_apex, sizeof(g.cone_apex));
        std::memcpy(g.cone_axis, m.cone_axis, sizeof(g.cone_axis));
        g.cone_cutoff = m.cone_cutoff;
        std::memcpy(meshlet_bytes.ptrw() + static_cast<int64_t>(i) * PrismMeshletData::kMeshletStride, &g, sizeof(g));
    }
    godot::PackedInt32Array verts, tris;
    verts.resize(static_cast<int64_t>(built.vertices.size()));
    for (size_t i = 0; i < built.vertices.size(); ++i) verts.set(static_cast<int64_t>(i), static_cast<int32_t>(built.vertices[i]));
    tris.resize(static_cast<int64_t>(built.triangles.size()));
    for (size_t i = 0; i < built.triangles.size(); ++i) tris.set(static_cast<int64_t>(i), static_cast<int32_t>(built.triangles[i]));

    result.instantiate();
    result->set_meshlets(meshlet_bytes);
    result->set_vertices(verts);
    result->set_triangles(tris);
    result->set_source_vertex_count(static_cast<int32_t>(positions.size()));
    return result;
}

} // namespace prism
