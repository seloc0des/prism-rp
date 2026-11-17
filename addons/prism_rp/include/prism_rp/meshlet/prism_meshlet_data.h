#pragma once

#include <godot_cpp/classes/mesh.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>

namespace prism {

// Meshlets of one mesh surface, laid out for direct upload to GPU storage buffers (Track B).
//   meshlets:  64 bytes each (std430-friendly):
//              uint vertex_offset, triangle_offset, vertex_count, triangle_count;
//              vec4 bounding sphere (center.xyz, radius);
//              vec4 cone apex (xyz, unused w); vec4 cone axis (xyz, cutoff)
//              Backface reject: dot(normalize(apex - eye), axis) >= cutoff (Godot front faces).
//   vertices:  meshlet-local vertex -> surface vertex index
//   triangles: per triangle, local indices packed a | b << 8 | c << 16 (Godot's winding preserved)
class PrismMeshletData : public godot::Resource {
    GDCLASS(PrismMeshletData, godot::Resource)

public:
    static constexpr int32_t kMeshletStride = 64;

    void set_meshlets(const godot::PackedByteArray &p_data);
    [[nodiscard]] godot::PackedByteArray get_meshlets() const;
    void set_vertices(const godot::PackedInt32Array &p_data);
    [[nodiscard]] godot::PackedInt32Array get_vertices() const;
    void set_triangles(const godot::PackedInt32Array &p_data);
    [[nodiscard]] godot::PackedInt32Array get_triangles() const;
    void set_source_vertex_count(int32_t p_count);
    [[nodiscard]] int32_t get_source_vertex_count() const;

    [[nodiscard]] int32_t get_meshlet_count() const;
    [[nodiscard]] int32_t get_triangle_count() const;
    // One meshlet as a Dictionary (vertex_offset, triangle_offset, vertex_count, triangle_count,
    // center, radius, cone_apex, cone_axis, cone_cutoff), for tools and tests.
    [[nodiscard]] godot::Dictionary get_meshlet(int32_t p_index) const;

protected:
    static void _bind_methods();

private:
    godot::PackedByteArray meshlets;
    godot::PackedInt32Array vertices;
    godot::PackedInt32Array triangles;
    int32_t source_vertex_count = 0;
};

// Builds PrismMeshletData from a Mesh surface (triangles only). Stateless.
class PrismMeshletBuilder : public godot::RefCounted {
    GDCLASS(PrismMeshletBuilder, godot::RefCounted)

public:
    static godot::Ref<PrismMeshletData> build_surface(const godot::Ref<godot::Mesh> &p_mesh, int32_t p_surface,
            int32_t p_max_vertices = 64, int32_t p_max_triangles = 124, float p_cone_weight = 0.25f);

protected:
    static void _bind_methods();
};

} // namespace prism
