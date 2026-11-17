#pragma once

// Godot-independent meshlet builder (Phase 3 groundwork). Splits an indexed triangle mesh into
// meshlets of at most 64 vertices / 124 triangles (the roadmap's Track B budget, sized for
// VK_EXT_mesh_shader and NVIDIA's preferred 126-primitive limit) using meshoptimizer, with a
// bounding sphere and a backface normal cone per meshlet.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace prism::meshlet {

struct Settings {
    uint32_t max_vertices = 64;
    uint32_t max_triangles = 124;
    float cone_weight = 0.25f;   // 0 = pure locality, 1 = favour tight normal cones (better backface culling)
    // Godot treats clockwise triangles as front-facing; meshoptimizer's normal cones assume
    // counter-clockwise. With this on, winding is reversed going in and restored going out, so the
    // cones describe Godot's front faces and the micro index buffer keeps the input winding.
    bool clockwise_front_faces = true;
};

struct Meshlet {
    uint32_t vertex_offset = 0;     // into MeshletData::vertices
    uint32_t triangle_offset = 0;   // into MeshletData::triangles (one packed uint32 per triangle)
    uint32_t vertex_count = 0;
    uint32_t triangle_count = 0;
    float center[3] = {0, 0, 0};    // bounding sphere
    float radius = 0.0f;
    float cone_apex[3] = {0, 0, 0}; // backface cone: reject when dot(normalize(apex - eye), axis) >= cutoff
    float cone_axis[3] = {0, 0, 0};
    float cone_cutoff = 1.0f;       // cos(half angle); >= 1 means the cone is degenerate (never reject)
};

struct MeshletData {
    std::vector<Meshlet> meshlets;
    std::vector<uint32_t> vertices;   // meshlet-local vertex -> mesh vertex index
    std::vector<uint32_t> triangles;  // per triangle: local indices packed a | b << 8 | c << 16
};

// p_positions: tightly packed float3. Returns empty data for empty or invalid input
// (index count not a multiple of 3, or an index out of range).
MeshletData build(const float *p_positions, size_t p_vertex_count, const uint32_t *p_indices, size_t p_index_count,
        const Settings &p_settings = Settings());

inline void unpack_triangle(uint32_t p_packed, uint32_t &r_a, uint32_t &r_b, uint32_t &r_c) {
    r_a = p_packed & 0xffu;
    r_b = (p_packed >> 8) & 0xffu;
    r_c = (p_packed >> 16) & 0xffu;
}

} // namespace prism::meshlet
