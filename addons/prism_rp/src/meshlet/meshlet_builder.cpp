#include "prism_rp/meshlet/meshlet_builder.h"

#include <meshoptimizer.h>

#include <utility>

namespace prism::meshlet {

MeshletData build(const float *p_positions, size_t p_vertex_count, const uint32_t *p_indices, size_t p_index_count,
        const Settings &p_settings) {
    MeshletData out;
    if (p_positions == nullptr || p_indices == nullptr || p_index_count == 0 || p_index_count % 3 != 0 || p_vertex_count == 0) {
        return out;
    }
    for (size_t i = 0; i < p_index_count; ++i) {
        if (p_indices[i] >= p_vertex_count) {
            return out;
        }
    }

    std::vector<uint32_t> indices(p_indices, p_indices + p_index_count);
    if (p_settings.clockwise_front_faces) {
        for (size_t i = 0; i < indices.size(); i += 3) {
            std::swap(indices[i + 1], indices[i + 2]);
        }
    }

    const size_t max_meshlets = meshopt_buildMeshletsBound(indices.size(), p_settings.max_vertices, p_settings.max_triangles);
    std::vector<meshopt_Meshlet> raw(max_meshlets);
    std::vector<unsigned int> meshlet_vertices(indices.size());
    std::vector<unsigned char> meshlet_triangles(indices.size());

    const size_t count = meshopt_buildMeshlets(raw.data(), meshlet_vertices.data(), meshlet_triangles.data(),
        indices.data(), indices.size(), p_positions, p_vertex_count, sizeof(float) * 3,
        p_settings.max_vertices, p_settings.max_triangles, p_settings.cone_weight);
    raw.resize(count);

    out.meshlets.reserve(count);
    for (const meshopt_Meshlet &m : raw) {
        // Reorder within the meshlet for vertex reuse / rasterizer locality.
        meshopt_optimizeMeshlet(&meshlet_vertices[m.vertex_offset], &meshlet_triangles[m.triangle_offset], m.triangle_count, m.vertex_count);
        const meshopt_Bounds b = meshopt_computeMeshletBounds(&meshlet_vertices[m.vertex_offset], &meshlet_triangles[m.triangle_offset],
            m.triangle_count, p_positions, p_vertex_count, sizeof(float) * 3);

        Meshlet ml;
        ml.vertex_offset = static_cast<uint32_t>(out.vertices.size());
        ml.triangle_offset = static_cast<uint32_t>(out.triangles.size());
        ml.vertex_count = m.vertex_count;
        ml.triangle_count = m.triangle_count;
        for (int k = 0; k < 3; ++k) {
            ml.center[k] = b.center[k];
            ml.cone_apex[k] = b.cone_apex[k];
            ml.cone_axis[k] = b.cone_axis[k];
        }
        ml.radius = b.radius;
        ml.cone_cutoff = b.cone_cutoff;

        out.vertices.insert(out.vertices.end(), meshlet_vertices.begin() + m.vertex_offset,
            meshlet_vertices.begin() + m.vertex_offset + m.vertex_count);
        for (uint32_t t = 0; t < m.triangle_count; ++t) {
            const unsigned char *tri = &meshlet_triangles[m.triangle_offset + t * 3];
            uint32_t a = tri[0], b1 = tri[1], c = tri[2];
            if (p_settings.clockwise_front_faces) {
                std::swap(b1, c);  // restore the caller's winding
            }
            out.triangles.push_back(a | (b1 << 8) | (c << 16));
        }
        out.meshlets.push_back(ml);
    }
    return out;
}

} // namespace prism::meshlet
