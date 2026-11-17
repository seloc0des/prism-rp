// Standalone tests for prism::meshlet::build (no Godot). Built and run by tests/unit/test_meshlet_builder.py.

#include "prism_rp/meshlet/meshlet_builder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <vector>

using prism::meshlet::MeshletData;
using prism::meshlet::Settings;

static int g_failures = 0;
#define CHECK(cond)                                                                        \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);  \
            ++g_failures;                                                                  \
        }                                                                                  \
    } while (0)

struct Mesh {
    std::vector<float> positions;
    std::vector<uint32_t> indices;
};

// n x n quads in the z = 0 plane. Triangles are clockwise seen from +Z (Godot's front face).
static Mesh grid(int n) {
    Mesh m;
    for (int y = 0; y <= n; ++y) {
        for (int x = 0; x <= n; ++x) {
            m.positions.insert(m.positions.end(), { float(x), float(y), 0.0f });
        }
    }
    auto id = [n](int x, int y) { return uint32_t(y * (n + 1) + x); };
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            // CW from +Z: (x,y) -> (x,y+1) -> (x+1,y)
            m.indices.insert(m.indices.end(), { id(x, y), id(x, y + 1), id(x + 1, y) });
            m.indices.insert(m.indices.end(), { id(x + 1, y), id(x, y + 1), id(x + 1, y + 1) });
        }
    }
    return m;
}

// UV sphere: closed surface, normals in every direction.
static Mesh sphere(int rings, int segments) {
    Mesh m;
    for (int r = 0; r <= rings; ++r) {
        const float phi = float(M_PI) * r / rings;
        for (int s = 0; s <= segments; ++s) {
            const float theta = 2.0f * float(M_PI) * s / segments;
            m.positions.insert(m.positions.end(), { std::sin(phi) * std::cos(theta), std::cos(phi), std::sin(phi) * std::sin(theta) });
        }
    }
    auto id = [segments](int r, int s) { return uint32_t(r * (segments + 1) + s); };
    for (int r = 0; r < rings; ++r) {
        for (int s = 0; s < segments; ++s) {
            m.indices.insert(m.indices.end(), { id(r, s), id(r + 1, s), id(r, s + 1) });
            m.indices.insert(m.indices.end(), { id(r, s + 1), id(r + 1, s), id(r + 1, s + 1) });
        }
    }
    return m;
}

static std::array<uint32_t, 3> canonical(uint32_t a, uint32_t b, uint32_t c) {
    // Rotate (keeping winding) so the smallest index comes first.
    std::array<uint32_t, 3> t{ a, b, c };
    while (t[0] != std::min({ a, b, c })) {
        t = { t[1], t[2], t[0] };
    }
    return t;
}

static void check_partition(const Mesh &mesh, const MeshletData &d, const Settings &s) {
    CHECK(!d.meshlets.empty());
    std::map<std::array<uint32_t, 3>, int> expected, got;
    for (size_t i = 0; i < mesh.indices.size(); i += 3) {
        ++expected[canonical(mesh.indices[i], mesh.indices[i + 1], mesh.indices[i + 2])];
    }
    for (const auto &m : d.meshlets) {
        CHECK(m.vertex_count >= 1 && m.vertex_count <= s.max_vertices);
        CHECK(m.triangle_count >= 1 && m.triangle_count <= s.max_triangles);
        CHECK(m.vertex_offset + m.vertex_count <= d.vertices.size());
        CHECK(m.triangle_offset + m.triangle_count <= d.triangles.size());
        for (uint32_t t = 0; t < m.triangle_count; ++t) {
            uint32_t a, b, c;
            prism::meshlet::unpack_triangle(d.triangles[m.triangle_offset + t], a, b, c);
            CHECK(a < m.vertex_count && b < m.vertex_count && c < m.vertex_count);
            ++got[canonical(d.vertices[m.vertex_offset + a], d.vertices[m.vertex_offset + b], d.vertices[m.vertex_offset + c])];
        }
        // Bounding sphere contains every vertex of the meshlet.
        for (uint32_t v = 0; v < m.vertex_count; ++v) {
            const float *p = &mesh.positions[3 * d.vertices[m.vertex_offset + v]];
            const float dx = p[0] - m.center[0], dy = p[1] - m.center[1], dz = p[2] - m.center[2];
            CHECK(std::sqrt(dx * dx + dy * dy + dz * dz) <= m.radius * 1.001f + 1e-4f);
        }
    }
    // Every input triangle exactly once, same winding.
    CHECK(got == expected);
}

int main() {
    const Settings settings;

    {   // Flat grid: 8192 triangles.
        const Mesh g = grid(64);
        const MeshletData d = prism::meshlet::build(g.positions.data(), g.positions.size() / 3, g.indices.data(), g.indices.size(), settings);
        check_partition(g, d, settings);
        // Godot CW winding: cones point at Godot's front face (+Z). For a flat meshlet the perspective
        // test (reject when dot(normalize(apex - eye), axis) >= cutoff) must keep an eye in front of
        // the plane and reject one behind it.
        auto rejects = [](const prism::meshlet::Meshlet &m, float ex, float ey, float ez) {
            float v[3] = { m.cone_apex[0] - ex, m.cone_apex[1] - ey, m.cone_apex[2] - ez };
            const float len = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            return (v[0] * m.cone_axis[0] + v[1] * m.cone_axis[1] + v[2] * m.cone_axis[2]) / len >= m.cone_cutoff;
        };
        for (const auto &m : d.meshlets) {
            CHECK(m.cone_axis[2] > 0.99f);
            CHECK(!rejects(m, 32.0f, 32.0f, 10.0f));
            CHECK(rejects(m, 32.0f, 32.0f, -10.0f));
        }
        // Reasonable packing: roughly full meshlets.
        CHECK(d.meshlets.size() <= g.indices.size() / 3 / 90);
        std::printf("grid: %zu triangles -> %zu meshlets\n", g.indices.size() / 3, d.meshlets.size());
    }

    {   // Same grid interpreted as CCW (non-Godot input): cones flip to -Z.
        const Mesh g = grid(16);
        Settings ccw = settings;
        ccw.clockwise_front_faces = false;
        const MeshletData d = prism::meshlet::build(g.positions.data(), g.positions.size() / 3, g.indices.data(), g.indices.size(), ccw);
        check_partition(g, d, ccw);
        for (const auto &m : d.meshlets) {
            CHECK(m.cone_axis[2] < -0.99f);
        }
    }

    {   // Sphere: backface culling sanity. A meshlet whose cone faces away from the eye must have
        // every triangle facing away (no false rejects).
        const Mesh sp = sphere(24, 48);
        const MeshletData d = prism::meshlet::build(sp.positions.data(), sp.positions.size() / 3, sp.indices.data(), sp.indices.size(), settings);
        check_partition(sp, d, settings);
        const float eye[3] = { 0.0f, 0.0f, 5.0f };
        int rejected = 0;
        for (const auto &m : d.meshlets) {
            float v[3] = { m.cone_apex[0] - eye[0], m.cone_apex[1] - eye[1], m.cone_apex[2] - eye[2] };
            const float len = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            const bool reject = (v[0] * m.cone_axis[0] + v[1] * m.cone_axis[1] + v[2] * m.cone_axis[2]) / len >= m.cone_cutoff;
            if (!reject) continue;
            ++rejected;
            for (uint32_t t = 0; t < m.triangle_count; ++t) {
                uint32_t a, b, c;
                prism::meshlet::unpack_triangle(d.triangles[m.triangle_offset + t], a, b, c);
                const float *pa = &sp.positions[3 * d.vertices[m.vertex_offset + a]];
                const float *pb = &sp.positions[3 * d.vertices[m.vertex_offset + b]];
                const float *pc = &sp.positions[3 * d.vertices[m.vertex_offset + c]];
                // Godot front-face normal for CW winding: (c - a) x (b - a).
                const float e1[3] = { pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2] };
                const float e2[3] = { pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2] };
                const float n[3] = { e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0] };
                const float to_eye[3] = { eye[0] - pa[0], eye[1] - pa[1], eye[2] - pa[2] };
                CHECK(n[0] * to_eye[0] + n[1] * to_eye[1] + n[2] * to_eye[2] <= 1e-4f);
            }
        }
        CHECK(rejected > 0);   // the far side of the sphere really gets culled
        std::printf("sphere: %zu meshlets, %d backface-culled from +Z\n", d.meshlets.size(), rejected);
    }

    {   // Invalid input returns empty data instead of crashing.
        const float pos[9] = { 0, 0, 0, 1, 0, 0, 0, 1, 0 };
        const uint32_t bad_count[2] = { 0, 1 };
        const uint32_t bad_index[3] = { 0, 1, 7 };
        CHECK(prism::meshlet::build(pos, 3, bad_count, 2, settings).meshlets.empty());
        CHECK(prism::meshlet::build(pos, 3, bad_index, 3, settings).meshlets.empty());
        CHECK(prism::meshlet::build(pos, 3, nullptr, 0, settings).meshlets.empty());
    }

    if (g_failures) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all meshlet builder checks passed\n");
    return 0;
}
