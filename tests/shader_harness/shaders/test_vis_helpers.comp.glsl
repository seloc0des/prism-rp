#version 460
#extension GL_GOOGLE_include_directive : require

// Test-only entry point exposing the shared Track B GLSL helpers to tests/shader_harness.
// Each case reads 4 vec4s and writes 4 vec4s; `mode` picks the helper.

#include "vis_shading_common.glsl"

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform TestPushConstants {
    uint mode;
    uint count;
    uint hiz_mip_count;
    uint _pad;
} pc;

layout(set = 0, binding = 0, std430) readonly buffer Inputs { vec4 in_data[]; };
layout(set = 0, binding = 1, std430) writeonly buffer Outputs { vec4 out_data[]; };
layout(set = 0, binding = 2) uniform sampler2D u_hiz;

const uint MODE_PAYLOAD = 0u;      // a = uint bits (instance, meshlet, primitive, alpha)
const uint MODE_CULL = 1u;         // a.xy, a.zw, b.xy = NDC corners; b.z = two-sided, b.w = mirrored
const uint MODE_BARYCENTRIC = 2u;  // a, b, c = clip corners; d.xy = pixel NDC, d.zw = 1 / viewport
const uint MODE_NORMAL = 3u;       // a, b, c = columns of a 3x3 matrix (xyz); d.xyz = normal
const uint MODE_HIZ = 4u;          // a = UV rect; b.x = nearest reverse-Z depth

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.count) {
        return;
    }
    vec4 a = in_data[4u * i];
    vec4 b = in_data[4u * i + 1u];
    vec4 c = in_data[4u * i + 2u];
    vec4 d = in_data[4u * i + 3u];
    vec4 r0 = vec4(0.0), r1 = vec4(0.0), r2 = vec4(0.0), r3 = vec4(0.0);

    if (pc.mode == MODE_PAYLOAD) {
        uvec4 u = floatBitsToUint(a);
        uvec2 p = pack_visibility_payload(u.x, u.y, u.z, u.w != 0u);
        uint instance_id, meshlet_id, prim_id;
        bool alpha;
        unpack_visibility_payload(p, instance_id, meshlet_id, prim_id, alpha);
        r0 = uintBitsToFloat(uvec4(p, vis_payload_is_empty(p) ? 1u : 0u, vis_payload_is_empty(PRISM_VIS_PAYLOAD_EMPTY) ? 1u : 0u));
        r1 = uintBitsToFloat(uvec4(instance_id, meshlet_id, prim_id, alpha ? 1u : 0u));
    } else if (pc.mode == MODE_CULL) {
        r0.x = vis_triangle_culled(a.xy, a.zw, b.xy, b.z != 0.0, b.w != 0.0) ? 1.0 : 0.0;
        r0.y = vis_ndc_twice_area(a.xy, a.zw, b.xy);
    } else if (pc.mode == MODE_BARYCENTRIC) {
        vec3 bary, ddx_bary, ddy_bary;
        bool ok = compute_barycentric_gradients(a, b, c, d.xy, d.zw, bary, ddx_bary, ddy_bary);
        r0 = vec4(bary, ok ? 1.0 : 0.0);
        r1 = vec4(ddx_bary, 0.0);
        r2 = vec4(ddy_bary, 0.0);
    } else if (pc.mode == MODE_NORMAL) {
        mat3 m = mat3(a.xyz, b.xyz, c.xyz);
        r0 = vec4(normalize(prism_normal_matrix(m) * d.xyz), prism_det3(m));
    } else if (pc.mode == MODE_HIZ) {
        r0.x = is_hiz_occluded(u_hiz, pc.hiz_mip_count, a, b.x) ? 1.0 : 0.0;
    }

    out_data[4u * i] = r0;
    out_data[4u * i + 1u] = r1;
    out_data[4u * i + 2u] = r2;
    out_data[4u * i + 3u] = r3;
}
