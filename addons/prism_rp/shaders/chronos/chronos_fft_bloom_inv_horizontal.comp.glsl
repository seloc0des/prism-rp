#version 460

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(rgba32f, set = 0, binding = 0) uniform readonly image2D u_fft_rg_in;
layout(rgba32f, set = 0, binding = 1) uniform readonly image2D u_fft_b_in;
layout(rgba16f, set = 0, binding = 2) uniform writeonly image2D u_bloom_spatial_out;

layout(push_constant, std430) uniform PushConstants {
    float bloom_intensity;
    float _pad0;
    float _pad1;
    float _pad2;
} pc;

shared vec2 s_r_a[256];
shared vec2 s_g_a[256];
shared vec2 s_b_a[256];

shared vec2 s_r_b[256];
shared vec2 s_g_b[256];
shared vec2 s_b_b[256];

vec2 cmul(vec2 a, vec2 b) {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

void main() {
    uint t = gl_LocalInvocationID.x;
    uint row = gl_WorkGroupID.y;

    uint x0 = t;
    uint x1 = t + 128u;

    vec4 rg0 = imageLoad(u_fft_rg_in, ivec2(int(x0), int(row)));
    vec4 b0  = imageLoad(u_fft_b_in,  ivec2(int(x0), int(row)));

    vec4 rg1 = imageLoad(u_fft_rg_in, ivec2(int(x1), int(row)));
    vec4 b1  = imageLoad(u_fft_b_in,  ivec2(int(x1), int(row)));

    s_r_a[x0] = rg0.xy;
    s_g_a[x0] = rg0.zw;
    s_b_a[x0] = b0.xy;

    s_r_a[x1] = rg1.xy;
    s_g_a[x1] = rg1.zw;
    s_b_a[x1] = b1.xy;

    barrier();

    for (uint stage = 0u; stage < 8u; ++stage) {
        uint m = 1u << stage;
        uint j = t & (m - 1u);
        uint k = t >> stage;

        uint idx0 = k * m + j;
        uint idx1 = idx0 + 128u;
        uint out0 = k * (2u * m) + j;
        uint out1 = out0 + m;

        float angle = 3.14159265359 * float(j) / float(m);
        vec2 w = vec2(cos(angle), sin(angle));

        if ((stage & 1u) == 0u) {
            vec2 tr = cmul(w, s_r_a[idx1]);
            vec2 tg = cmul(w, s_g_a[idx1]);
            vec2 tb = cmul(w, s_b_a[idx1]);

            s_r_b[out0] = s_r_a[idx0] + tr;
            s_r_b[out1] = s_r_a[idx0] - tr;
            s_g_b[out0] = s_g_a[idx0] + tg;
            s_g_b[out1] = s_g_a[idx0] - tg;
            s_b_b[out0] = s_b_a[idx0] + tb;
            s_b_b[out1] = s_b_a[idx0] - tb;
        } else {
            vec2 tr = cmul(w, s_r_b[idx1]);
            vec2 tg = cmul(w, s_g_b[idx1]);
            vec2 tb = cmul(w, s_b_b[idx1]);

            s_r_a[out0] = s_r_b[idx0] + tr;
            s_r_a[out1] = s_r_b[idx0] - tr;
            s_g_a[out0] = s_g_b[idx0] + tg;
            s_g_a[out1] = s_g_b[idx0] - tg;
            s_b_a[out0] = s_b_b[idx0] + tb;
            s_b_a[out1] = s_b_b[idx0] - tb;
        }

        barrier();
    }

    const float scale = (1.0 / 256.0) * pc.bloom_intensity;
    vec3 spatial0 = max(vec3(s_r_a[x0].x, s_g_a[x0].x, s_b_a[x0].x) * scale, vec3(0.0));
    vec3 spatial1 = max(vec3(s_r_a[x1].x, s_g_a[x1].x, s_b_a[x1].x) * scale, vec3(0.0));

    imageStore(u_bloom_spatial_out, ivec2(int(x0), int(row)), vec4(spatial0, 1.0));
    imageStore(u_bloom_spatial_out, ivec2(int(x1), int(row)), vec4(spatial1, 1.0));
}
