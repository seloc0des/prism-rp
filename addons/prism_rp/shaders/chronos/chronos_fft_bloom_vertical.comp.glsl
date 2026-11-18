#version 460

layout(local_size_x = 1, local_size_y = 128, local_size_z = 1) in;

layout(rgba32f, set = 0, binding = 0) uniform readonly image2D u_fft_rg_in;
layout(rgba32f, set = 0, binding = 1) uniform readonly image2D u_fft_b_in;
layout(rgba32f, set = 0, binding = 2) uniform writeonly image2D u_fft_rg_out;
layout(rgba32f, set = 0, binding = 3) uniform writeonly image2D u_fft_b_out;

layout(push_constant, std430) uniform PushConstants {
    float bloom_radius;
    float diffraction_strength;
    float halation_strength;
    float aperture_blades;
    float aspect_ratio;     // source width / height: the frame is squeezed into a square grid
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

// Diffraction spikes: each fades exponentially along its length (decay length in grid texels, times
// bloom_radius) and has a Gaussian cross-section (its spectrum's width along the strip, in normalized
// frequency, times 1 / bloom_radius).
const float kSpikeLength = 8.0;
const float kSpikeSpectrumWidth = 0.45;

vec3 evaluate_optical_otf(uint kx, uint ky) {
    float fx = (kx >= 128u) ? float(int(kx) - 256) : float(kx);
    float fy = (ky >= 128u) ? float(int(ky) - 256) : float(ky);

    // The full frame maps to a square grid, so grid x is stretched by the aspect ratio relative
    // to display pixels. Scaling fx by 1/aspect keeps the PSF round on screen.
    vec2 freq = vec2(fx / max(pc.aspect_ratio, 1e-3), fy) / 128.0;
    float r = length(freq);
    float radius = max(pc.bloom_radius, 0.25);

    float r_scaled = r * radius;
    float core_otf = 1.0 / pow(1.0 + 18.0 * r_scaled * r_scaled, 1.65);

    // An N-blade aperture gives N spikes for even N and 2N for odd N, i.e. spike_count / 2 streaks
    // through the PSF centre. A streak is a strip through the origin of frequency space, perpendicular
    // to it and of constant thickness: a Lorentzian across the strip (the exponential fade along the
    // streak) times a Gaussian along it (the cross-section). Both transforms are positive, so the
    // spikes add no negative lobes, carry exactly diffraction_strength of energy, and stay real and
    // point-symmetric. Spikes built as angular wedges instead get infinitely thin towards zero
    // frequency, i.e. as long as the grid: they crossed the whole frame and wrapped around its edges.
    float blades = max(round(pc.aperture_blades), 4.0);
    float spike_count = (mod(blades, 2.0) == 0.0) ? blades : 2.0 * blades;
    int streaks = int(spike_count * 0.5);
    float spikes = 0.0;
    for (int i = 0; i < streaks; ++i) {
        float angle = 6.28318530718 * float(i) / spike_count;
        vec2 dir = vec2(cos(angle), sin(angle));
        float along = dot(freq, dir) * radius / kSpikeSpectrumWidth;
        float across = dot(freq, vec2(-dir.y, dir.x)) * 3.14159265359 * kSpikeLength * radius;
        spikes += exp(-along * along) / (1.0 + across * across);
    }
    float diffraction = pc.diffraction_strength * spikes / float(streaks);

    float halation_r = pc.halation_strength * exp(-r_scaled * r_scaled * 14.0);
    float halation_g = pc.halation_strength * 0.14 * exp(-r_scaled * r_scaled * 28.0);
    float halation_b = pc.halation_strength * 0.02 * exp(-r_scaled * r_scaled * 42.0);

    vec3 otf = vec3(
        core_otf + diffraction + halation_r,
        core_otf + diffraction * 0.96 + halation_g,
        core_otf + diffraction * 0.92 + halation_b
    );

    // No DC (zero-frequency) suppression: scaling only the DC term subtracts a constant from every
    // pixel of the grid, which the inverse pass's max(0) then clips away almost everywhere. The bloom's
    // energy is OTF(0) = 1 + diffraction + halation per channel, times bloom_intensity.

    return otf;
}

void run_stockham_fft_vertical(float dir_sign) {
    uint t = gl_LocalInvocationID.y;
    for (uint stage = 0u; stage < 8u; ++stage) {
        uint m = 1u << stage;
        uint j = t & (m - 1u);
        uint k = t >> stage;

        uint idx0 = k * m + j;
        uint idx1 = idx0 + 128u;
        uint out0 = k * (2u * m) + j;
        uint out1 = out0 + m;

        float angle = dir_sign * 3.14159265359 * float(j) / float(m);
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
}

void main() {
    uint col = gl_WorkGroupID.x;
    uint t = gl_LocalInvocationID.y;

    uint y0 = t;
    uint y1 = t + 128u;

    vec4 rg0 = imageLoad(u_fft_rg_in, ivec2(int(col), int(y0)));
    vec4 b0  = imageLoad(u_fft_b_in,  ivec2(int(col), int(y0)));

    vec4 rg1 = imageLoad(u_fft_rg_in, ivec2(int(col), int(y1)));
    vec4 b1  = imageLoad(u_fft_b_in,  ivec2(int(col), int(y1)));

    s_r_a[y0] = rg0.xy;
    s_g_a[y0] = rg0.zw;
    s_b_a[y0] = b0.xy;

    s_r_a[y1] = rg1.xy;
    s_g_a[y1] = rg1.zw;
    s_b_a[y1] = b1.xy;

    barrier();

    run_stockham_fft_vertical(-1.0);

    vec3 otf0 = evaluate_optical_otf(col, y0);
    vec3 otf1 = evaluate_optical_otf(col, y1);

    s_r_a[y0] *= otf0.r;
    s_g_a[y0] *= otf0.g;
    s_b_a[y0] *= otf0.b;

    s_r_a[y1] *= otf1.r;
    s_g_a[y1] *= otf1.g;
    s_b_a[y1] *= otf1.b;

    barrier();

    run_stockham_fft_vertical(1.0);

    const float inv_n = 1.0 / 256.0;
    imageStore(u_fft_rg_out, ivec2(int(col), int(y0)), vec4(s_r_a[y0] * inv_n, s_g_a[y0] * inv_n));
    imageStore(u_fft_b_out,  ivec2(int(col), int(y0)), vec4(s_b_a[y0] * inv_n, 0.0, 0.0));

    imageStore(u_fft_rg_out, ivec2(int(col), int(y1)), vec4(s_r_a[y1] * inv_n, s_g_a[y1] * inv_n));
    imageStore(u_fft_b_out,  ivec2(int(col), int(y1)), vec4(s_b_a[y1] * inv_n, 0.0, 0.0));
}
