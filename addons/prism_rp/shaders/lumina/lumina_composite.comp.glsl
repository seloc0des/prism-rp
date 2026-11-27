#version 460
#extension GL_GOOGLE_include_directive : require

#include "prism_common.glsl"
#include "lumina_common.glsl"

// Adds Lumina's indirect diffuse to Godot's opaque color buffer (POST_OPAQUE), or shows a debug view.
//
// Forward+ has no albedo buffer and a compositor can't reach material lighting, so GI is applied as
// albedo_proxy * irradiance. The proxy keeps the pixel's chromaticity and assumes mid-grey (0.5)
// reflectance; dark pixels fall back to neutral grey. This is an approximation: Track B's G-buffer
// replaces it with real albedo.

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(rgba16f, set = 0, binding = 0) uniform image2D u_color_io;
layout(set = 0, binding = 1) uniform sampler2D u_gi_tex;          // denoised: rgb + variance
layout(set = 0, binding = 2) uniform sampler2D u_raw_tex;         // raw integrated irradiance
layout(set = 0, binding = 3) uniform sampler2D u_history_tex;     // a = history length
layout(set = 0, binding = 4) uniform sampler2D u_depth_tex;
layout(set = 0, binding = 5) uniform sampler2D u_normal_roughness_tex;

layout(push_constant, std430) uniform PushConstants {
    ivec2 internal_resolution;
    float intensity;
    int debug_view_mode;   // 0 final, 1 raw irradiance, 2 denoised, 3 history length, 4 variance, 5 normals, 6 albedo proxy
    int albedo_mode;       // 0 proxy, 1 white (lighting only)
    float max_history;
    vec2 _pad;
} pc;

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= pc.internal_resolution.x || pixel.y >= pc.internal_resolution.y) {
        return;
    }

    vec4 color = imageLoad(u_color_io, pixel);
    float depth = texelFetch(u_depth_tex, pixel, 0).r;
    vec4 gi = texelFetch(u_gi_tex, pixel, 0);

    if (pc.debug_view_mode == 1) {
        imageStore(u_color_io, pixel, vec4(texelFetch(u_raw_tex, pixel, 0).rgb, 1.0));
        return;
    } else if (pc.debug_view_mode == 2) {
        imageStore(u_color_io, pixel, vec4(gi.rgb, 1.0));
        return;
    } else if (pc.debug_view_mode == 3) {
        float h = clamp(texelFetch(u_history_tex, pixel, 0).a / max(pc.max_history, 1.0), 0.0, 1.0);
        imageStore(u_color_io, pixel, vec4(mix(vec3(0.9, 0.1, 0.05), vec3(0.1, 0.8, 0.3), h), 1.0));
        return;
    } else if (pc.debug_view_mode == 4) {
        float v = clamp(sqrt(gi.a) * 4.0, 0.0, 1.0);
        imageStore(u_color_io, pixel, vec4(vec3(v), 1.0));
        return;
    } else if (pc.debug_view_mode == 5) {
        vec3 n = lumina_decode_normal(texelFetch(u_normal_roughness_tex, pixel, 0));
        imageStore(u_color_io, pixel, vec4(n * 0.5 + 0.5, 1.0));
        return;
    } else if (pc.debug_view_mode == 6) {
        imageStore(u_color_io, pixel, vec4(lumina_albedo_proxy(color.rgb), 1.0));
        return;
    }

    if (lumina_is_sky(depth)) {
        return;
    }
    vec3 albedo = (pc.albedo_mode == 1) ? vec3(1.0) : lumina_albedo_proxy(color.rgb);
    imageStore(u_color_io, pixel, vec4(color.rgb + albedo * max(gi.rgb, vec3(0.0)) * pc.intensity, color.a));
}
