#version 460
#extension GL_GOOGLE_include_directive : require

#include "prism_common.glsl"
#include "lumina_common.glsl"

// Mip 0 of the radiance mip chain every cascade level traces against: the colour buffer's radiance (rgb)
// and the pixel's linear view depth (a). Sky pixels get no radiance: a ray that reaches the sky finds no
// light in Lumina, so the background colour must not blur into hits beside it. The coarser mips keep the
// nearest depth they cover (lumina_radiance_downsample.comp.glsl), which tells the trace when a blurred
// texel takes in something in front of the surface its ray hit (lumina_rc_trace.comp.glsl).
//
// Godot's colour buffer holds direct light only, so on its own every level traces one bounce. Each pixel
// also sends out last frame's GI times its albedo (scaled by the composite's intensity), the light the viewer
// saw it reflect: what the cascades trace then bounces once more every frame. On the custom build the albedo is
// the scene mirror's, traced through each pixel (lumina_rt_albedo.comp.glsl); elsewhere, and where that ray
// missed, it is the composite's proxy, the pixel's hue at a mid-grey reflectance. Last
// frame's GI is reprojected with the motion vectors and used only on the same surface, as SVGF's temporal
// pass decides it (lumina_svgf_temporal.comp.glsl). bounce_gain 0 turns this off (single-bounce GI).
// The proxy gives a saturated colour an albedo of 1 in its channel, which nothing real has; bounced again
// and again that runs away in a saturated room, so the bounce caps it at kMaxBounceAlbedo. Above an
// intensity of 1 the cap applies to albedo x intensity, or a closed room gains light every frame without end.

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D u_color_tex;
layout(set = 0, binding = 1) uniform sampler2D u_depth_tex;
layout(rgba16f, set = 0, binding = 2) uniform writeonly image2D u_dst;
layout(set = 0, binding = 3) uniform sampler2D u_prev_gi_tex;         // last frame's denoised GI (rgb)
layout(set = 0, binding = 4) uniform sampler2D u_velocity_tex;
layout(set = 0, binding = 5) uniform sampler2D u_prev_geometry_tex;   // last frame's view normal, linear depth
layout(set = 0, binding = 6) uniform sampler2D u_normal_roughness_tex;
layout(set = 0, binding = 7) uniform sampler2D u_scene_albedo_tex;    // the scene mirror's albedo, a = 1 where a ray hit

layout(push_constant, std430) uniform PushConstants {
    ivec2 size;
    vec2 depth_params;
    float bounce_gain;   // the composite's intensity; 0: no previous GI to bounce
    float scene_albedo;  // 1: u_scene_albedo_tex holds this frame's albedo (lumina_rt_albedo.comp.glsl)
    vec2 _pad1;
} pc;

const float kMaxBounceAlbedo = 0.8;
const float kSkyDepth = 60000.0;   // beyond any scene depth and still inside half-float range: never nearer

void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(p, pc.size))) {
        return;
    }
    float depth = texelFetch(u_depth_tex, p, 0).r;
    if (lumina_is_sky(depth)) {
        imageStore(u_dst, p, vec4(0.0, 0.0, 0.0, kSkyDepth));
        return;
    }
    vec3 radiance = max(texelFetch(u_color_tex, p, 0).rgb, vec3(0.0));
    float z = prism_linearize_depth(depth, pc.depth_params);
    if (pc.bounce_gain > 0.0) {
        // Godot velocity = prev_uv - curr_uv.
        vec2 prev_uv = (vec2(p) + 0.5) / vec2(pc.size) + texelFetch(u_velocity_tex, p, 0).rg;
        if (all(greaterThanEqual(prev_uv, vec2(0.0))) && all(lessThanEqual(prev_uv, vec2(1.0)))) {
            vec4 prev_geo = textureLod(u_prev_geometry_tex, prev_uv, 0.0);
            vec3 n = lumina_decode_normal(texelFetch(u_normal_roughness_tex, p, 0));
            if (prev_geo.w > 0.0 && abs(prev_geo.w - z) / z < 0.1 && dot(normalize(prev_geo.xyz), n) > 0.9) {
                vec3 gi = max(textureLod(u_prev_gi_tex, prev_uv, 0.0).rgb, vec3(0.0));
                vec4 traced = pc.scene_albedo > 0.0 ? texelFetch(u_scene_albedo_tex, p, 0) : vec4(0.0);
                vec3 guess = traced.a > 0.5 ? traced.rgb : lumina_albedo_proxy(radiance);
                vec3 albedo = min(guess, vec3(kMaxBounceAlbedo / max(pc.bounce_gain, 1.0)));
                radiance += albedo * gi * pc.bounce_gain;
            }
        }
    }
    imageStore(u_dst, p, vec4(radiance, z));
}
