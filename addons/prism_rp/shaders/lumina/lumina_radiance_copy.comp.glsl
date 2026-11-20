#version 460
#extension GL_GOOGLE_include_directive : require

#include "prism_common.glsl"
#include "lumina_common.glsl"

// Mip 0 of the radiance mip chain every cascade level traces against: the colour buffer's radiance (rgb)
// and the pixel's linear view depth (a). Sky pixels get no radiance: a ray that reaches the sky finds no
// light in Lumina, so the background colour must not blur into hits beside it. The coarser mips keep the
// nearest depth they cover (lumina_radiance_downsample.comp.glsl), which tells the trace when a blurred
// texel takes in something in front of the surface its ray hit (lumina_rc_trace.comp.glsl).

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D u_color_tex;
layout(set = 0, binding = 1) uniform sampler2D u_depth_tex;
layout(rgba16f, set = 0, binding = 2) uniform writeonly image2D u_dst;

layout(push_constant, std430) uniform PushConstants {
    ivec2 size;
    vec2 depth_params;
} pc;

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
    imageStore(u_dst, p, vec4(radiance, prism_linearize_depth(depth, pc.depth_params)));
}
