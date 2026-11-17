#version 460
#extension GL_GOOGLE_include_directive : require

#include "prism_common.glsl"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D u_depth_tex;
layout(set = 0, binding = 1) uniform sampler2D u_velocity_tex;
layout(rg16f, set = 0, binding = 2) uniform writeonly image2D u_dilated_velocity_out;

layout(push_constant, std430) uniform PushConstants {
    ivec2 resolution;
    float near_plane;
    float _pad;
} pc;

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= pc.resolution.x || pixel.y >= pc.resolution.y) {
        return;
    }

    // 3x3 neighborhood closest-depth search (reverse-Z: largest depth value is nearest camera)
    float closest_depth = 0.0;
    ivec2 closest_offset = ivec2(0);

    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            ivec2 sample_coord = clamp(pixel + ivec2(x, y), ivec2(0), pc.resolution - ivec2(1));
            float d = texelFetch(u_depth_tex, sample_coord, 0).r;
            if (d > closest_depth) {
                closest_depth = d;
                closest_offset = ivec2(x, y);
            }
        }
    }

    vec2 dilated_velocity = texelFetch(u_velocity_tex, pixel + closest_offset, 0).rg;
    imageStore(u_dilated_velocity_out, pixel, vec4(dilated_velocity, 0.0, 0.0));
}
