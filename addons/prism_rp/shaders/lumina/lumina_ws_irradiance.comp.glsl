#version 460
#extension GL_GOOGLE_include_directive : require

#include "prism_common.glsl"
#include "lumina_common.glsl"
#include "lumina_world_common.glsl"

// One invocation per texel of the world cache's irradiance atlas, laid out as the radiance atlas but with each
// cell's 8 x 8 octahedral texels standing for surface normals. Each holds the cosine-weighted mean of the cell's cached
// radiance over the hemisphere around its normal: irradiance / pi, so a Lambertian surface of albedo a lit by it
// sends out a times this. lumina_ws_update_rt reads it where its rays hit, which is how light bounces more than once.
//
// Runs after lumina_ws_classify and before the update, on last frame's radiance. A slot that changed cells this frame
// reads as empty, since its radiance belongs to the cell that left.
// rgb = irradiance / pi; a = the share of the hemisphere's cosine weight that has samples (0: nothing known).

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D u_world_probes_tex;
layout(set = 0, binding = 1) uniform sampler2D u_state_tex;
layout(rgba16f, set = 0, binding = 2) uniform writeonly image2D u_world_irradiance;

void main() {
    int n = wpc.a0.z;
    ivec2 texel = ivec2(gl_GlobalInvocationID.xy);
    if (texel.x >= n * n * WORLD_BLOCK || texel.y >= WORLD_LEVELS * n * WORLD_BLOCK) {
        return;
    }
    ivec2 probe2d = texel / WORLD_BLOCK;
    ivec2 normal_texel = texel - probe2d * WORLD_BLOCK;
    ivec3 slot;
    int level;
    lumina_world_probe_of_texel(texel, n, slot, level);
    if (texelFetch(u_state_tex, lumina_state_texel(slot, level, n), 0).g > 0.5) {
        imageStore(u_world_irradiance, texel, vec4(0.0));
        return;
    }

    vec3 normal = lumina_block_direction(normal_texel, WORLD_BLOCK);
    ivec2 base = probe2d * WORLD_BLOCK;
    vec3 sum = vec3(0.0);
    float known = 0.0;
    float total = 0.0;
    for (int y = 0; y < WORLD_BLOCK; ++y) {
        for (int x = 0; x < WORLD_BLOCK; ++x) {
            float w = dot(normal, lumina_block_direction(ivec2(x, y), WORLD_BLOCK));
            if (w <= 0.0) {
                continue;
            }
            total += w;
            vec4 radiance = texelFetch(u_world_probes_tex, base + ivec2(x, y), 0);
            if (radiance.a > 0.0) {
                sum += radiance.rgb * w;
                known += w;
            }
        }
    }
    imageStore(u_world_irradiance, texel, known > 0.0 ? vec4(sum / known, known / total) : vec4(0.0));
}
