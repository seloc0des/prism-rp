#version 460

// One level of the prefiltered radiance pyramid the coarse cascade levels trace against: each texel is
// the average of the 2x2 source texels it covers (one bilinear tap at their shared corner).
//
// Level i's rays reach i doublings further per direction step while their directions halve in angle,
// so each sample stands for a patch about 2^i pixels across. Sampled at full resolution, a small
// bright patch (the hotspot a light makes on a nearby wall) is either hit at full strength or missed,
// and the coarse probes that hit it leave probe-sized blobs. Level i traces against pyramid level
// i - 1, whose texels are 2^i pixels across.

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D u_src;        // linear sampler: twice this size
layout(rgba16f, set = 0, binding = 1) uniform writeonly image2D u_dst;

layout(push_constant, std430) uniform PushConstants {
    ivec2 dst_size;
    ivec2 _pad;
} pc;

void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(p, pc.dst_size))) {
        return;
    }
    vec2 uv = (vec2(p) + 0.5) / vec2(pc.dst_size);
    imageStore(u_dst, p, vec4(max(textureLod(u_src, uv, 0.0).rgb, vec3(0.0)), 1.0));
}
