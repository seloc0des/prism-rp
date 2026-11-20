#version 460

// One mip of the radiance mip chain every cascade level traces against, made from the mip above it: rgb is
// the average of the source texels it covers, a the nearest linear depth among them. Mip 0 comes from
// lumina_radiance_copy.comp.glsl. The trace picks the mip by each hit's cone footprint and uses the nearest
// depth to step down when the texel takes in something in front of the hit (lumina_rc_trace.comp.glsl).

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D u_src;        // the mip above, as a view of that mip alone
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
    // Each texel covers 2 x 2 source texels. An odd source dimension leaves one row or column over, which the
    // last texel takes in (3 wide), as in vis_hi_z_build.comp.glsl, so nothing at the right or bottom is lost.
    ivec2 src_size = textureSize(u_src, 0);
    ivec2 span = ivec2(2) + ivec2(equal(p, pc.dst_size - ivec2(1))) * (src_size & ivec2(1));
    ivec2 last = src_size - ivec2(1);
    vec3 rgb = vec3(0.0);
    float nearest = 65504.0;   // the largest half float: no stored depth is larger
    for (int y = 0; y < span.y; ++y) {
        for (int x = 0; x < span.x; ++x) {
            vec4 s = texelFetch(u_src, min(2 * p + ivec2(x, y), last), 0);
            rgb += s.rgb;
            nearest = min(nearest, s.a);
        }
    }
    imageStore(u_dst, p, vec4(rgb / float(span.x * span.y), nearest));
}
