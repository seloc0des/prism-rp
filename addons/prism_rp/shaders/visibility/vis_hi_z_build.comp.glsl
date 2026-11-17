#version 460

// ============================================================================
// Prism RP — Reverse-Z Hi-Z Pyramid Builder
// Generates R32_SFLOAT depth pyramid using conservative MIN reduction
// (Reverse-Z: 1.0 = near, 0.0 = far -> min() preserves farthest depth in footprint)
// ============================================================================

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

const uint HIZ_MODE_COPY_DEPTH    = 0u;
const uint HIZ_MODE_DOWNSAMPLE_MIN = 1u;

layout(set = 0, binding = 0) uniform sampler2D u_src_depth;
layout(set = 0, binding = 1, r32f) uniform writeonly image2D u_dst_mip;

layout(push_constant) uniform HiZPushConstants {
    ivec2 src_size;
    ivec2 dst_size;
    uint  mode;      // 0 = copy full-res D32_SFLOAT to mip 0, 1 = 2x2 min downsample
    uint  src_mip;   // Source mip level (0 for copy, N-1 for downsample)
} pc;

void main() {
    ivec2 dst_coord = ivec2(gl_GlobalInvocationID.xy);
    if (dst_coord.x >= pc.dst_size.x || dst_coord.y >= pc.dst_size.y) {
        return;
    }

    if (pc.mode == HIZ_MODE_COPY_DEPTH) {
        ivec2 src_coord = min(dst_coord, pc.src_size - ivec2(1));
        float d = texelFetch(u_src_depth, src_coord, 0).r;
        imageStore(u_dst_mip, dst_coord, vec4(d, 0.0, 0.0, 0.0));
        return;
    }

    // 2x2 conservative min downsample with odd-dimension edge tap support
    ivec2 base = dst_coord * 2;
    ivec2 max_coord = pc.src_size - ivec2(1);

    ivec2 c00 = min(base + ivec2(0, 0), max_coord);
    ivec2 c10 = min(base + ivec2(1, 0), max_coord);
    ivec2 c01 = min(base + ivec2(0, 1), max_coord);
    ivec2 c11 = min(base + ivec2(1, 1), max_coord);

    int mip = int(pc.src_mip);
    float d00 = texelFetch(u_src_depth, c00, mip).r;
    float d10 = texelFetch(u_src_depth, c10, mip).r;
    float d01 = texelFetch(u_src_depth, c01, mip).r;
    float d11 = texelFetch(u_src_depth, c11, mip).r;

    float min_z = min(min(d00, d10), min(d01, d11));

    // When source dimension is odd, include the 3rd boundary texel so edge pixels
    // cannot leak false occlusion on the right/bottom screen border.
    bool extra_x = ((pc.src_size.x & 1) != 0) && (dst_coord.x == pc.dst_size.x - 1);
    bool extra_y = ((pc.src_size.y & 1) != 0) && (dst_coord.y == pc.dst_size.y - 1);

    if (extra_x) {
        ivec2 c20 = min(base + ivec2(2, 0), max_coord);
        ivec2 c21 = min(base + ivec2(2, 1), max_coord);
        min_z = min(min_z, min(texelFetch(u_src_depth, c20, mip).r,
                               texelFetch(u_src_depth, c21, mip).r));
    }
    if (extra_y) {
        ivec2 c02 = min(base + ivec2(0, 2), max_coord);
        ivec2 c12 = min(base + ivec2(1, 2), max_coord);
        min_z = min(min_z, min(texelFetch(u_src_depth, c02, mip).r,
                               texelFetch(u_src_depth, c12, mip).r));
    }
    if (extra_x && extra_y) {
        ivec2 c22 = min(base + ivec2(2, 2), max_coord);
        min_z = min(min_z, texelFetch(u_src_depth, c22, mip).r);
    }

    imageStore(u_dst_mip, dst_coord, vec4(min_z, 0.0, 0.0, 0.0));
}
