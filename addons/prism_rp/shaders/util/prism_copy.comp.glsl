#version 460

// Texel-exact copy through a sampler. Godot 4.4 doesn't give the scene color buffer
// TEXTURE_USAGE_CAN_COPY_FROM_BIT, so RenderingDevice.texture_copy() can't read it; it is
// sampleable, so this does the copy instead. Sampling a per-view layer view also reads the right
// XR eye (texture_copy addresses layers of the underlying image and ignores the view's layer).

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D u_src;
layout(rgba16f, set = 0, binding = 1) uniform writeonly image2D u_dst;

layout(push_constant, std430) uniform PushConstants {
    ivec2 size;
    vec2 _pad;
} pc;

void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x >= pc.size.x || p.y >= pc.size.y) {
        return;
    }
    imageStore(u_dst, p, texelFetch(u_src, p, 0));
}
