#version 460

// Radiance cascade merge, run from the top level down. For each (probe, direction) of level i:
//   merged = interval_i.rgb + interval_i.a * parent
// where `parent` is level i+1 (already merged), averaged over the 2x2 child directions this
// direction owns and bilinearly interpolated over the 4 nearest parent probes.

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D u_parent_tex;    // level i+1, merged
layout(rgba16f, set = 0, binding = 1) uniform image2D u_cascade_io;  // level i: interval in, merged out

layout(push_constant, std430) uniform PushConstants {
    ivec2 probe_count;
    ivec2 parent_probe_count;
    int block;
    int parent_block;          // = 2 * block
    float probe_spacing_px;
    float parent_spacing_px;
} pc;

vec4 parent_direction_average(ivec2 parent_probe, ivec2 dir_texel) {
    ivec2 base = parent_probe * pc.parent_block + dir_texel * 2;
    return 0.25 * (
        texelFetch(u_parent_tex, base, 0) +
        texelFetch(u_parent_tex, base + ivec2(1, 0), 0) +
        texelFetch(u_parent_tex, base + ivec2(0, 1), 0) +
        texelFetch(u_parent_tex, base + ivec2(1, 1), 0));
}

void main() {
    ivec2 texel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 probe = texel / pc.block;
    ivec2 dir_texel = texel - probe * pc.block;
    if (probe.x >= pc.probe_count.x || probe.y >= pc.probe_count.y) {
        return;
    }

    vec4 interval = imageLoad(u_cascade_io, texel);

    // Probe center in parent-probe coordinates.
    vec2 center_px = (vec2(probe) + 0.5) * pc.probe_spacing_px;
    vec2 parent_coord = center_px / pc.parent_spacing_px - 0.5;
    ivec2 p0 = ivec2(floor(parent_coord));
    vec2 f = parent_coord - vec2(p0);
    ivec2 max_probe = pc.parent_probe_count - ivec2(1);

    vec4 parent =
        parent_direction_average(clamp(p0,               ivec2(0), max_probe), dir_texel) * (1.0 - f.x) * (1.0 - f.y) +
        parent_direction_average(clamp(p0 + ivec2(1, 0), ivec2(0), max_probe), dir_texel) * f.x * (1.0 - f.y) +
        parent_direction_average(clamp(p0 + ivec2(0, 1), ivec2(0), max_probe), dir_texel) * (1.0 - f.x) * f.y +
        parent_direction_average(clamp(p0 + ivec2(1, 1), ivec2(0), max_probe), dir_texel) * f.x * f.y;

    vec3 radiance = interval.rgb + interval.a * parent.rgb;
    float transmittance = interval.a * parent.a;
    imageStore(u_cascade_io, texel, vec4(radiance, transmittance));
}
