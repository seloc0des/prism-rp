#ifndef LUMINA_WORLD_COMMON_GLSL
#define LUMINA_WORLD_COMMON_GLSL

// Lumina world-space radiance cache: an N^3 grid of probes centred on the camera, spaced
// `spacing` world units apart, each holding a WORLD_BLOCK x WORLD_BLOCK octahedral radiance map.
//
// Addressing is toroidal: world cell c lives in slot (c mod N), so the camera moving only changes
// which cells are live (base_cell), never where existing cells are stored. Atlas layout:
//   width  = N * N * WORLD_BLOCK   (slot x + slot z * N, then direction x)
//   height = N * WORLD_BLOCK       (slot y, then direction y)
// Cell ids (to detect a slot being reused by a new cell) live in a separate N*N x N state texture.
//
// Every world pass shares one push-constant block (112 bytes, under the 128-byte minimum some
// drivers enforce). Field meanings differ per pass where noted.

#define WORLD_BLOCK 8

layout(push_constant, std430) uniform WorldPushConstants {
    ivec4 a0;       // xy: internal resolution, z: grid N, w: (merge) screen block
    ivec4 a1;       // xyz: base cell, w: (update) frame index / (merge) flags
    vec4 cam_pos;   // xyz: camera world position, w: probe spacing (world units)
    vec4 cam_right; // xyz: camera +X in world, w: (update) thickness
    vec4 cam_up;    // xyz: camera +Y in world, w: (update) blend alpha minimum
    vec4 proj;      // depth-corrected projection (P00, P11, P20, P21)
    vec4 misc;      // xy: depth params, z: (update) ray length world / (merge) screen probe spacing px,
                    // w: (update) steps / (update_rt) 1 to bounce the cache's irradiance off hits, 0 for one bounce
} wpc;

vec3 lumina_cam_back() { return cross(wpc.cam_right.xyz, wpc.cam_up.xyz); }   // camera +Z (looks down -Z)

vec3 lumina_world_to_view(vec3 p) {
    vec3 d = p - wpc.cam_pos.xyz;
    return vec3(dot(d, wpc.cam_right.xyz), dot(d, wpc.cam_up.xyz), dot(d, lumina_cam_back()));
}

vec3 lumina_world_to_view_dir(vec3 d) {
    return vec3(dot(d, wpc.cam_right.xyz), dot(d, wpc.cam_up.xyz), dot(d, lumina_cam_back()));
}

vec3 lumina_view_to_world_dir(vec3 v) {
    return wpc.cam_right.xyz * v.x + wpc.cam_up.xyz * v.y + lumina_cam_back() * v.z;
}

vec3 lumina_view_to_world(vec3 v) {
    return wpc.cam_pos.xyz + lumina_view_to_world_dir(v);
}

ivec3 lumina_positive_mod(ivec3 a, int n) {
    return ((a % n) + n) % n;
}

// The live cell occupying slot s (base <= cell < base + N).
ivec3 lumina_cell_for_slot(ivec3 slot, ivec3 base, int n) {
    return base + lumina_positive_mod(slot - base, n);
}

vec3 lumina_cell_center(ivec3 cell, float spacing) {
    return (vec3(cell) + 0.5) * spacing;
}

// Exact in a 32-bit float (24 bits). Cells closer than 256 apart on an axis never alias.
float lumina_encode_cell(ivec3 c) {
    ivec3 m = lumina_positive_mod(c, 256);
    return float(m.x | (m.y << 8) | (m.z << 16));
}

ivec2 lumina_world_atlas_texel(ivec3 slot, ivec2 dir, int n) {
    return ivec2((slot.x + slot.z * n) * WORLD_BLOCK + dir.x, slot.y * WORLD_BLOCK + dir.y);
}

ivec2 lumina_state_texel(ivec3 slot, int n) {
    return ivec2(slot.x + slot.z * n, slot.y);
}

// Inverse of lumina_oct_decode.
vec2 lumina_oct_encode(vec3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    vec2 e = n.xy;
    if (n.z < 0.0) {
        e = (1.0 - abs(n.yx)) * lumina_sign_not_zero(n.xy);
    }
    return e;
}

ivec2 lumina_direction_texel(vec3 dir, int block) {
    vec2 e = lumina_oct_encode(normalize(dir)) * 0.5 + 0.5;
    return clamp(ivec2(e * float(block)), ivec2(0), ivec2(block - 1));
}

// Reads an atlas laid out as the radiance cache (the cache itself, or lumina_ws_irradiance's output) at world point p
// on a surface facing normal: trilinear over the 8 nearest cells at direction texel dir_texel. Cells behind the
// surface see another space (outside the room, inside a pillar) and don't apply; cells outside the live grid, buried
// probes (rays starting inside geometry) and texels never sampled (a <= 0) are missing. coverage: the share of the
// applicable weight that had a value.
vec3 lumina_world_sample(sampler2D atlas, sampler2D state_tex, vec3 p, vec3 normal, ivec2 dir_texel, out float coverage) {
    int n = wpc.a0.z;
    float spacing = wpc.cam_pos.w;
    vec3 coord = p / spacing - 0.5;
    ivec3 c0 = ivec3(floor(coord));
    vec3 f = coord - vec3(c0);

    vec3 sum = vec3(0.0);
    float weight = 0.0;
    float behind = 0.0;   // trilinear weight of cells behind the surface: not missing, just not applicable
    for (int i = 0; i < 8; ++i) {
        ivec3 o = ivec3(i & 1, (i >> 1) & 1, i >> 2);
        ivec3 cell = c0 + o;
        vec3 w3 = mix(vec3(1.0) - f, f, vec3(o));
        float w = w3.x * w3.y * w3.z;
        // The tolerance keeps cells lying on the surface's plane, where normal quantisation alone would decide the sign.
        if (dot(lumina_cell_center(cell, spacing) - p, normal) < -0.01 * spacing) {
            behind += w;
            continue;
        }
        if (any(lessThan(cell, wpc.a1.xyz)) || any(greaterThanEqual(cell, wpc.a1.xyz + n))) {
            continue;
        }
        ivec3 slot = lumina_positive_mod(cell, n);
        if (texelFetch(state_tex, lumina_state_texel(slot, n), 0).b > 0.5) {
            continue;
        }
        vec4 value = texelFetch(atlas, lumina_world_atlas_texel(slot, dir_texel, n), 0);
        if (value.a <= 0.0) {
            continue;
        }
        sum += value.rgb * w;
        weight += w;
    }
    float applicable = 1.0 - behind;
    if (weight <= 0.0 || weight <= 1e-4 * applicable) {
        coverage = 0.0;
        return vec3(0.0);
    }
    coverage = weight / applicable;
    return sum / weight;
}

#endif // LUMINA_WORLD_COMMON_GLSL
