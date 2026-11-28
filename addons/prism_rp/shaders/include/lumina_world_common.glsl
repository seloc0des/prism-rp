#ifndef LUMINA_WORLD_COMMON_GLSL
#define LUMINA_WORLD_COMMON_GLSL

// Lumina world-space radiance cache: WORLD_LEVELS nested N^3 grids of probes centred on the camera, each holding a
// WORLD_BLOCK x WORLD_BLOCK octahedral radiance map. Level 0's probes are `spacing` world units apart; each level
// above doubles the spacing and so reaches twice as far. One 16^3 grid at 1 m covers 8 m around the camera, less
// than a courtyard: surfaces beyond it got no light from the cache at all.
//
// Addressing is toroidal per level: world cell c lives in slot (c mod N), so the camera moving only changes
// which cells are live (base_cell), never where existing cells are stored. Atlas layout:
//   width  = N * N * WORLD_BLOCK              (slot x + slot z * N, then direction x)
//   height = WORLD_LEVELS * N * WORLD_BLOCK   (slot y + level * N, then direction y)
// Cell ids (to detect a slot being reused by a new cell) live in a separate N*N x WORLD_LEVELS*N state texture.
//
// Every world pass shares one push-constant block (112 bytes, under the 128-byte minimum some
// drivers enforce). Field meanings differ per pass where noted.

#define WORLD_BLOCK 8
#define WORLD_LEVELS 2   // LuminaCascadesCompositorEffect::kWorldLevels

layout(push_constant, std430) uniform WorldPushConstants {
    ivec4 a0;       // xy: internal resolution, z: grid N, w: (merge) screen block
    ivec4 a1;       // xyz: base cell, w: (update) frame index / (merge) flags
    vec4 cam_pos;   // xyz: camera world position, w: probe spacing (world units)
    vec4 cam_right; // xyz: camera +X in world, w: (update) thickness
    vec4 cam_up;    // xyz: camera +Y in world, w: (update) blend alpha minimum
    vec4 proj;      // depth-corrected projection (P00, P11, P20, P21)
    vec4 misc;      // xy: depth params, z: (update) level 0's ray length, world units / (merge) screen probe spacing px,
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

float lumina_world_spacing(int level) {
    return wpc.cam_pos.w * float(1 << level);
}

// The level's lowest live cell. a1.xyz is level 0's, the camera's cell minus N/2; a level's camera cell is the one
// below's halved and rounded down (>> on a signed int keeps the sign), so this is floor(camera / spacing) - N/2 on
// every level.
ivec3 lumina_world_base(int level, int n) {
    return ((wpc.a1.xyz + n / 2) >> level) - n / 2;
}

// The slot and level of the probe that owns a texel of the radiance atlas (or an atlas laid out like it).
void lumina_world_probe_of_texel(ivec2 texel, int n, out ivec3 slot, out int level) {
    ivec2 probe2d = texel / WORLD_BLOCK;
    level = probe2d.y / n;
    slot = ivec3(probe2d.x % n, probe2d.y - level * n, probe2d.x / n);
}

// Exact in a 32-bit float (24 bits). Cells closer than 256 apart on an axis never alias.
float lumina_encode_cell(ivec3 c) {
    ivec3 m = lumina_positive_mod(c, 256);
    return float(m.x | (m.y << 8) | (m.z << 16));
}

ivec2 lumina_world_atlas_texel(ivec3 slot, int level, ivec2 dir, int n) {
    return ivec2((slot.x + slot.z * n) * WORLD_BLOCK + dir.x, (slot.y + level * n) * WORLD_BLOCK + dir.y);
}

ivec2 lumina_state_texel(ivec3 slot, int level, int n) {
    return ivec2(slot.x + slot.z * n, slot.y + level * n);
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

// One level of lumina_world_sample: trilinear over the level's 8 nearest cells at direction texel dir_texel. Cells
// behind the surface see another space (outside the room, inside a pillar) and don't apply; cells outside the
// level's live grid, buried probes (rays starting inside geometry) and texels never sampled (a <= 0) are missing.
// coverage: the share of the applicable weight that had a value.
vec3 lumina_world_sample_level(sampler2D atlas, sampler2D state_tex, vec3 p, vec3 normal, ivec2 dir_texel, int level,
                               out float coverage) {
    int n = wpc.a0.z;
    float spacing = lumina_world_spacing(level);
    ivec3 base = lumina_world_base(level, n);
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
        if (any(lessThan(cell, base)) || any(greaterThanEqual(cell, base + n))) {
            continue;
        }
        ivec3 slot = lumina_positive_mod(cell, n);
        if (texelFetch(state_tex, lumina_state_texel(slot, level, n), 0).b > 0.5) {
            continue;
        }
        vec4 value = texelFetch(atlas, lumina_world_atlas_texel(slot, level, dir_texel, n), 0);
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

// Level 0 hands over to level 1 across this many of its cells before the edge of its grid, so the seam between a
// fine and a coarse answer doesn't show as a line.
const float kWorldLevelBlendCells = 2.0;

// How much of p's answer comes from level 0: 1 well inside its grid, falling to 0 at the edge of the region where
// all 8 of its cells around p are live.
float lumina_world_level0_weight(vec3 p, int n) {
    vec3 coord = p / wpc.cam_pos.w - 0.5 - vec3(wpc.a1.xyz);   // 0 .. n - 1 where all 8 cells are live
    vec3 inside = min(coord, vec3(float(n - 1)) - coord);
    return clamp(min(inside.x, min(inside.y, inside.z)) / kWorldLevelBlendCells, 0.0, 1.0);
}

// Reads an atlas laid out as the radiance cache (the cache itself, or lumina_ws_irradiance's output) at world point p
// on a surface facing normal, in direction texel dir_texel: from level 0 near the camera, level 1 beyond it, and a
// blend of the two between. coverage: the share of the answer that had a value (lumina_world_sample_level).
vec3 lumina_world_sample(sampler2D atlas, sampler2D state_tex, vec3 p, vec3 normal, ivec2 dir_texel, out float coverage) {
    float w0 = lumina_world_level0_weight(p, wpc.a0.z);
    float c0 = 0.0;
    float c1 = 0.0;
    vec3 v0 = w0 > 0.0 ? lumina_world_sample_level(atlas, state_tex, p, normal, dir_texel, 0, c0) : vec3(0.0);
    vec3 v1 = w0 < 1.0 ? lumina_world_sample_level(atlas, state_tex, p, normal, dir_texel, 1, c1) : vec3(0.0);
    float k0 = w0 * c0;
    float k1 = (1.0 - w0) * c1;
    coverage = k0 + k1;
    return coverage > 0.0 ? (v0 * k0 + v1 * k1) / coverage : vec3(0.0);
}

#endif // LUMINA_WORLD_COMMON_GLSL
