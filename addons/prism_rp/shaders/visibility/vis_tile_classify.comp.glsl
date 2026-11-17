#version 460
#extension GL_GOOGLE_include_directive : require

#include "vis_shading_common.glsl"

// ============================================================================
// Prism RP — 8x8 Tile Material Classification Pass
// Dispatched with workgroups = (ceil(width / 8), ceil(height / 8), 1).
// Inspects the 64 pixels in each 8x8 tile of the R32G32_UINT Visibility Buffer,
// deduplicates active material_ids in shared memory, and appends compact
// (tile_coord, material_id) work items into MaterialWorkBuffer.
// ============================================================================

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std140) uniform ShadeUniformBlock {
    VisibilityShadeUBO u_shade;
};

// 64-bit Visibility Buffer (R32G32_UINT); uvec2(0, 0) indicates background/sky
layout(set = 0, binding = 1) uniform usampler2D u_vis_buffer;

layout(set = 0, binding = 2, std430) readonly buffer InstanceBuffer {
    GpuInstance b_instances[];
};

layout(set = 0, binding = 3, std430) writeonly buffer MaterialWorkBuffer {
    MaterialTileWorkItem b_work_items[];
};

// b_indirect_dispatch.x = total emitted (tile, material) workgroups; y = 1, z = 1
layout(set = 0, binding = 4, std430) buffer ShadeIndirectBuffer {
    VkDispatchIndirectCommand b_indirect_dispatch;
};

// 256-bit material presence bitmask for this 8x8 tile (8 x 32-bit words)
shared uint s_material_mask[8];
shared uint s_tile_base_slot;
shared uint s_tile_unique_count;

void main() {
    uint local_tid = gl_LocalInvocationIndex; // 0..63
    ivec2 pixel    = ivec2(gl_GlobalInvocationID.xy);

    // Clear shared bitmask
    if (local_tid < 8u) {
        s_material_mask[local_tid] = 0u;
    }
    if (local_tid == 0u) {
        s_tile_unique_count = 0u;
    }
    barrier();

    // 1. Each thread reads its pixel's visibility payload and sets its material bit
    if (pixel.x < int(u_shade.viewport_and_limits.x) &&
        pixel.y < int(u_shade.viewport_and_limits.y)) {
        uvec2 payload = texelFetch(u_vis_buffer, pixel, 0).rg;

        // Non-zero payload means valid rasterized geometry
        if ((payload.x | payload.y) != 0u) {
            uint instance_id = (payload.x >> 12u) & 0xFFFFFu;
            uint mat_id      = b_instances[instance_id].material_id;
            uint clamped_id  = min(mat_id, PRISM_MAX_TILE_MATERIALS - 1u);

            uint word_idx = clamped_id >> 5u;
            uint bit_mask = 1u << (clamped_id & 31u);
            atomicOr(s_material_mask[word_idx], bit_mask);
        }
    }

    barrier();

    // 2. Count unique materials in this tile (first 8 threads sum popcounts)
    uint word_val   = (local_tid < 8u) ? s_material_mask[local_tid] : 0u;
    uint word_count = bitCount(word_val);

    if (local_tid < 8u && word_count > 0u) {
        atomicAdd(s_tile_unique_count, word_count);
    }

    barrier();

    if (s_tile_unique_count == 0u) {
        return; // Pure sky/background tile -> zero shading work emitted
    }

    // 3. Reserve contiguous range in global MaterialWorkBuffer once per tile
    if (local_tid == 0u) {
        s_tile_base_slot = atomicAdd(b_indirect_dispatch.x, s_tile_unique_count);
        b_indirect_dispatch.y = 1u;
        b_indirect_dispatch.z = 1u;
    }

    barrier();

    // 4. First 8 threads iterate set bits in their 32-bit word and write work items
    if (local_tid < 8u && word_val != 0u) {
        // Compute prefix offset across the 8 mask words
        uint prefix = 0u;
        for (uint w = 0u; w < local_tid; ++w) {
            prefix += bitCount(s_material_mask[w]);
        }

        uint packed_tile = (uint(gl_WorkGroupID.x) & 0xFFFFu)
                         | ((uint(gl_WorkGroupID.y) & 0xFFFFu) << 16u);

        uint bits = word_val;
        uint local_idx = 0u;
        while (bits != 0u) {
            uint bit_pos = uint(findLSB(bits));
            bits &= (bits - 1u);

            uint global_slot = s_tile_base_slot + prefix + local_idx;
            if (global_slot < u_shade.viewport_and_limits.w) {
                MaterialTileWorkItem item;
                item.packed_tile_xy = packed_tile;
                item.material_id    = (local_tid << 5u) | bit_pos;
                b_work_items[global_slot] = item;
            }
            local_idx++;
        }
    }
}
