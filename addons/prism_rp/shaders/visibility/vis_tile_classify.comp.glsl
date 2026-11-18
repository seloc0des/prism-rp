#version 460
#extension GL_GOOGLE_include_directive : require

#include "vis_shading_common.glsl"

// ============================================================================
// Prism RP — 8x8 Tile Material Classification Pass
// Dispatched with workgroups = (ceil(width / 8), ceil(height / 8), 1).
// Inspects the 64 pixels in each 8x8 tile of the R32G32_UINT Visibility Buffer,
// deduplicates active material bins in shared memory, and appends compact
// (tile_coord, material_bin) work items into MaterialWorkBuffer.
//
// The shade pass runs (PRISM_SHADE_DISPATCH_WIDTH, rows, 1) workgroups through
// vkCmdDispatchIndirect on ShadeDispatch; this pass grows `rows` as work items arrive.
// Host, each frame: ShadeDispatch = {PRISM_SHADE_DISPATCH_WIDTH, 0, 1, 0}.
// ============================================================================

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std140) uniform ShadeUniformBlock {
    VisibilityShadeUBO u_shade;
};

// 64-bit visibility buffer (R32G32_UINT), cleared to PRISM_VIS_PAYLOAD_EMPTY
layout(set = 0, binding = 1) uniform usampler2D u_vis_buffer;

layout(set = 0, binding = 2, std430) readonly buffer InstanceBuffer {
    GpuInstance b_instances[];
};

layout(set = 0, binding = 3, std430) writeonly buffer MaterialWorkBuffer {
    MaterialTileWorkItem b_work_items[];
};

layout(set = 0, binding = 4, std430) buffer ShadeDispatchBuffer {
    ShadeDispatch b_dispatch;
};

// 256-bit material presence bitmask for this 8x8 tile (8 x 32-bit words)
shared uint s_material_mask[8];
shared uint s_tile_base_slot;
shared uint s_tile_unique_count;

void main() {
    uint local_tid = gl_LocalInvocationIndex; // 0..63
    ivec2 pixel    = ivec2(gl_GlobalInvocationID.xy);

    if (local_tid < 8u) {
        s_material_mask[local_tid] = 0u;
    }
    if (local_tid == 0u) {
        s_tile_unique_count = 0u;
    }
    barrier();

    // 1. Each thread reads its pixel's payload and sets its material bin's bit
    if (pixel.x < int(u_shade.viewport_and_limits.x) &&
        pixel.y < int(u_shade.viewport_and_limits.y)) {
        uvec2 payload = texelFetch(u_vis_buffer, pixel, 0).rg;
        if (!vis_payload_is_empty(payload)) {
            uint mat_id = b_instances[payload.x].material_id;
            uint bin    = min(mat_id, PRISM_MAX_TILE_MATERIALS - 1u);
            atomicOr(s_material_mask[bin >> 5u], 1u << (bin & 31u));
        }
    }
    barrier();

    // 2. Count unique bins in this tile (first 8 threads sum popcounts)
    uint word_val = (local_tid < 8u) ? s_material_mask[local_tid] : 0u;
    if (local_tid < 8u && word_val != 0u) {
        atomicAdd(s_tile_unique_count, uint(bitCount(word_val)));
    }
    barrier();

    if (s_tile_unique_count == 0u) {
        return; // background-only tile: no shading work
    }

    // 3. Reserve a contiguous range of work items once per tile and grow the dispatch to cover it
    if (local_tid == 0u) {
        uint base = atomicAdd(b_dispatch.work_item_count, s_tile_unique_count);
        s_tile_base_slot = base;
        uint stored_end = min(base + s_tile_unique_count, u_shade.viewport_and_limits.w);
        if (stored_end > base) {
            atomicMax(b_dispatch.group_count_y, (stored_end + PRISM_SHADE_DISPATCH_WIDTH - 1u) / PRISM_SHADE_DISPATCH_WIDTH);
        }
    }
    barrier();

    // 4. First 8 threads write one work item per set bit of their word
    if (local_tid < 8u && word_val != 0u) {
        uint prefix = 0u;
        for (uint w = 0u; w < local_tid; ++w) {
            prefix += uint(bitCount(s_material_mask[w]));
        }

        uint packed_tile = (gl_WorkGroupID.x & 0xFFFFu) | ((gl_WorkGroupID.y & 0xFFFFu) << 16u);
        uint bits = word_val;
        uint local_idx = 0u;
        while (bits != 0u) {
            uint bit_pos = uint(findLSB(bits));
            bits &= bits - 1u;

            uint global_slot = s_tile_base_slot + prefix + local_idx;
            if (global_slot < u_shade.viewport_and_limits.w) {
                b_work_items[global_slot] = MaterialTileWorkItem(packed_tile, (local_tid << 5u) | bit_pos);
            }
            local_idx++;
        }
    }
}
