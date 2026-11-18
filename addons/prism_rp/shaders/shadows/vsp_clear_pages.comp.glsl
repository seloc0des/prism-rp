#version 460
#extension GL_GOOGLE_include_directive : require

#include "vsp_common.glsl"

// ============================================================================
// Prism RP — Virtual Shadow Pages clear
// One workgroup per queued page: vkCmdDispatchIndirect on VspFrameState.clear_dispatch.
// Sets the page's 128x128 atlas texels to far depth (1.0) before vsp_raster_pages writes them.
// A separate pass so rasterization only ever does imageAtomicMin (no store/atomic race).
// ============================================================================

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std140) uniform VspUniformBlock {
    VspUniformUBO u_vsp;
};

layout(set = 0, binding = 1, std430) readonly buffer DirtyPageQueueBuffer {
    VspPageWorkItem b_dirty_pages[];
};

layout(set = 0, binding = 2, std430) readonly buffer VspFrameStateBuffer {
    VspFrameState b_state;
};

// Physical atlas, 4096x4096 R32_UINT
layout(set = 0, binding = 3, r32ui) uniform writeonly uimage2D u_physical_shadow_atlas;

void main() {
    uint page_idx = gl_WorkGroupID.x;
    if (page_idx >= min(b_state.dirty_page_count, u_vsp.viewport_and_counts.w)) {
        return;
    }

    ivec2 origin = vsp_physical_page_origin(b_dirty_pages[page_idx].physical_page & VSP_PTE_PAGE_MASK);
    uint far_depth = float_depth_to_ordered_uint(1.0);
    for (uint py = gl_LocalInvocationID.y; py < PRISM_VSP_PAGE_SIZE; py += 8u) {
        for (uint px = gl_LocalInvocationID.x; px < PRISM_VSP_PAGE_SIZE; px += 8u) {
            imageStore(u_physical_shadow_atlas, origin + ivec2(px, py), uvec4(far_depth, 0u, 0u, 0u));
        }
    }
}
