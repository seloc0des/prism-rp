#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_fragment_shader_barycentric : require

#include "visibility_common.glsl"

// ============================================================================
// Prism RP — 64-Bit Visibility Buffer Fragment Shader (R32G32_UINT)
// Zero texture fetches or PBR math on opaque geometry: packs InstanceID,
// MeshletLocalID, PrimitiveID, and perspective-correct barycentrics into 64 bits.
// ============================================================================

layout(location = 0) flat in uvec4 in_prim_meta;

layout(location = 0) out uvec2 out_vis_payload;

// Enforce early depth-stencil testing on opaque geometry so occluded fragments
// are rejected in hardware before invoking fragment shader lanes.
layout(early_fragment_tests) in;

void main() {
    uint instance_id      = in_prim_meta.x;
    uint meshlet_local_id = in_prim_meta.y;
    uint primitive_id     = in_prim_meta.z;
    uint instance_flags   = in_prim_meta.w;

    bool alpha_tested = (instance_flags & PRISM_INSTANCE_FLAG_ALPHA_TEST) != 0u;

    // gl_BaryCoordEXT is perspective-correct (lambda0, lambda1, lambda2).
    // Store (lambda1, lambda2) at 12-bit UNORM; lambda0 is reconstructed as 1 - l1 - l2.
    out_vis_payload = pack_visibility_payload(
        instance_id,
        meshlet_local_id,
        primitive_id,
        gl_BaryCoordEXT.yz,
        alpha_tested
    );
}
