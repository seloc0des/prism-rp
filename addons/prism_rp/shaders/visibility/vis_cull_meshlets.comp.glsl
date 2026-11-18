#version 460
#extension GL_GOOGLE_include_directive : require

#include "visibility_common.glsl"

// ============================================================================
// Prism RP — Two-Phase GPU Meshlet Culler
//
// Pass 0 (PRISM_CULL_PASS_PHASE1_MAIN):
//   - Evaluates candidate (instance_id, meshlet_local_id) items.
//   - Frustum + backface normal-cone culling, then a Hi-Z test of where the meshlet was LAST frame
//     against last frame's Hi-Z ("was it hidden last frame?").
//   - Visible meshlets -> visible list + draw commands. Hi-Z rejects -> false-negative queue for
//     Phase 2 (drawn right away if the queue is full, so nothing disappears).
//
// Pass 1 (PRISM_CULL_PASS_PHASE2_DISOCCLUSION):
//   - Re-tests the false-negative queue against THIS frame's Hi-Z, rebuilt after Phase 1 drew.
//   - Newly disoccluded meshlets -> visible list + draw commands.
//
// Output lists (opaque / masked, see PRISM_VIS_LIST_*): entry i of list L lives at
// L * max_visible_meshlets + i in b_visible_meshlets and b_draw_commands.
//
// Host, before each phase: counters[0] = counters[2] = 0 and both mesh-task commands = {0, 1, 1};
// before Phase 1 only: counters[1] = 0. Draw a list with vkCmdDrawIndirectCount (count =
// min(counter, max_visible_meshlets)) or vkCmdDrawMeshTasksIndirectEXT on its mesh-task command.
// ============================================================================

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std140) uniform CullUniformBlock {
    VisibilityCullUBO u_cull;
};

layout(set = 0, binding = 1, std430) readonly buffer MeshletBuffer {
    GpuMeshlet b_meshlets[];
};

layout(set = 0, binding = 2, std430) readonly buffer InstanceBuffer {
    GpuInstance b_instances[];
};

// Phase 1 candidate list (packed uvec2(instance_id, meshlet_local_id))
layout(set = 0, binding = 3, std430) readonly buffer CandidateBuffer {
    uvec2 b_candidates[];
};

// False-negative queue written in Phase 1, read in Phase 2
layout(set = 0, binding = 4, std430) buffer FalseNegativeBuffer {
    FalseNegativeItem b_false_negatives[];
};

// Visible meshlets (2 * max_visible_meshlets entries: opaque list, then masked list)
layout(set = 0, binding = 5, std430) writeonly buffer VisibleMeshletBuffer {
    VisibleMeshlet b_visible_meshlets[];
};

// Non-indexed draw per visible meshlet for the vertex-pulling path (same layout as the lists):
// vertexCount = 3 * triangles, firstInstance = the entry's index (vis_raster.vert reads it back).
layout(set = 0, binding = 6, std430) writeonly buffer DrawIndirectBuffer {
    VkDrawIndirectCommand b_draw_commands[];
};

// Atomic counters:
//   counters[0] = opaque visible count, counters[1] = false-negative count,
//   counters[2] = masked visible count. Counts can exceed their capacity; clamp when drawing.
layout(set = 0, binding = 7, std430) buffer CullCountersBuffer {
    uint b_counters[];
};

// Mesh-shader indirect commands, one per list (one workgroup per visible meshlet)
layout(set = 0, binding = 8, std430) buffer MeshTasksIndirectBuffer {
    VkDrawMeshTasksIndirectCommandEXT b_mesh_tasks_cmd[2];
};

// Phase 1: previous frame's Hi-Z pyramid. Phase 2: this frame's, rebuilt after Phase 1.
layout(set = 0, binding = 9) uniform sampler2D u_hiz_pyramid;

void emit_visible_meshlet(uint instance_id, uint meshlet_local_id, uint triangle_count, uint list) {
    uint slot = atomicAdd(b_counters[list == PRISM_VIS_LIST_MASKED ? 2u : 0u], 1u);
    if (slot >= u_cull.max_visible_meshlets) {
        return; // list full: size max_visible_meshlets for the worst case to never drop geometry
    }
    uint index = list * u_cull.max_visible_meshlets + slot;
    b_visible_meshlets[index] = VisibleMeshlet(instance_id, meshlet_local_id);
    b_draw_commands[index] = VkDrawIndirectCommand(triangle_count * 3u, 1u, 0u, index);
    atomicMax(b_mesh_tasks_cmd[list].groupCountX, slot + 1u);
}

float max_axis_scale(mat4 m) {
    return max(length(m[0].xyz), max(length(m[1].xyz), length(m[2].xyz)));
}

void main() {
    uint tid = gl_GlobalInvocationID.x;
    bool phase1 = u_cull.cull_pass_mode == PRISM_CULL_PASS_PHASE1_MAIN;

    uint instance_id;
    uint meshlet_local_id;
    if (phase1) {
        if (tid >= u_cull.total_candidate_count) {
            return;
        }
        uvec2 item       = b_candidates[tid];
        instance_id      = item.x;
        meshlet_local_id = item.y;
    } else {
        uint fn_count = min(b_counters[1], u_cull.max_false_negatives);
        if (tid >= fn_count) {
            return;
        }
        FalseNegativeItem fn = b_false_negatives[tid];
        instance_id          = fn.instance_id;
        meshlet_local_id     = fn.meshlet_local_id;
    }

    GpuInstance inst = b_instances[instance_id];
    if ((inst.flags & PRISM_INSTANCE_FLAG_VISIBLE) == 0u) {
        return;
    }
    if (meshlet_local_id >= inst.meshlet_count) {
        return;
    }

    GpuMeshlet m = b_meshlets[inst.meshlet_offset + meshlet_local_id];
    if (m.triangle_count == 0u) {
        return;
    }
    uint list = (inst.flags & PRISM_INSTANCE_FLAG_ALPHA_TEST) != 0u ? PRISM_VIS_LIST_MASKED : PRISM_VIS_LIST_OPAQUE;

    // Bounding sphere in world space (radius scaled by the largest axis: conservative)
    mat4 M = inst.world_from_local;
    vec3 world_center  = (M * vec4(m.bound_sphere.xyz, 1.0)).xyz;
    float world_radius = m.bound_sphere.w * max_axis_scale(M);

    // 1. Frustum and normal-cone culling (Phase 1 only; Phase 2 items already passed both)
    if (phase1) {
        if (u_cull.enable_frustum_cull != 0u &&
                !sphere_in_frustum(world_center, world_radius, u_cull.frustum_planes)) {
            return;
        }

        // Cone culling assumes the transform preserves angles between normals: uniform scale
        // (mirrors included). Under non-uniform scale the cone's cutoff no longer bounds the
        // normals, so skip the test rather than risk culling visible triangles.
        bool two_sided = (inst.flags & PRISM_INSTANCE_FLAG_TWO_SIDED) != 0u;
        vec3 scale = vec3(length(M[0].xyz), length(M[1].xyz), length(M[2].xyz));
        bool uniform_scale = max(scale.x, max(scale.y, scale.z)) <= 1.001 * min(scale.x, min(scale.y, scale.z));
        if (!two_sided && uniform_scale && u_cull.enable_cone_cull != 0u) {
            vec3 world_apex = (M * vec4(m.cone_apex.xyz, 1.0)).xyz;
            // For uniform scale, M maps normals like its inverse-transpose (up to scale), and keeps
            // outward normals outward even when it mirrors.
            vec3 world_axis = normalize(mat3(M) * m.cone_axis_cut.xyz);
            if (cone_backface_culled(world_apex, world_axis, m.cone_axis_cut.w, u_cull.camera_pos_and_near.xyz)) {
                return;
            }
        }
    }

    // 2. Reverse-Z Hi-Z occlusion culling
    if (u_cull.enable_hiz_cull != 0u) {
        vec4 uv_rect;
        float nearest_rev_z;
        bool testable;
        if (phase1) {
            // Where the meshlet was last frame, against last frame's depth
            mat4 prev_M = inst.prev_world_from_local;
            testable = project_sphere_reverse_z(
                (prev_M * vec4(m.bound_sphere.xyz, 1.0)).xyz,
                m.bound_sphere.w * max_axis_scale(prev_M),
                u_cull.prev_view_proj, u_cull.camera_pos_and_near.w, uv_rect, nearest_rev_z);
        } else {
            testable = project_sphere_reverse_z(
                world_center, world_radius,
                u_cull.view_proj, u_cull.camera_pos_and_near.w, uv_rect, nearest_rev_z);
        }

        if (testable && is_hiz_occluded(u_hiz_pyramid, u_cull.hiz_mip_count, uv_rect, nearest_rev_z)) {
            if (!phase1) {
                return; // hidden by this frame's depth
            }
            uint fn_slot = atomicAdd(b_counters[1], 1u);
            if (fn_slot < u_cull.max_false_negatives) {
                b_false_negatives[fn_slot] = FalseNegativeItem(instance_id, meshlet_local_id);
                return; // re-tested in Phase 2
            }
            // Queue full: draw it now instead of losing it this frame.
        }
    }

    emit_visible_meshlet(instance_id, meshlet_local_id, m.triangle_count, list);
}
