#version 460
#extension GL_GOOGLE_include_directive : require

#include "visibility_common.glsl"

// ============================================================================
// Prism RP — Two-Phase GPU Meshlet Culler
//
// Pass 0 (PRISM_CULL_PASS_PHASE1_MAIN):
//   - Evaluates candidate (instance_id, meshlet_local_id) items.
//   - Runs Frustum + Backface Normal-Cone + Previous-Frame Reprojected Hi-Z.
//   - Visible meshlets -> append to VisibleMeshletBuffer + DrawIndirect commands.
//   - Hi-Z occluded meshlets -> append to FalseNegativeBuffer for Phase 2.
//
// Pass 1 (PRISM_CULL_PASS_PHASE2_DISOCCLUSION):
//   - Reads FalseNegativeBuffer emitted in Phase 1.
//   - Tests against CURRENT-frame Hi-Z rebuilt after Phase 1 rasterization.
//   - Newly disoccluded meshlets -> append to VisibleMeshletBuffer + DrawIndirect.
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

// Visible meshlet list consumed by VisBuffer rasterizer
layout(set = 0, binding = 5, std430) writeonly buffer VisibleMeshletBuffer {
    VisibleMeshlet b_visible_meshlets[];
};

// Indirect draw commands for standard vertex-pulling fallback path
layout(set = 0, binding = 6, std430) writeonly buffer DrawIndirectBuffer {
    VkDrawIndexedIndirectCommand b_draw_commands[];
};

// Atomic counters:
//   counters[0] = visible_meshlet_count (reset to 0 before each phase)
//   counters[1] = false_negative_count  (reset to 0 before Phase 1; read in Phase 2)
layout(set = 0, binding = 7, std430) buffer CullCountersBuffer {
    uint b_counters[];
};

// Indirect command for VK_EXT_mesh_shader path (1 workgroup per visible meshlet)
layout(set = 0, binding = 8, std430) buffer MeshTasksIndirectBuffer {
    VkDrawMeshTasksIndirectCommandEXT b_mesh_tasks_cmd;
};

// Phase 1: previous frame's Hi-Z pyramid
// Phase 2: current frame's post-Phase-1 Hi-Z pyramid
layout(set = 0, binding = 9) uniform sampler2D u_hiz_pyramid;

void emit_visible_meshlet(uint instance_id, uint meshlet_local_id, GpuMeshlet m) {
    uint slot = atomicAdd(b_counters[0], 1u);
    if (slot >= u_cull.max_visible_meshlets) {
        return;
    }

    VisibleMeshlet vis;
    vis.instance_id      = instance_id;
    vis.meshlet_local_id = meshlet_local_id;
    b_visible_meshlets[slot] = vis;

    // Populate vertex-pulling indirect draw command (3 indices per micro-triangle)
    VkDrawIndexedIndirectCommand cmd;
    cmd.indexCount    = m.triangle_count * 3u;
    cmd.instanceCount = 1u;
    cmd.firstIndex    = 0u;
    cmd.vertexOffset  = 0;
    cmd.firstInstance = slot;
    b_draw_commands[slot] = cmd;

    // Keep mesh-shader indirect workgroup count synchronized
    atomicMax(b_mesh_tasks_cmd.groupCountX, slot + 1u);
    b_mesh_tasks_cmd.groupCountY = 1u;
    b_mesh_tasks_cmd.groupCountZ = 1u;
}

void main() {
    uint tid = gl_GlobalInvocationID.x;

    uint instance_id      = 0u;
    uint meshlet_local_id = 0u;

    if (u_cull.cull_pass_mode == PRISM_CULL_PASS_PHASE1_MAIN) {
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

    // Transform bounding sphere into world space (conservative uniform max-axis scale)
    mat4 M = inst.world_from_local;
    vec3 world_center = (M * vec4(m.bound_sphere.xyz, 1.0)).xyz;
    float sx = length(M[0].xyz);
    float sy = length(M[1].xyz);
    float sz = length(M[2].xyz);
    float max_scale = max(sx, max(sy, sz));
    float world_radius = m.bound_sphere.w * max_scale;

    // 1. Frustum Culling (Phase 1 only; Phase 2 items already passed frustum & cone)
    if (u_cull.cull_pass_mode == PRISM_CULL_PASS_PHASE1_MAIN) {
        if (u_cull.enable_frustum_cull != 0u) {
            if (!sphere_in_frustum(world_center, world_radius, u_cull.frustum_planes)) {
                return;
            }
        }

        // 2. Normal-Cone Backface Culling (skip for two-sided materials)
        bool two_sided = (inst.flags & PRISM_INSTANCE_FLAG_TWO_SIDED) != 0u;
        if (!two_sided && (u_cull.enable_cone_cull != 0u)) {
            vec3 world_apex = (M * vec4(m.cone_apex.xyz, 1.0)).xyz;
            // Transform cone axis by inverse-transpose upper 3x3 for non-uniform scale safety
            mat3 normal_mat = transpose(inverse(mat3(M)));
            vec3 world_axis = normalize(normal_mat * m.cone_axis_cut.xyz);
            if (cone_backface_culled(
                    world_apex,
                    world_axis,
                    m.cone_axis_cut.w,
                    u_cull.camera_pos_and_near.xyz)) {
                return;
            }
        }
    }

    // 3. Reverse-Z Hi-Z Occlusion Culling
    if (u_cull.enable_hiz_cull != 0u) {
        vec4 uv_rect;
        float nearest_rev_z;
        bool valid_proj;

        if (u_cull.cull_pass_mode == PRISM_CULL_PASS_PHASE1_MAIN) {
            // Reproject previous-frame world sphere into previous-frame Hi-Z
            mat4 prev_M = inst.prev_world_from_local;
            vec3 prev_world_center = (prev_M * vec4(m.bound_sphere.xyz, 1.0)).xyz;
            float prev_scale = max(
                length(prev_M[0].xyz),
                max(length(prev_M[1].xyz), length(prev_M[2].xyz))
            );
            valid_proj = project_sphere_reverse_z(
                prev_world_center,
                m.bound_sphere.w * prev_scale,
                u_cull.view,
                u_cull.prev_view_proj,
                u_cull.camera_pos_and_near.w,
                uv_rect,
                nearest_rev_z
            );
        } else {
            // Phase 2: test current-frame world sphere against post-Phase-1 Hi-Z
            valid_proj = project_sphere_reverse_z(
                world_center,
                world_radius,
                u_cull.view,
                u_cull.view_proj,
                u_cull.camera_pos_and_near.w,
                uv_rect,
                nearest_rev_z
            );
        }

        if (valid_proj && is_hiz_occluded(
                u_hiz_pyramid,
                uv_rect,
                nearest_rev_z,
                u_cull.hiz_viewport_info.xy,
                u_cull.hiz_mip_count)) {
            // In Phase 1, defer occluded meshlets to Phase 2 disocclusion pass
            if (u_cull.cull_pass_mode == PRISM_CULL_PASS_PHASE1_MAIN) {
                uint fn_slot = atomicAdd(b_counters[1], 1u);
                if (fn_slot < u_cull.max_false_negatives) {
                    FalseNegativeItem fn;
                    fn.instance_id      = instance_id;
                    fn.meshlet_local_id = meshlet_local_id;
                    b_false_negatives[fn_slot] = fn;
                }
            }
            return;
        }
    }

    // Passed all culling stages -> emit for Visibility Buffer rasterization
    emit_visible_meshlet(instance_id, meshlet_local_id, m);
}
