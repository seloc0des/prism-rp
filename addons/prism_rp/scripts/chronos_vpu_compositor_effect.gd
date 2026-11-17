@tool
class_name ChronosVPUCompositorEffect
extends CompositorEffect

var rd: RenderingDevice
var vpu_shader: RID
var vpu_pipeline: RID

func _init() -> void:
	effect_callback_type = EFFECT_CALLBACK_TYPE_POST_TRANSPARENT
	access_resolved_color = true
	access_resolved_depth = true
	needs_motion_vectors = true
	rd = RenderingServer.get_rendering_device()

func _render_callback(p_effect_callback_type: int, p_render_data: RenderData) -> void:
	var scene_buffers := p_render_data.get_render_scene_buffers() as RenderSceneBuffersRD
	if not scene_buffers or not vpu_pipeline.is_valid():
		return

	var internal_size := scene_buffers.get_internal_size()
	for view in range(scene_buffers.get_view_count()):
		var color_tex := scene_buffers.get_color_layer(view)
		var depth_tex := scene_buffers.get_depth_layer(view)
		var velocity_tex := scene_buffers.get_velocity_layer(view)

		# Bind HDR color, depth, dilated motion vectors, and temporal history buffer
		# Dispatch compute shader for SVGF denoising + reactive-mask temporal upscaling
		var compute_list := rd.compute_list_begin()
		rd.compute_list_bind_compute_pipeline(compute_list, vpu_pipeline)
		rd.compute_list_dispatch(compute_list, ceili(internal_size.x / 8.0), ceili(internal_size.y / 8.0), 1)
		rd.compute_list_end()
