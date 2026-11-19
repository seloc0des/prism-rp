extends SceneTree
# Checks RenderingDevice.draw_list_draw_indirect_count on the custom engine build: of four
# single-triangle draws in an indirect buffer, only as many as the count buffer says are drawn.
# Run through tests/engine/run_indirect_count.sh.

const SIZE := 64
const DRAW_COUNT := 4
const DRAW_STRIDE := 16  # vertex count, instance count, first vertex, first instance

const VERTEX_SOURCE := """
#version 450
// Draw k covers vertices 3k..3k+2 and lands in its own quadrant.
const vec2 centers[4] = vec2[](vec2(-0.5, -0.5), vec2(0.5, -0.5), vec2(-0.5, 0.5), vec2(0.5, 0.5));
const vec2 corners[3] = vec2[](vec2(-0.25, -0.25), vec2(0.25, -0.25), vec2(0.0, 0.25));
void main() {
	gl_Position = vec4(centers[gl_VertexIndex / 3] + corners[gl_VertexIndex % 3], 0.0, 1.0);
}
"""

const FRAGMENT_SOURCE := """
#version 450
layout(location = 0) out vec4 color;
void main() {
	color = vec4(1.0);
}
"""

var rd: RenderingDevice
var failures := 0
var owned: Array[RID] = []  # freed in reverse order, or the validation layers report leaks


func _init() -> void:
	rd = RenderingServer.create_local_rendering_device()
	if rd == null:
		_fail("no local RenderingDevice (is the renderer running with Vulkan?)")
		quit(1)
		return

	_check(rd.has_feature(RenderingDevice.SUPPORTS_DRAW_INDIRECT_COUNT), "SUPPORTS_DRAW_INDIRECT_COUNT is true")

	var shader := _own(_create_shader())
	var texture := _own(_create_target())
	var framebuffer := _own(rd.framebuffer_create([texture]))
	var pipeline := _own(_create_pipeline(shader, framebuffer))

	var draws := PackedByteArray()
	for k in DRAW_COUNT:
		for field in [3, 1, 3 * k, 0]:
			draws.append_array(_u32(field))
	var draw_buffer := _own(rd.storage_buffer_create(draws.size(), draws, RenderingDevice.STORAGE_BUFFER_USAGE_DISPATCH_INDIRECT))

	var all_drawn := _render(framebuffer, texture, pipeline, draw_buffer, -1, 0)  # plain indirect, 4 draws
	var first_two := _render(framebuffer, texture, pipeline, draw_buffer, -1, 2)  # plain indirect, 2 draws
	var counted_two := _render(framebuffer, texture, pipeline, draw_buffer, 2, 0)
	var counted_zero := _render(framebuffer, texture, pipeline, draw_buffer, 0, 0)
	var counted_all := _render(framebuffer, texture, pipeline, draw_buffer, 4, 0)

	_check(_lit_pixels(first_two) > 0, "the reference render of two draws is not empty")
	_check(_lit_pixels(first_two) < _lit_pixels(all_drawn), "four draws light more pixels than two")
	_check(counted_two == first_two, "a count of 2 draws exactly the first two draws")
	_check(_lit_pixels(counted_zero) == 0, "a count of 0 draws nothing")
	_check(counted_all == all_drawn, "a count of 4 draws all four")

	owned.reverse()
	for rid in owned:
		rd.free_rid(rid)
	rd.free()

	print("INDIRECT_COUNT_TEST: ", "PASS" if failures == 0 else "FAIL")
	quit(0 if failures == 0 else 1)


func _u32(value: int) -> PackedByteArray:
	var bytes := PackedByteArray()
	bytes.resize(4)
	bytes.encode_u32(0, value)
	return bytes


func _own(rid: RID) -> RID:
	owned.push_back(rid)
	return rid


func _create_shader() -> RID:
	var source := RDShaderSource.new()
	source.language = RenderingDevice.SHADER_LANGUAGE_GLSL
	source.source_vertex = VERTEX_SOURCE
	source.source_fragment = FRAGMENT_SOURCE
	var spirv := rd.shader_compile_spirv_from_source(source)
	for stage_error in [spirv.compile_error_vertex, spirv.compile_error_fragment]:
		if stage_error != "":
			_fail("shader compile error: " + stage_error)
	return rd.shader_create_from_spirv(spirv)


func _create_target() -> RID:
	var format := RDTextureFormat.new()
	format.width = SIZE
	format.height = SIZE
	format.format = RenderingDevice.DATA_FORMAT_R8G8B8A8_UNORM
	format.usage_bits = RenderingDevice.TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT
	return rd.texture_create(format, RDTextureView.new())


func _create_pipeline(shader: RID, framebuffer: RID) -> RID:
	var blend := RDPipelineColorBlendState.new()
	blend.attachments.push_back(RDPipelineColorBlendStateAttachment.new())
	return rd.render_pipeline_create(
		shader, rd.framebuffer_get_format(framebuffer), RenderingDevice.INVALID_ID,
		RenderingDevice.RENDER_PRIMITIVE_TRIANGLES, RDPipelineRasterizationState.new(),
		RDPipelineMultisampleState.new(), RDPipelineDepthStencilState.new(), blend)


# count < 0 draws with plain indirect and `plain_draws` draws; count >= 0 uses the count buffer
# (max draw count 4) holding that count.
func _render(framebuffer: RID, texture: RID, pipeline: RID, draw_buffer: RID, count: int, plain_draws: int) -> PackedByteArray:
	var draw_list := rd.draw_list_begin(framebuffer, RenderingDevice.DRAW_CLEAR_ALL, PackedColorArray([Color.BLACK]))
	rd.draw_list_bind_render_pipeline(draw_list, pipeline)
	if count < 0:
		rd.draw_list_draw_indirect(draw_list, false, draw_buffer, 0, plain_draws if plain_draws > 0 else DRAW_COUNT, DRAW_STRIDE)
	else:
		var count_buffer := _own(rd.storage_buffer_create(4, _u32(count), RenderingDevice.STORAGE_BUFFER_USAGE_DISPATCH_INDIRECT))
		rd.draw_list_draw_indirect_count(draw_list, false, draw_buffer, 0, count_buffer, 0, DRAW_COUNT, DRAW_STRIDE)
	rd.draw_list_end()
	rd.submit()
	rd.sync()
	return rd.texture_get_data(texture, 0)


func _lit_pixels(data: PackedByteArray) -> int:
	var lit := 0
	for i in range(0, data.size(), 4):
		if data[i] > 0:
			lit += 1
	return lit


func _check(condition: bool, what: String) -> void:
	if condition:
		print("ok: ", what)
	else:
		_fail(what)


func _fail(what: String) -> void:
	failures += 1
	printerr("FAILED: ", what)
