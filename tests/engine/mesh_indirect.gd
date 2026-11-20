extends SceneTree
# Checks indirect mesh draws on the custom engine build. Run through tests/engine/run_mesh_shader.sh.
#
# Default pass: every frame a transfer clears the indirect and count buffers, a compute shader
# writes the mesh task commands (and the count) into them, and a draw reads them as indirect
# commands. Command k launches 1 x (k + 1) x 1 workgroups, and workgroup (x, y) draws a triangle in
# cell (x, y) of a grid, so the first K commands light exactly the cells in column 0 above row K.
#
# "negative" pass (script argument): every misuse of the new calls must fail with an engine error,
# without a crash or a Vulkan validation error. The runner checks the error messages.

const SIZE := 64
const CELL := SIZE / 4  # Pixels per grid cell.
const COMMAND_SIZE := 12  # VkDrawMeshTasksIndirectCommandEXT: x, y and z group counts.
const COMMAND_COUNT := 4

const COMPUTE_SOURCE := """
#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 0, std430) restrict writeonly buffer Commands {
	uint data[];
} commands;
layout(set = 0, binding = 1, std430) restrict writeonly buffer Count {
	uint value;
} count;
layout(push_constant, std430) uniform Params {
	uint count;
	uint pad0;
	uint pad1;
	uint pad2;
} params;
void main() {
	for (uint k = 0; k < 4; k++) {
		commands.data[3 * k] = 1;
		commands.data[3 * k + 1] = k + 1;
		commands.data[3 * k + 2] = 1;
	}
	count.value = params.count;
}
"""

const VERTEX_SOURCE := """
#version 450
const vec2 positions[3] = vec2[](vec2(-0.5, -0.5), vec2(0.5, -0.5), vec2(0.0, 0.5));
void main() {
	gl_Position = vec4(positions[gl_VertexIndex % 3], 0.0, 1.0);
}
"""

const MESH_SOURCE := """
#version 450
#extension GL_EXT_mesh_shader : require

layout(local_size_x = 3) in;
layout(triangles, max_vertices = 3, max_primitives = 1) out;

const vec2 corners[3] = vec2[](vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(0.0, 1.0));

void main() {
	// Workgroup (x, y) draws in cell (x, y) of a 4 x 4 grid; a cell is 0.5 wide in NDC.
	vec2 center = vec2(-0.75 + 0.5 * float(gl_WorkGroupID.x), -0.75 + 0.5 * float(gl_WorkGroupID.y));
	SetMeshOutputsEXT(3, 1);
	gl_MeshVerticesEXT[gl_LocalInvocationIndex].gl_Position = vec4(center + corners[gl_LocalInvocationIndex] * 0.2, 0.0, 1.0);
	if (gl_LocalInvocationIndex == 0) {
		gl_PrimitiveTriangleIndicesEXT[0] = uvec3(0, 1, 2);
	}
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

	_check(rd.has_feature(RenderingDevice.SUPPORTS_MESH_SHADER), "SUPPORTS_MESH_SHADER is true")
	_check(rd.has_feature(RenderingDevice.SUPPORTS_DRAW_INDIRECT_COUNT), "SUPPORTS_DRAW_INDIRECT_COUNT is true")
	if failures > 0:
		_finish("MESH_INDIRECT_TEST")
		return

	if OS.get_cmdline_user_args().has("negative"):
		_run_negative()
		_finish("MESH_INDIRECT_NEGATIVE_TEST")
	else:
		_run_positive()
		_finish("MESH_INDIRECT_TEST")


func _finish(marker: String) -> void:
	owned.reverse()
	for rid in owned:
		rd.free_rid(rid)
	rd.free()
	print(marker, ": ", "PASS" if failures == 0 else "FAIL")
	quit(0 if failures == 0 else 1)


func _run_positive() -> void:
	var texture := _own(_create_target())
	var framebuffer := _own(rd.framebuffer_create([texture]))
	var commands := _own(_create_indirect_buffer(COMMAND_SIZE * COMMAND_COUNT))
	var count := _own(_create_indirect_buffer(16))

	var compute_shader := _own(_create_shader({"compute": COMPUTE_SOURCE}))
	var compute_pipeline := _own(rd.compute_pipeline_create(compute_shader))
	var compute_set := _own(_create_set(compute_shader, [commands, count]))
	var mesh_shader := _own(_create_shader({"mesh": MESH_SOURCE, "fragment": FRAGMENT_SOURCE}))
	var pipeline := _own(_create_pipeline(mesh_shader, rd.framebuffer_get_format(framebuffer)))

	# Plain indirect draws: the first 4, 2 and 1 commands.
	for draws in [4, 2, 1]:
		var draw_call := func(dl: int) -> void: rd.draw_list_draw_mesh_tasks_indirect(dl, commands, 0, draws, COMMAND_SIZE)
		var image := _render(framebuffer, texture, compute_pipeline, compute_set, commands, count, pipeline, draw_call)
		_check_rows(image, draws, "an indirect draw of %d commands lights %d cells" % [draws, draws])

	# Starting at the second command: commands 1 and 2 light rows 0 to 2.
	var offset_call := func(dl: int) -> void: rd.draw_list_draw_mesh_tasks_indirect(dl, commands, COMMAND_SIZE, 2, COMMAND_SIZE)
	var offset_image := _render(framebuffer, texture, compute_pipeline, compute_set, commands, count, pipeline, offset_call)
	_check_rows(offset_image, 3, "an indirect draw of 2 commands from offset 12 lights 3 cells")

	# Indirect-count draws with a maximum of 4 commands and a GPU-written count of 4, 2 and 0.
	for k in [4, 2, 0]:
		var count_call := func(dl: int) -> void: rd.draw_list_draw_mesh_tasks_indirect_count(dl, commands, 0, count, 0, COMMAND_COUNT, COMMAND_SIZE)
		var image := _render(framebuffer, texture, compute_pipeline, compute_set, commands, count, pipeline, count_call, k)
		_check_rows(image, k, "an indirect-count draw with a count of %d lights %d cells" % [k, k])


func _run_negative() -> void:
	var texture := _own(_create_target())
	var framebuffer := _own(rd.framebuffer_create([texture]))
	var framebuffer_format := rd.framebuffer_get_format(framebuffer)

	var mesh_pipeline := _own(_create_pipeline(_own(_create_shader({"mesh": MESH_SOURCE, "fragment": FRAGMENT_SOURCE})), framebuffer_format))
	var vertex_pipeline := _own(_create_pipeline(_own(_create_shader({"vertex": VERTEX_SOURCE, "fragment": FRAGMENT_SOURCE})), framebuffer_format))
	var commands := _own(_create_indirect_buffer(COMMAND_SIZE * COMMAND_COUNT))
	var count := _own(_create_indirect_buffer(16))
	var plain_buffer := _own(_create_buffer(COMMAND_SIZE * COMMAND_COUNT, 0))
	_check(mesh_pipeline.is_valid() and vertex_pipeline.is_valid(), "the mesh and vertex pipelines were created")

	var draw_list := rd.draw_list_begin(framebuffer, RenderingDevice.DRAW_CLEAR_ALL, PackedColorArray([Color.BLACK]))

	rd.draw_list_bind_render_pipeline(draw_list, mesh_pipeline)
	rd.draw_list_draw_mesh_tasks_indirect(draw_list, plain_buffer, 0, 1, COMMAND_SIZE)  # No indirect usage flag.
	rd.draw_list_draw_mesh_tasks_indirect_count(draw_list, commands, 0, plain_buffer, 0, 1, COMMAND_SIZE)  # Count buffer without the flag.
	rd.draw_list_draw_mesh_tasks_indirect(draw_list, commands, 48, 1, COMMAND_SIZE)  # Offset at the end of the buffer.
	rd.draw_list_draw_mesh_tasks_indirect(draw_list, commands, 40, 1, COMMAND_SIZE)  # The command straddles the end.
	rd.draw_list_draw_mesh_tasks_indirect(draw_list, commands, 0, 5, COMMAND_SIZE)  # Five commands do not fit.
	rd.draw_list_draw_mesh_tasks_indirect_count(draw_list, commands, 0, count, 16, 1, COMMAND_SIZE)  # Count past the end.
	rd.draw_list_draw_mesh_tasks_indirect(draw_list, commands, 0, 2, 4)  # Stride below the command size.
	rd.draw_list_draw_mesh_tasks_indirect(draw_list, commands, 2, 1, COMMAND_SIZE)  # Misaligned offset.
	rd.draw_list_draw_mesh_tasks_indirect_count(draw_list, commands, 0, count, 2, 1, COMMAND_SIZE)  # Misaligned count offset.

	rd.draw_list_bind_render_pipeline(draw_list, vertex_pipeline)
	rd.draw_list_draw_mesh_tasks_indirect(draw_list, commands, 0, 1, COMMAND_SIZE)  # No mesh shader.
	rd.draw_list_draw_mesh_tasks_indirect_count(draw_list, commands, 0, count, 0, 1, COMMAND_SIZE)  # No mesh shader.

	rd.draw_list_end()
	rd.submit()
	rd.sync()


func _own(rid: RID) -> RID:
	owned.push_back(rid)
	return rid


func _create_target() -> RID:
	var format := RDTextureFormat.new()
	format.width = SIZE
	format.height = SIZE
	format.format = RenderingDevice.DATA_FORMAT_R8G8B8A8_UNORM
	format.usage_bits = RenderingDevice.TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT
	return rd.texture_create(format, RDTextureView.new())


func _create_indirect_buffer(size: int) -> RID:
	return _create_buffer(size, RenderingDevice.STORAGE_BUFFER_USAGE_DISPATCH_INDIRECT)


func _create_buffer(size: int, usage: int) -> RID:
	var zeros := PackedByteArray()
	zeros.resize(size)
	return rd.storage_buffer_create(size, zeros, usage)


# sources maps a stage name (mesh, vertex, fragment, compute) to its GLSL.
func _create_shader(sources: Dictionary) -> RID:
	var source := RDShaderSource.new()
	source.language = RenderingDevice.SHADER_LANGUAGE_GLSL
	for stage in sources:
		source.set("source_" + stage, sources[stage])
	var spirv := rd.shader_compile_spirv_from_source(source)
	for stage in sources:
		var stage_error: String = spirv.get("compile_error_" + stage)
		if stage_error != "":
			_fail("shader compile error in the %s stage: %s" % [stage, stage_error])
	return rd.shader_create_from_spirv(spirv)


# buffers are bound to bindings 0, 1, ... as storage buffers.
func _create_set(shader: RID, buffers: Array[RID]) -> RID:
	var uniforms: Array[RDUniform] = []
	for binding in buffers.size():
		var uniform := RDUniform.new()
		uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		uniform.binding = binding
		uniform.add_id(buffers[binding])
		uniforms.push_back(uniform)
	return rd.uniform_set_create(uniforms, shader, 0)


func _create_pipeline(shader: RID, framebuffer_format: int) -> RID:
	var blend := RDPipelineColorBlendState.new()
	blend.attachments.push_back(RDPipelineColorBlendStateAttachment.new())
	return rd.render_pipeline_create(
		shader, framebuffer_format, RenderingDevice.INVALID_ID, RenderingDevice.RENDER_PRIMITIVE_TRIANGLES,
		RDPipelineRasterizationState.new(), RDPipelineMultisampleState.new(), RDPipelineDepthStencilState.new(), blend)


# Clears the buffers, has the compute shader write the commands and the count, then runs draw_call
# on a draw list with the mesh pipeline bound, all in one frame.
func _render(framebuffer: RID, texture: RID, compute_pipeline: RID, compute_set: RID, commands: RID, count: RID, pipeline: RID, draw_call: Callable, count_value: int = 0) -> PackedByteArray:
	rd.buffer_clear(commands, 0, COMMAND_SIZE * COMMAND_COUNT)
	rd.buffer_clear(count, 0, 16)

	var params := PackedByteArray()
	params.resize(16)
	params.encode_u32(0, count_value)
	var compute_list := rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list, compute_pipeline)
	rd.compute_list_bind_uniform_set(compute_list, compute_set, 0)
	rd.compute_list_set_push_constant(compute_list, params, params.size())
	rd.compute_list_dispatch(compute_list, 1, 1, 1)
	rd.compute_list_end()

	var draw_list := rd.draw_list_begin(framebuffer, RenderingDevice.DRAW_CLEAR_ALL, PackedColorArray([Color.BLACK]))
	rd.draw_list_bind_render_pipeline(draw_list, pipeline)
	draw_call.call(draw_list)
	rd.draw_list_end()
	rd.submit()
	rd.sync()
	return rd.texture_get_data(texture, 0)


# Cells (0, 0) to (0, lit_rows - 1) must be lit and every other cell dark.
func _check_rows(data: PackedByteArray, lit_rows: int, what: String) -> void:
	var ok := true
	for row in 4:
		for column in 4:
			var lit := data[((row * CELL + CELL / 2) * SIZE + column * CELL + CELL / 2) * 4] > 0
			if lit != (column == 0 and row < lit_rows):
				ok = false
	_check(ok, what)


func _check(condition: bool, what: String) -> void:
	if condition:
		print("ok: ", what)
	else:
		_fail(what)


func _fail(what: String) -> void:
	failures += 1
	printerr("FAILED: ", what)
