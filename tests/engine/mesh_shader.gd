extends SceneTree
# Checks mesh and task shaders on the custom engine build. Run through tests/engine/run_mesh_shader.sh.
#
# Default pass: every frame a transfer clears a storage buffer, a compute shader writes a triangle
# half-size (0.2) into it, and then
#   - a mesh shader reads it and draws one triangle in the left half of the target, and
#   - a task shader reads it, passes 1.5 times that size to two mesh workgroups through a payload,
#     and each workgroup draws one triangle (half-size 0.3) in its own half.
# The triangle areas show which value each stage used, and the clear, compute and draw passes
# make the barrier and stage-mask logic visible to synchronization validation.
#
# "negative" pass (script argument): every misuse of the new API must fail with an engine error,
# without a crash or a Vulkan validation error. The runner checks the error messages.

const SIZE := 64
const PIXELS_PER_UNIT := SIZE / 2.0  # NDC spans 2 units over SIZE pixels.

const COMPUTE_SOURCE := """
#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 0, std430) restrict buffer Size {
	float value;
} size_buf;
void main() {
	size_buf.value = 0.2;
}
"""

const VERTEX_SOURCE := """
#version 450
const vec2 positions[3] = vec2[](vec2(-0.5, -0.5), vec2(0.5, -0.5), vec2(0.0, 0.5));
void main() {
	gl_Position = vec4(positions[gl_VertexIndex % 3], 0.0, 1.0);
}
"""

# The //DEFINES line picks where the triangle size comes from: USE_PAYLOAD (from a task shader),
# USE_SIZE_BUFFER (from the storage buffer) or neither (a constant).
const MESH_SOURCE := """
#version 450
#extension GL_EXT_mesh_shader : require
//DEFINES

layout(local_size_x = 3) in;
layout(triangles, max_vertices = 3, max_primitives = 1) out;

#if defined(USE_PAYLOAD)
struct Payload {
	float half_size;
};
taskPayloadSharedEXT Payload payload;
#elif defined(USE_SIZE_BUFFER)
layout(set = 0, binding = 0, std430) readonly buffer Size {
	float value;
} size_buf;
#endif

const vec2 corners[3] = vec2[](vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(0.0, 1.0));

void main() {
#if defined(USE_PAYLOAD)
	float half_size = payload.half_size;
#elif defined(USE_SIZE_BUFFER)
	float half_size = size_buf.value;
#else
	float half_size = 0.25;
#endif
	// Workgroup k is centered at x = -0.5 + k, so group 0 is the left half and group 1 the right half.
	vec2 center = vec2(-0.5 + float(gl_WorkGroupID.x), 0.0);
	SetMeshOutputsEXT(3, 1);
	gl_MeshVerticesEXT[gl_LocalInvocationIndex].gl_Position = vec4(center + corners[gl_LocalInvocationIndex] * half_size, 0.0, 1.0);
	if (gl_LocalInvocationIndex == 0) {
		gl_PrimitiveTriangleIndicesEXT[0] = uvec3(0, 1, 2);
	}
}
"""

const TASK_SOURCE := """
#version 450
#extension GL_EXT_mesh_shader : require
//DEFINES

layout(local_size_x = 1) in;

struct Payload {
	float half_size;
};
taskPayloadSharedEXT Payload payload;

#ifdef USE_SIZE_BUFFER
layout(set = 0, binding = 0, std430) readonly buffer Size {
	float value;
} size_buf;
#endif

void main() {
#ifdef USE_SIZE_BUFFER
	payload.half_size = size_buf.value * 1.5;
#else
	payload.half_size = 0.25;
#endif
	EmitMeshTasksEXT(2, 1, 1);
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
	_check(rd.has_feature(RenderingDevice.SUPPORTS_TASK_SHADER), "SUPPORTS_TASK_SHADER is true")
	if failures > 0:
		_finish("MESH_SHADER_TEST")
		return

	if OS.get_cmdline_user_args().has("negative"):
		_run_negative()
		_finish("MESH_SHADER_NEGATIVE_TEST")
	else:
		_run_positive()
		_finish("MESH_SHADER_TEST")


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
	var size_buffer := _own(rd.storage_buffer_create(16, PackedByteArray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])))

	var compute_shader := _own(_create_shader({"compute": COMPUTE_SOURCE}))
	var compute_pipeline := _own(rd.compute_pipeline_create(compute_shader))
	var compute_set := _own(_create_set(compute_shader, size_buffer))

	var mesh_only := _render_with_compute(framebuffer, texture, size_buffer, compute_pipeline, compute_set,
			{"mesh": MESH_SOURCE.replace("//DEFINES", "#define USE_SIZE_BUFFER"), "fragment": FRAGMENT_SOURCE})
	var with_task := _render_with_compute(framebuffer, texture, size_buffer, compute_pipeline, compute_set,
			{"task": TASK_SOURCE.replace("//DEFINES", "#define USE_SIZE_BUFFER"), "mesh": MESH_SOURCE.replace("//DEFINES", "#define USE_PAYLOAD"), "fragment": FRAGMENT_SOURCE})

	# A triangle with half-size h covers 2 * h^2 * PIXELS_PER_UNIT^2 pixels: 82 for 0.2, 184 for 0.3, and 128 for
	# the 0.25 a stage that ignored its input would use. Pixel coverage at the edges moves the counts by about 10%.
	var left_mesh_only := _lit_pixels(mesh_only, 0, SIZE / 2)
	var right_mesh_only := _lit_pixels(mesh_only, SIZE / 2, SIZE)
	_check(_lit(mesh_only, 16, 32), "a mesh shader draws a triangle in the left half")
	_check(right_mesh_only == 0, "a mesh shader without a task shader draws nothing in the right half")
	_check(_near(left_mesh_only, _triangle_area(0.2), 0.2), "the mesh shader used the half-size the compute shader wrote (0.2): %d pixels" % left_mesh_only)

	var left_with_task := _lit_pixels(with_task, 0, SIZE / 2)
	var right_with_task := _lit_pixels(with_task, SIZE / 2, SIZE)
	_check(_lit(with_task, 16, 32) and _lit(with_task, 48, 32), "a task shader launching two mesh groups draws a triangle in each half")
	_check(not _lit(with_task, 32, 32), "nothing is drawn between the two triangles")
	_check(_near(left_with_task, _triangle_area(0.3), 0.2), "the left triangle used the payload size (0.3): %d pixels" % left_with_task)
	_check(_near(right_with_task, _triangle_area(0.3), 0.2), "the right triangle used the payload size (0.3): %d pixels" % right_with_task)


func _run_negative() -> void:
	var texture := _own(_create_target())
	var framebuffer := _own(rd.framebuffer_create([texture]))
	var framebuffer_format := rd.framebuffer_get_format(framebuffer)

	var mesh_shader := _own(_create_shader({"mesh": MESH_SOURCE, "fragment": FRAGMENT_SOURCE}))
	var task_shader := _own(_create_shader({"task": TASK_SOURCE.replace("//DEFINES", ""), "mesh": MESH_SOURCE.replace("//DEFINES", "#define USE_PAYLOAD"), "fragment": FRAGMENT_SOURCE}))
	var vertex_shader := _own(_create_shader({"vertex": VERTEX_SOURCE, "fragment": FRAGMENT_SOURCE}))
	var mesh_vertex_shader := _own(_create_shader({"vertex": VERTEX_SOURCE, "mesh": MESH_SOURCE, "fragment": FRAGMENT_SOURCE}))
	var task_only_shader := _own(_create_shader({"task": TASK_SOURCE.replace("//DEFINES", ""), "fragment": FRAGMENT_SOURCE}))
	var task_vertex_shader := _own(_create_shader({"task": TASK_SOURCE.replace("//DEFINES", ""), "vertex": VERTEX_SOURCE, "fragment": FRAGMENT_SOURCE}))

	var attribute := RDVertexAttribute.new()
	attribute.format = RenderingDevice.DATA_FORMAT_R32G32_SFLOAT
	attribute.stride = 8
	var vertex_format := rd.vertex_format_create([attribute])

	_check(not _create_pipeline(mesh_vertex_shader, framebuffer_format).is_valid(), "a pipeline with vertex and mesh shaders is rejected")
	_check(not _create_pipeline(mesh_shader, framebuffer_format, vertex_format).is_valid(), "a mesh pipeline with a vertex format is rejected")
	_check(not _create_pipeline(task_only_shader, framebuffer_format).is_valid(), "a pipeline with a task shader and no vertex or mesh shader is rejected")
	_check(not _create_pipeline(task_vertex_shader, framebuffer_format).is_valid(), "a pipeline with a task shader and a vertex shader is rejected")

	var mesh_pipeline := _own(_create_pipeline(mesh_shader, framebuffer_format))
	var task_pipeline := _own(_create_pipeline(task_shader, framebuffer_format))
	var vertex_pipeline := _own(_create_pipeline(vertex_shader, framebuffer_format))
	_check(mesh_pipeline.is_valid() and task_pipeline.is_valid() and vertex_pipeline.is_valid(), "the valid mesh, task and vertex pipelines were created")

	var indirect_buffer := _own(rd.storage_buffer_create(32, PackedByteArray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), RenderingDevice.STORAGE_BUFFER_USAGE_DISPATCH_INDIRECT))
	var mesh_limits := [
		rd.limit_get(RenderingDevice.LIMIT_MAX_MESH_WORKGROUP_COUNT_X),
		rd.limit_get(RenderingDevice.LIMIT_MAX_MESH_WORKGROUP_COUNT_Y),
		rd.limit_get(RenderingDevice.LIMIT_MAX_MESH_WORKGROUP_COUNT_Z),
		rd.limit_get(RenderingDevice.LIMIT_MAX_MESH_WORKGROUP_COUNT_TOTAL)]
	var task_limit_x := rd.limit_get(RenderingDevice.LIMIT_MAX_TASK_WORKGROUP_COUNT_X)
	print("mesh workgroup count limits (x, y, z, total): ", mesh_limits, ", task x: ", task_limit_x)

	var draw_list := rd.draw_list_begin(framebuffer, RenderingDevice.DRAW_CLEAR_ALL, PackedColorArray([Color.BLACK]))

	# Ordinary draws on a mesh pipeline.
	rd.draw_list_bind_render_pipeline(draw_list, mesh_pipeline)
	rd.draw_list_draw(draw_list, false, 1, 3)
	rd.draw_list_draw_indirect(draw_list, false, indirect_buffer, 0, 1, 16)
	rd.draw_list_draw_indirect_count(draw_list, false, indirect_buffer, 0, indirect_buffer, 16, 1, 16)

	# Counts over the limits, for a mesh and a task pipeline.
	if mesh_limits[0] < 0xFFFFFFFF:
		rd.draw_list_draw_mesh_tasks(draw_list, mesh_limits[0] + 1, 1, 1)
	if mesh_limits[0] * mesh_limits[1] * mesh_limits[2] > mesh_limits[3] and mesh_limits[0] <= 0xFFFFFFFF and mesh_limits[1] <= 0xFFFFFFFF and mesh_limits[2] <= 0xFFFFFFFF:
		rd.draw_list_draw_mesh_tasks(draw_list, mesh_limits[0], mesh_limits[1], mesh_limits[2])
	rd.draw_list_bind_render_pipeline(draw_list, task_pipeline)
	if task_limit_x < 0xFFFFFFFF:
		rd.draw_list_draw_mesh_tasks(draw_list, task_limit_x + 1, 1, 1)

	# A mesh draw on a pipeline without a mesh shader.
	rd.draw_list_bind_render_pipeline(draw_list, vertex_pipeline)
	rd.draw_list_draw_mesh_tasks(draw_list, 1, 1, 1)

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


# sources maps a stage name (task, mesh, vertex, fragment, compute) to its GLSL.
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


func _create_set(shader: RID, buffer: RID) -> RID:
	var uniform := RDUniform.new()
	uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	uniform.binding = 0
	uniform.add_id(buffer)
	return rd.uniform_set_create([uniform], shader, 0)


func _create_pipeline(shader: RID, framebuffer_format: int, vertex_format: int = RenderingDevice.INVALID_ID) -> RID:
	var blend := RDPipelineColorBlendState.new()
	blend.attachments.push_back(RDPipelineColorBlendStateAttachment.new())
	return rd.render_pipeline_create(
		shader, framebuffer_format, vertex_format, RenderingDevice.RENDER_PRIMITIVE_TRIANGLES,
		RDPipelineRasterizationState.new(), RDPipelineMultisampleState.new(), RDPipelineDepthStencilState.new(), blend)


# Clears the size buffer, has the compute shader write the size, then draws with a mesh pipeline
# that reads it, all in one frame.
func _render_with_compute(framebuffer: RID, texture: RID, size_buffer: RID, compute_pipeline: RID, compute_set: RID, sources: Dictionary) -> PackedByteArray:
	var shader := _own(_create_shader(sources))
	var pipeline := _own(_create_pipeline(shader, rd.framebuffer_get_format(framebuffer)))
	var shader_set := _own(_create_set(shader, size_buffer))

	rd.buffer_clear(size_buffer, 0, 16)

	var compute_list := rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list, compute_pipeline)
	rd.compute_list_bind_uniform_set(compute_list, compute_set, 0)
	rd.compute_list_dispatch(compute_list, 1, 1, 1)
	rd.compute_list_end()

	var draw_list := rd.draw_list_begin(framebuffer, RenderingDevice.DRAW_CLEAR_ALL, PackedColorArray([Color.BLACK]))
	rd.draw_list_bind_render_pipeline(draw_list, pipeline)
	rd.draw_list_bind_uniform_set(draw_list, shader_set, 0)
	rd.draw_list_draw_mesh_tasks(draw_list, 1, 1, 1)
	rd.draw_list_end()
	rd.submit()
	rd.sync()
	return rd.texture_get_data(texture, 0)


func _triangle_area(half_size: float) -> float:
	return 2.0 * half_size * half_size * PIXELS_PER_UNIT * PIXELS_PER_UNIT


func _near(value: int, expected: float, tolerance: float) -> bool:
	return absf(value - expected) <= expected * tolerance


func _lit(data: PackedByteArray, x: int, y: int) -> bool:
	return data[(y * SIZE + x) * 4] > 0


# Lit pixels in the columns from x_from (inclusive) to x_to (exclusive).
func _lit_pixels(data: PackedByteArray, x_from: int, x_to: int) -> int:
	var lit := 0
	for y in SIZE:
		for x in range(x_from, x_to):
			if data[(y * SIZE + x) * 4] > 0:
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
