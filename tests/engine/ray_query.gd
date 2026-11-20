extends SceneTree
# Checks ray queries (patch 0006) on the custom engine build. Run through tests/engine/run_ray_query.sh.
#
# Default pass: a BLAS holds one triangle, and a TLAS with room for two instances holds one instance of it, with no
# hit shader binding table range (ray queries do not use one). A compute shader traces a GRID x GRID grid of rays
# along -Z with rayQueryEXT and stores each ray's hit distance, barycentrics and instance custom index. Four traces,
# each after a build that moves the instance:
#   A. at the origin (custom index 7). The build takes a new instance buffer;
#   B. in the same submission, +0.5 in x (custom index 9). The build takes the second slot of that buffer, so trace A
#      must still see A's instance. Stock 4.7.2 put it in the first slot, over A's, before the GPU had read them
#      (fixed in patch 0006);
#   C. still in the same submission, -0.5 in x (custom index 11). The buffer is full, so the build takes a second one;
#   D. in the next submission, back at the origin (custom index 13). The first buffer is free again and is reused from
#      the start, the way a renderer rebuilds its TLAS every frame.
# Each rebuild must wait for the trace before it to finish reading the TLAS, and each trace must wait for its build,
# which synchronization validation checks. Each trace is compared ray by ray with a point-in-triangle test on the CPU.
#
# "negative" pass (script argument): a TLAS build must still reject an instance whose BLAS was never built and one
# whose BLAS was freed, without a crash or a Vulkan validation error. The runner checks the error messages.

const GRID := 8  # matches local_size and the 8u in TRACE_SOURCE
const RAYS := GRID * GRID
const TRACES := 4
const TOLERANCE := 1e-4

# Model-space corners of the triangle, at z = 0.
const TRIANGLE: Array[Vector2] = [Vector2(-0.5, -0.5), Vector2(0.5, -0.5), Vector2(0.0, 0.5)]

# Rays start at z = 1 over [-1, 1]^2, one per grid cell centre, so a hit is at t = 1. No cell centre lies on a
# triangle edge at any of the instance positions, so every ray is a clear hit or a clear miss.
const TRACE_SOURCE := """
#version 460
#extension GL_EXT_ray_query : require

layout(local_size_x = 8, local_size_y = 8) in;

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;
layout(set = 0, binding = 1, std430) restrict writeonly buffer Hits {
	vec4 data[];
} hits;

layout(push_constant, std430) uniform Params {
	uint trace;
} params;

void main() {
	uvec2 cell = gl_GlobalInvocationID.xy;
	vec2 xy = (vec2(cell) + 0.5) * (2.0 / 8.0) - 1.0;
	rayQueryEXT query;
	rayQueryInitializeEXT(query, tlas, gl_RayFlagsOpaqueEXT, 0xFF, vec3(xy, 1.0), 0.0, vec3(0.0, 0.0, -1.0), 4.0);
	while (rayQueryProceedEXT(query)) {
	}
	// (t, barycentrics, custom index) for a hit; all -1 for a miss.
	vec4 result = vec4(-1.0);
	if (rayQueryGetIntersectionTypeEXT(query, true) == gl_RayQueryCommittedIntersectionTriangleEXT) {
		result = vec4(rayQueryGetIntersectionTEXT(query, true), rayQueryGetIntersectionBarycentricsEXT(query, true),
				float(rayQueryGetIntersectionInstanceCustomIndexEXT(query, true)));
	}
	hits.data[params.trace * 64u + cell.y * 8u + cell.x] = result;
}
"""

var rd: RenderingDevice
var failures := 0
var owned: Array[RID] = []  # freed in reverse order: freeing a resource first also frees what depends on it


func _init() -> void:
	rd = RenderingServer.create_local_rendering_device()
	if rd == null:
		_fail("no local RenderingDevice (is the renderer running with Vulkan?)")
		quit(1)
		return

	var negative := OS.get_cmdline_user_args().has("negative")
	var marker := "RAY_QUERY_NEGATIVE_TEST" if negative else "RAY_QUERY_TEST"
	print("device: ", rd.get_device_name())
	_check(rd.has_feature(RenderingDevice.SUPPORTS_RAY_QUERY), "SUPPORTS_RAY_QUERY is true")
	if failures == 0:
		if negative:
			_run_negative()
		else:
			_run_positive()
	_finish(marker)


func _finish(marker: String) -> void:
	owned.reverse()
	for rid in owned:
		rd.free_rid(rid)
	rd.free()
	print(marker, ": ", "PASS" if failures == 0 else "FAIL")
	quit(0 if failures == 0 else 1)


func _run_positive() -> void:
	var blas := _own(_create_built_blas())
	var tlas := _own(rd.tlas_create(2, 0))  # two slots per instance buffer: B shares A's buffer, C needs another
	var hit_buffer := _own(rd.storage_buffer_create(TRACES * RAYS * 16))
	var shader := _own(_create_shader(TRACE_SOURCE))
	var pipeline := _own(rd.compute_pipeline_create(shader))
	var uniform_set := _own(_create_set(shader, tlas, hit_buffer))
	_check(tlas.is_valid() and pipeline.is_valid() and uniform_set.is_valid(), "the TLAS, the ray query pipeline and its uniform set were created")
	if failures > 0:
		return

	var offsets: Array[Vector3] = [Vector3.ZERO, Vector3(0.5, 0.0, 0.0), Vector3(-0.5, 0.0, 0.0), Vector3.ZERO]
	var custom_indices: Array[int] = [7, 9, 11, 13]

	_check(rd.tlas_build(tlas, [_instance(blas, offsets[0], custom_indices[0])]) == OK, "trace A: the TLAS builds with a zero hit shader binding table range")
	_trace(pipeline, uniform_set, 0)
	_check(rd.tlas_build(tlas, [_instance(blas, offsets[1], custom_indices[1])]) == OK, "trace B: the TLAS rebuilds in the same submission")
	_trace(pipeline, uniform_set, 1)
	_check(rd.tlas_build(tlas, [_instance(blas, offsets[2], custom_indices[2])]) == OK, "trace C: the TLAS rebuilds again in the same submission")
	_trace(pipeline, uniform_set, 2)
	rd.submit()
	rd.sync()

	_check(rd.tlas_build(tlas, [_instance(blas, offsets[3], custom_indices[3])]) == OK, "trace D: the TLAS rebuilds in the next submission")
	_trace(pipeline, uniform_set, 3)
	rd.submit()
	rd.sync()

	var hits := rd.buffer_get_data(hit_buffer).to_float32_array()
	for trace in TRACES:
		_check_trace(hits, trace, offsets[trace], custom_indices[trace])


func _run_negative() -> void:
	var tlas := _own(rd.tlas_create(1, 0))
	var unbuilt_blas := _own(_create_blas())
	_check(rd.tlas_build(tlas, [_instance(unbuilt_blas, Vector3.ZERO, 0)]) != OK, "a TLAS build rejects a BLAS that was never built")

	var freed_blas := _create_built_blas()
	rd.free_rid(freed_blas)
	_check(rd.tlas_build(tlas, [_instance(freed_blas, Vector3.ZERO, 0)]) != OK, "a TLAS build rejects a freed BLAS")
	rd.submit()
	rd.sync()


func _check_trace(hits: PackedFloat32Array, trace: int, offset: Vector3, custom_index: int) -> void:
	var label: String = ["A", "B", "C", "D"][trace]
	var expected_hits := 0
	var wrong: Array[String] = []
	for y in GRID:
		for x in GRID:
			var point := (Vector2(x, y) + Vector2(0.5, 0.5)) * (2.0 / GRID) - Vector2.ONE
			var expected := _barycentrics(point - Vector2(offset.x, offset.y))
			var i := (trace * RAYS + y * GRID + x) * 4
			var got := Vector4(hits[i], hits[i + 1], hits[i + 2], hits[i + 3])
			var inside := expected.x >= 0.0 and expected.y >= 0.0 and expected.x + expected.y <= 1.0
			var want := Vector4(1.0, expected.x, expected.y, custom_index) if inside else Vector4(-1.0, -1.0, -1.0, -1.0)
			if inside:
				expected_hits += 1
			var diff := (got - want).abs()
			if maxf(maxf(diff.x, diff.y), maxf(diff.z, diff.w)) > TOLERANCE:
				wrong.push_back("(%d, %d) got %s, want %s" % [x, y, got, want])
	_check(expected_hits > 0 and expected_hits < RAYS, "trace %s: the grid has both hits and misses (%d hits)" % [label, expected_hits])
	_check(wrong.is_empty(), "trace %s: every ray's hit, distance, barycentrics and custom index match the CPU" % label)
	for line in wrong.slice(0, 8):
		printerr("  ", line)


# Barycentrics (weights of the second and third corners) of a model-space point, as rayQueryGetIntersectionBarycentricsEXT returns them.
func _barycentrics(point: Vector2) -> Vector2:
	var e1 := TRIANGLE[1] - TRIANGLE[0]
	var e2 := TRIANGLE[2] - TRIANGLE[0]
	var d := point - TRIANGLE[0]
	var det := e1.cross(e2)
	return Vector2(d.cross(e2), e1.cross(d)) / det


func _trace(pipeline: RID, uniform_set: RID, trace: int) -> void:
	var compute_list := rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list, pipeline)
	rd.compute_list_bind_uniform_set(compute_list, uniform_set, 0)
	var params := PackedInt32Array([trace]).to_byte_array()
	rd.compute_list_set_push_constant(compute_list, params, params.size())
	rd.compute_list_dispatch(compute_list, 1, 1, 1)
	rd.compute_list_end()


func _own(rid: RID) -> RID:
	owned.push_back(rid)
	return rid


func _create_blas() -> RID:
	var vertices := PackedFloat32Array()
	for corner in TRIANGLE:
		vertices.append(corner.x)
		vertices.append(corner.y)
		vertices.append(0.0)
	var vertex_buffer := _own(rd.vertex_buffer_create(vertices.size() * 4, vertices.to_byte_array(),
			RenderingDevice.BUFFER_CREATION_DEVICE_ADDRESS_BIT | RenderingDevice.BUFFER_CREATION_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT))
	var geometry := RDAccelerationStructureGeometry.new()
	geometry.flags = RenderingDevice.ACCELERATION_STRUCTURE_GEOMETRY_OPAQUE_BIT
	geometry.vertex_buffer = vertex_buffer
	geometry.vertex_format = RenderingDevice.DATA_FORMAT_R32G32B32_SFLOAT
	geometry.vertex_stride = 12
	geometry.vertex_count = TRIANGLE.size()
	return rd.blas_create([geometry], 0)


func _create_built_blas() -> RID:
	var blas := _create_blas()
	_check(blas.is_valid() and rd.blas_build(blas) == OK, "the one-triangle BLAS was created and built")
	return blas


# hit_sbt_range stays at its default of 0: the instance has no hit shaders.
func _instance(blas: RID, offset: Vector3, custom_index: int) -> RDAccelerationStructureInstance:
	var instance := RDAccelerationStructureInstance.new()
	instance.blas = blas
	instance.transform = Transform3D(Basis.IDENTITY, offset)
	instance.id = custom_index
	return instance


func _create_shader(compute_source: String) -> RID:
	var source := RDShaderSource.new()
	source.language = RenderingDevice.SHADER_LANGUAGE_GLSL
	source.source_compute = compute_source
	var spirv := rd.shader_compile_spirv_from_source(source)
	if spirv.compile_error_compute != "":
		_fail("shader compile error in the compute stage: %s" % spirv.compile_error_compute)
	return rd.shader_create_from_spirv(spirv)


func _create_set(shader: RID, tlas: RID, hit_buffer: RID) -> RID:
	var tlas_uniform := RDUniform.new()
	tlas_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_ACCELERATION_STRUCTURE
	tlas_uniform.binding = 0
	tlas_uniform.add_id(tlas)
	var hit_uniform := RDUniform.new()
	hit_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	hit_uniform.binding = 1
	hit_uniform.add_id(hit_buffer)
	return rd.uniform_set_create([tlas_uniform, hit_uniform], shader, 0)


func _check(condition: bool, what: String) -> void:
	if condition:
		print("ok: ", what)
	else:
		_fail(what)


func _fail(what: String) -> void:
	failures += 1
	printerr("FAILED: ", what)
