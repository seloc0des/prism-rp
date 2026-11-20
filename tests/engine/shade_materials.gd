extends SceneTree
# Dispatches the Track B material shading pass (vis_tile_classify, then vis_shade_materials) through
# RenderingDevice on the custom engine build. Run through tests/engine/run_shade_materials.sh.
#
# vis_shade_materials indexes a runtime-sized sampler2D array with nonuniformEXT, so the device must have
# enabled the descriptor-indexing features (patch 0005). The scene is two instances of one triangle that
# covers the whole 32x32 target, a material each, both unshaded: the left half of the visibility buffer
# belongs to instance 0 and the right half to instance 1. The textures are three 1x1 colours, and the
# materials point at texture 2 (blue) and texture 1 (green) so a wrong index shows up as a wrong colour.
#
# u_textures[] has a fixed size (PRISM_MAX_MATERIAL_TEXTURES), so the uniform set binds every slot: the
# three colours first, the rest filled with the first texture as a placeholder.
#
# Pass the shader directory (addons/prism_rp/shaders) as --shader-dir=<path>.

const SIZE := 32
const TILE := 8
const MATERIAL_FLAG_UNSHADED := 1 << 2
const INVALID_TEX := 0xFFFFFFFF
const MAX_MATERIAL_TEXTURES := 256  # PRISM_MAX_MATERIAL_TEXTURES in vis_shading_common.glsl

var rd: RenderingDevice
var failures := 0
var shader_dir := ""
var owned: Array[RID] = []  # freed in reverse order, or the validation layers report leaks


func _init() -> void:
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--shader-dir="):
			shader_dir = argument.trim_prefix("--shader-dir=")
	if shader_dir == "":
		_fail("--shader-dir=<path to addons/prism_rp/shaders> is required")
		quit(1)
		return

	rd = RenderingServer.create_local_rendering_device()
	if rd == null:
		_fail("no local RenderingDevice (is the renderer running with Vulkan?)")
		quit(1)
		return

	_check(rd.has_feature(RenderingDevice.SUPPORTS_NONUNIFORM_SAMPLED_IMAGE_INDEXING), "SUPPORTS_NONUNIFORM_SAMPLED_IMAGE_INDEXING is true")
	if failures == 0:
		_run()

	owned.reverse()
	for rid in owned:
		rd.free_rid(rid)
	rd.free()
	print("SHADE_MATERIALS_TEST: ", "PASS" if failures == 0 else "FAIL")
	quit(0 if failures == 0 else 1)


func _run() -> void:
	var classify := _own(_create_compute_shader("visibility/vis_tile_classify.comp.glsl"))
	var shade := _own(_create_compute_shader("visibility/vis_shade_materials.comp.glsl"))
	if not classify.is_valid() or not shade.is_valid():
		_fail("the Track B shaders did not create")
		return
	var classify_pipeline := _own(rd.compute_pipeline_create(classify))
	var shade_pipeline := _own(rd.compute_pipeline_create(shade))
	_check(classify_pipeline.is_valid() and shade_pipeline.is_valid(), "both compute pipelines were created")
	if failures > 0:
		return

	# One triangle that covers the whole target in NDC; view_proj is the identity, so world space is NDC.
	var positions := PackedFloat32Array([-1, -1, 0.5, 1, 3, -1, 0.5, 1, -1, 3, 0.5, 1])
	var attributes := PackedFloat32Array()
	for corner in 3:
		attributes.append_array([0, 0, 1, 0,  1, 0, 0, 1,  0.5, 0.5, 0, 0])  # normal, tangent, uv
	var meshlet := _u32s([0, 0, 3, 1]) + _f32s([0, 0, 0, 4,  0, 0, 0, 0,  0, 0, 1, 1])
	var instances := _instance(0) + _instance(1)
	var materials := _material(2) + _material(1)

	var identity := _f32s([1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1])
	var ubo := identity + identity + identity + _f32s([0, 0, 3, 1,  0, 0, 1, 1,  1, 1, 1, 0]) + _u32s([SIZE, SIZE, SIZE / TILE, 64])

	var vis := PackedByteArray()
	for y in SIZE:
		for x in SIZE:
			# Payload: X = instance, Y = meshlet << 8 | primitive << 1 | alpha-tested.
			vis.append_array(_u32s([0 if x < SIZE / 2 else 1, 0]))

	var vis_texture := _own(_create_texture(RenderingDevice.DATA_FORMAT_R32G32_UINT, SIZE, SIZE, RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT, vis))
	var hdr := _own(_create_texture(RenderingDevice.DATA_FORMAT_R16G16B16A16_SFLOAT, SIZE, SIZE, RenderingDevice.TEXTURE_USAGE_STORAGE_BIT | RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT))
	var albedo := _own(_create_texture(RenderingDevice.DATA_FORMAT_R8G8B8A8_UNORM, SIZE, SIZE, RenderingDevice.TEXTURE_USAGE_STORAGE_BIT))
	var normal := _own(_create_texture(RenderingDevice.DATA_FORMAT_R16G16B16A16_SFLOAT, SIZE, SIZE, RenderingDevice.TEXTURE_USAGE_STORAGE_BIT))
	var motion := _own(_create_texture(RenderingDevice.DATA_FORMAT_R16G16_SFLOAT, SIZE, SIZE, RenderingDevice.TEXTURE_USAGE_STORAGE_BIT))
	var colours := [Color.RED, Color.GREEN, Color.BLUE]
	var textures: Array[RID] = []
	for colour in colours:
		var texel := PackedByteArray([int(colour.r * 255), int(colour.g * 255), int(colour.b * 255), 255])
		textures.push_back(_own(_create_texture(RenderingDevice.DATA_FORMAT_R8G8B8A8_UNORM, 1, 1, RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT, texel)))

	var sampler_state := RDSamplerState.new()
	sampler_state.min_filter = RenderingDevice.SAMPLER_FILTER_NEAREST
	sampler_state.mag_filter = RenderingDevice.SAMPLER_FILTER_NEAREST
	var sampler := _own(rd.sampler_create(sampler_state))

	var texture_ids: Array = []
	for slot in MAX_MATERIAL_TEXTURES:
		texture_ids.append_array([sampler, textures[slot] if slot < textures.size() else textures[0]])

	var ubo_buffer := _own(rd.uniform_buffer_create(ubo.size(), ubo))
	var meshlet_buffer := _own(rd.storage_buffer_create(meshlet.size(), meshlet))
	var instance_buffer := _own(rd.storage_buffer_create(instances.size(), instances))
	var position_buffer := _own(rd.storage_buffer_create(positions.size() * 4, positions.to_byte_array()))
	var attribute_buffer := _own(rd.storage_buffer_create(attributes.size() * 4, attributes.to_byte_array()))
	var vertex_index_buffer := _own(rd.storage_buffer_create(12, _u32s([0, 1, 2])))
	var triangle_buffer := _own(rd.storage_buffer_create(4, _u32s([0 | (1 << 8) | (2 << 16)])))
	var material_buffer := _own(rd.storage_buffer_create(materials.size(), materials))
	var work_buffer := _own(rd.storage_buffer_create(64 * 8))
	var dispatch_buffer := _own(rd.storage_buffer_create(16, _u32s([256, 0, 1, 0]), RenderingDevice.STORAGE_BUFFER_USAGE_DISPATCH_INDIRECT))

	var classify_set := _own(rd.uniform_set_create([
		_uniform(RenderingDevice.UNIFORM_TYPE_UNIFORM_BUFFER, 0, [ubo_buffer]),
		_uniform(RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 1, [sampler, vis_texture]),
		_uniform(RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER, 2, [instance_buffer]),
		_uniform(RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER, 3, [work_buffer]),
		_uniform(RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER, 4, [dispatch_buffer]),
	], classify, 0))

	var shade_set := _own(rd.uniform_set_create([
		_uniform(RenderingDevice.UNIFORM_TYPE_UNIFORM_BUFFER, 0, [ubo_buffer]),
		_uniform(RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 1, [sampler, vis_texture]),
		_uniform(RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER, 2, [meshlet_buffer]),
		_uniform(RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER, 3, [instance_buffer]),
		_uniform(RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER, 4, [position_buffer]),
		_uniform(RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER, 5, [attribute_buffer]),
		_uniform(RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER, 6, [vertex_index_buffer]),
		_uniform(RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER, 7, [triangle_buffer]),
		_uniform(RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER, 8, [material_buffer]),
		_uniform(RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER, 9, [work_buffer]),
		_uniform(RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 10, texture_ids),
		_uniform(RenderingDevice.UNIFORM_TYPE_IMAGE, 11, [hdr]),
		_uniform(RenderingDevice.UNIFORM_TYPE_IMAGE, 12, [albedo]),
		_uniform(RenderingDevice.UNIFORM_TYPE_IMAGE, 13, [normal]),
		_uniform(RenderingDevice.UNIFORM_TYPE_IMAGE, 14, [motion]),
		_uniform(RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER, 15, [dispatch_buffer]),
	], shade, 0))
	_check(classify_set.is_valid() and shade_set.is_valid(), "the uniform sets, with the 256-slot texture array at binding 10, were created")
	if failures > 0:
		return

	var list := rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(list, classify_pipeline)
	rd.compute_list_bind_uniform_set(list, classify_set, 0)
	rd.compute_list_dispatch(list, SIZE / TILE, SIZE / TILE, 1)
	rd.compute_list_add_barrier(list)
	rd.compute_list_bind_compute_pipeline(list, shade_pipeline)
	rd.compute_list_bind_uniform_set(list, shade_set, 0)
	rd.compute_list_dispatch_indirect(list, dispatch_buffer, 0)
	rd.compute_list_end()
	rd.submit()
	rd.sync()

	# Each tile holds one material, so the classify pass queues 16 work items.
	var dispatch := rd.buffer_get_data(dispatch_buffer)
	_check(dispatch.decode_u32(12) == 16, "the classify pass queued one work item per tile (got %d)" % dispatch.decode_u32(12))

	var out := rd.texture_get_data(hdr, 0)
	var wrong := 0
	for y in SIZE:
		for x in SIZE:
			var offset := (y * SIZE + x) * 8
			var got := Color(out.decode_half(offset), out.decode_half(offset + 2), out.decode_half(offset + 4), out.decode_half(offset + 6))
			var want: Color = colours[2] if x < SIZE / 2 else colours[1]
			if not got.is_equal_approx(Color(want.r, want.g, want.b, 1.0)):
				wrong += 1
				if wrong <= 3:
					print("pixel ", x, ",", y, " is ", got, ", expected ", want)
	_check(wrong == 0, "every pixel took the colour of its material's texture (left blue, right green): %d wrong of %d" % [wrong, SIZE * SIZE])


func _instance(material_id: int) -> PackedByteArray:
	var identity := _f32s([1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1])
	# world_from_local, prev_world_from_local, then meshlet_offset, meshlet_count, vertex_base, material_id, flags.
	return identity + identity + _u32s([0, 1, 0, material_id, 1, 0, 0, 0])


# An unshaded material whose albedo is texture p_albedo_texture.
func _material(p_albedo_texture: int) -> PackedByteArray:
	return _f32s([1, 1, 1, 1,  0, 0, 0, 1,  0, 1, 0.5, 1]) + _u32s([p_albedo_texture, INVALID_TEX, INVALID_TEX, INVALID_TEX, MATERIAL_FLAG_UNSHADED, 0, 0, 0])


func _uniform(type: int, binding: int, ids: Array) -> RDUniform:
	var uniform := RDUniform.new()
	uniform.uniform_type = type
	uniform.binding = binding
	for id in ids:
		uniform.add_id(id)
	return uniform


func _create_texture(format: int, width: int, height: int, usage: int, data := PackedByteArray()) -> RID:
	var texture_format := RDTextureFormat.new()
	texture_format.format = format
	texture_format.width = width
	texture_format.height = height
	texture_format.usage_bits = usage
	return rd.texture_create(texture_format, RDTextureView.new(), [data] if data.size() > 0 else [])


func _create_compute_shader(relative_path: String) -> RID:
	var source := RDShaderSource.new()
	source.language = RenderingDevice.SHADER_LANGUAGE_GLSL
	source.source_compute = _resolve_includes(shader_dir.path_join(relative_path))
	var spirv := rd.shader_compile_spirv_from_source(source)
	if spirv.compile_error_compute != "":
		_fail("%s: %s" % [relative_path, spirv.compile_error_compute])
		return RID()
	return rd.shader_create_from_spirv(spirv)


# Godot's glslang has no file includes, so inline them the way the addon does at runtime.
func _resolve_includes(path: String) -> String:
	var lines := PackedStringArray()
	for line in FileAccess.get_file_as_string(path).split("\n"):
		var trimmed := line.strip_edges()
		if trimmed.begins_with("#extension GL_GOOGLE_include_directive"):
			continue
		if trimmed.begins_with("#include"):
			var name := trimmed.split("\"")[1]
			var local := path.get_base_dir().path_join(name)
			lines.push_back(_resolve_includes(local if FileAccess.file_exists(local) else shader_dir.path_join("include").path_join(name)))
		else:
			lines.push_back(line)
	return "\n".join(lines)


func _u32s(values: Array) -> PackedByteArray:
	var bytes := PackedByteArray()
	bytes.resize(values.size() * 4)
	for i in values.size():
		bytes.encode_u32(i * 4, values[i])
	return bytes


func _f32s(values: Array) -> PackedByteArray:
	var bytes := PackedByteArray()
	bytes.resize(values.size() * 4)
	for i in values.size():
		bytes.encode_float(i * 4, values[i])
	return bytes


func _own(rid: RID) -> RID:
	owned.push_back(rid)
	return rid


func _check(condition: bool, description: String) -> void:
	if condition:
		print("ok: ", description)
	else:
		_fail(description)


func _fail(description: String) -> void:
	failures += 1
	printerr("FAILED: ", description)
