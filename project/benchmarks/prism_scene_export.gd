class_name PrismSceneExport
extends RefCounted
## Writes the benchmark scene as rendered (world-space triangles, materials, lights, camera,
## environment) to <dir>/scene.json plus PNG textures, for the Gate 2 path-traced reference
## (tests/reference/cycles_reference.py). Exporting the real triangles means Blender rebuilds
## exactly what Godot drew, with no second copy of the scene definitions to keep in sync.
## Not exported: particles and volumetric fog (scene 05 isn't part of Gate 2).

const FORMAT_VERSION := 1

static func export_scene(root: Node, camera: Camera3D, environment: Environment, out_dir: String, resolution: Vector2i) -> Error:
	DirAccess.make_dir_recursive_absolute(out_dir)
	var data := {
		"version": FORMAT_VERSION,
		"resolution": [resolution.x, resolution.y],
		"camera": _camera(camera),
		"environment": _environment(environment),
		"meshes": [],
		"lights": [],
	}
	var textures := {}
	_walk(root, data, textures, out_dir)
	var file := FileAccess.open(out_dir.path_join("scene.json"), FileAccess.WRITE)
	if file == null:
		return FileAccess.get_open_error()
	file.store_string(JSON.stringify(data))
	return OK

static func _walk(node: Node, data: Dictionary, textures: Dictionary, out_dir: String) -> void:
	if node is MeshInstance3D and (node as MeshInstance3D).visible:
		var mi := node as MeshInstance3D
		if mi.mesh != null:
			for s in range(mi.mesh.get_surface_count()):
				var mat := mi.material_override if mi.material_override != null else mi.get_active_material(s)
				data["meshes"].append(_surface(mi.name, mi.mesh, s, mi.global_transform, mat, textures, out_dir))
	elif node is MultiMeshInstance3D and (node as MultiMeshInstance3D).visible:
		var mmi := node as MultiMeshInstance3D
		var mm := mmi.multimesh
		if mm != null and mm.mesh != null:
			for i in range(mm.instance_count):
				var xf := mmi.global_transform * mm.get_instance_transform(i)
				for s in range(mm.mesh.get_surface_count()):
					var mat := mmi.material_override if mmi.material_override != null else mm.mesh.surface_get_material(s)
					data["meshes"].append(_surface("%s_%d" % [mmi.name, i], mm.mesh, s, xf, mat, textures, out_dir))
	elif node is OmniLight3D and (node as OmniLight3D).visible:
		var omni := node as OmniLight3D
		data["lights"].append({
			"type": "omni",
			"position": _v3(omni.global_position),
			"color": _color(omni.light_color),
			"energy": omni.light_energy,
			"range": omni.omni_range,
			"attenuation": omni.omni_attenuation,
			"shadow": omni.shadow_enabled,
		})
	elif node is DirectionalLight3D and (node as DirectionalLight3D).visible:
		var sun := node as DirectionalLight3D
		data["lights"].append({
			"type": "directional",
			"direction": _v3(-sun.global_transform.basis.z.normalized()),
			"color": _color(sun.light_color),
			"energy": sun.light_energy,
			"shadow": sun.shadow_enabled,
		})
	for child in node.get_children():
		_walk(child, data, textures, out_dir)

static func _surface(name: String, mesh: Mesh, surface: int, xf: Transform3D, mat: Material, textures: Dictionary, out_dir: String) -> Dictionary:
	var arrays := mesh.surface_get_arrays(surface)
	var verts: PackedVector3Array = arrays[Mesh.ARRAY_VERTEX]
	var normals: PackedVector3Array = arrays[Mesh.ARRAY_NORMAL] if arrays[Mesh.ARRAY_NORMAL] != null else PackedVector3Array()
	var uvs: PackedVector2Array = arrays[Mesh.ARRAY_TEX_UV] if arrays[Mesh.ARRAY_TEX_UV] != null else PackedVector2Array()
	var indices: PackedInt32Array = arrays[Mesh.ARRAY_INDEX] if arrays[Mesh.ARRAY_INDEX] != null else PackedInt32Array()
	if indices.is_empty():
		indices.resize(verts.size())
		for i in range(verts.size()):
			indices[i] = i
	var normal_basis := xf.basis.inverse().transposed()
	var pos := []
	var nrm := []
	var uv := []
	for i in range(verts.size()):
		var p := xf * verts[i]
		pos.append_array([p.x, p.y, p.z])
		if i < normals.size():
			var n := (normal_basis * normals[i]).normalized()
			nrm.append_array([n.x, n.y, n.z])
		if i < uvs.size():
			uv.append_array([uvs[i].x, uvs[i].y])
	return {
		"name": name,
		"positions": pos,
		"normals": nrm,
		"uvs": uv,
		"indices": Array(indices),      # Godot winding: clockwise front faces
		"material": _material(mat, textures, out_dir),
	}

static func _material(mat: Material, textures: Dictionary, out_dir: String) -> Dictionary:
	var m := {
		"albedo": [1.0, 1.0, 1.0, 1.0], "roughness": 1.0, "metallic": 0.0, "emission": [0.0, 0.0, 0.0],
		"albedo_texture": null, "transparency": "opaque", "alpha_scissor": 0.5, "unshaded": false,
	}
	if not (mat is BaseMaterial3D):
		return m
	var bm := mat as BaseMaterial3D
	m["albedo"] = [bm.albedo_color.r, bm.albedo_color.g, bm.albedo_color.b, bm.albedo_color.a]
	m["roughness"] = bm.roughness
	m["metallic"] = bm.metallic
	m["unshaded"] = bm.shading_mode == BaseMaterial3D.SHADING_MODE_UNSHADED
	if bm.emission_enabled:
		var e := bm.emission * bm.emission_energy_multiplier
		m["emission"] = [e.r, e.g, e.b]
	match bm.transparency:
		BaseMaterial3D.TRANSPARENCY_ALPHA:
			m["transparency"] = "alpha"
		BaseMaterial3D.TRANSPARENCY_ALPHA_SCISSOR:
			m["transparency"] = "scissor"
			m["alpha_scissor"] = bm.alpha_scissor_threshold
	if bm.albedo_texture != null:
		var key := bm.albedo_texture.get_instance_id()
		if not textures.has(key):
			var file := "texture_%d.png" % textures.size()
			bm.albedo_texture.get_image().save_png(out_dir.path_join(file))
			textures[key] = file
		m["albedo_texture"] = textures[key]
	return m

static func _camera(camera: Camera3D) -> Dictionary:
	var b := camera.global_transform.basis
	return {
		"basis_columns": [_v3(b.x), _v3(b.y), _v3(b.z)],
		"origin": _v3(camera.global_position),
		"fov_deg": camera.fov,
		"keep_aspect": "width" if camera.keep_aspect == Camera3D.KEEP_WIDTH else "height",
		"near": camera.near,
		"far": camera.far,
	}

static func _environment(env: Environment) -> Dictionary:
	if env == null:
		return {"background": [0.0, 0.0, 0.0], "ambient": [0.0, 0.0, 0.0]}
	var amb := env.ambient_light_color * env.ambient_light_energy
	return {
		"background": _color(env.background_color),
		"ambient": [amb.r, amb.g, amb.b] if env.ambient_light_source == Environment.AMBIENT_SOURCE_COLOR else [0.0, 0.0, 0.0],
	}

static func _v3(v: Vector3) -> Array:
	return [v.x, v.y, v.z]

static func _color(c: Color) -> Array:
	return [c.r, c.g, c.b]
