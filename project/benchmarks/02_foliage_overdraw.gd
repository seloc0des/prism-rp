extends PrismBenchmarkRig

var foliage_multimesh: MultiMeshInstance3D
var foliage_transforms: Array[Transform3D] = []

func _init() -> void:
	base_camera_position = Vector3(0.0, 1.85, 5.2)
	camera_look_target = Vector3(0.0, 1.35, 0.0)
	camera_fov = 48.0
	orbit_amplitude = Vector3(0.45, 0.14, 0.30)
	orbit_frequency_hz = 0.40

func _build_benchmark_scene() -> void:
	var ground_mat := create_pbr_material(Color(0.14, 0.18, 0.11), 0.92, 0.0)
	add_box(self, "ForestFloor", Vector3(16.0, 0.2, 16.0), Vector3(0.0, -0.1, 0.0), ground_mat)

	var sun := DirectionalLight3D.new()
	sun.name = "CanopySun"
	sun.rotation_degrees = Vector3(-48.0, 32.0, 0.0)
	sun.light_color = Color(1.0, 0.96, 0.88)
	sun.light_energy = 2.8
	sun.shadow_enabled = true
	add_child(sun)

	var leaf_tex := _generate_serrated_leaf_texture(128)
	var leaf_mat := StandardMaterial3D.new()
	leaf_mat.albedo_texture = leaf_tex
	leaf_mat.albedo_color = Color(0.28, 0.56, 0.19)
	leaf_mat.roughness = 0.48
	leaf_mat.metallic = 0.0
	leaf_mat.cull_mode = BaseMaterial3D.CULL_DISABLED
	leaf_mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA_SCISSOR
	leaf_mat.alpha_scissor_threshold = 0.45
	leaf_mat.backlight_enabled = true
	leaf_mat.backlight = Color(0.32, 0.48, 0.14)

	var card_mesh := QuadMesh.new()
	card_mesh.size = Vector2(0.85, 1.05)

	const INSTANCE_COUNT := 480
	var mm := MultiMesh.new()
	mm.transform_format = MultiMesh.TRANSFORM_3D
	mm.mesh = card_mesh
	mm.instance_count = INSTANCE_COUNT

	var rng := RandomNumberGenerator.new()
	rng.seed = 202602

	foliage_transforms.resize(INSTANCE_COUNT)
	for i in range(INSTANCE_COUNT):
		var layer := float(i % 12) / 12.0
		var radius := rng.randf_range(0.15, 2.8)
		var angle := rng.randf_range(0.0, TAU)
		var pos := Vector3(
			cos(angle) * radius,
			0.35 + layer * 2.25 + rng.randf_range(-0.15, 0.15),
			sin(angle) * radius * 0.85
		)
		var basis := Basis.from_euler(Vector3(
			rng.randf_range(-0.55, 0.55),
			rng.randf_range(0.0, TAU),
			rng.randf_range(-0.45, 0.45)
		))
		var scale_val := rng.randf_range(0.75, 1.35)
		var xform := Transform3D(basis.scaled(Vector3.ONE * scale_val), pos)
		foliage_transforms[i] = xform
		mm.set_instance_transform(i, xform)

	foliage_multimesh = MultiMeshInstance3D.new()
	foliage_multimesh.name = "OverdrawFoliageCluster"
	foliage_multimesh.multimesh = mm
	foliage_multimesh.material_override = leaf_mat
	add_child(foliage_multimesh)

func _step_benchmark_frame(frame: int) -> void:
	if foliage_multimesh == null:
		return
	var t := float(frame) / 60.0
	var mm := foliage_multimesh.multimesh
	for i in range(foliage_transforms.size()):
		var base_xf := foliage_transforms[i]
		var gust := sin(t * 4.2 + base_xf.origin.x * 1.7 + base_xf.origin.y * 1.1) * 0.085
		var swayed_basis := base_xf.basis.rotated(Vector3.FORWARD, gust)
		mm.set_instance_transform(i, Transform3D(swayed_basis, base_xf.origin))

func _generate_serrated_leaf_texture(size: int) -> ImageTexture:
	var img := Image.create(size, size, false, Image.FORMAT_RGBA8)
	for y in range(size):
		var v := float(y) / float(size - 1)
		for x in range(size):
			var u := (float(x) / float(size - 1)) * 2.0 - 1.0
			var profile := sin(v * PI) * 0.82
			var serration := sin(v * PI * 14.0) * 0.09
			var inside := absf(u) <= (profile + serration) and v > 0.04 and v < 0.96
			var vein := 1.0 - clampf(absf(u) * 14.0, 0.0, 1.0) * 0.25
			var shade := clampf(0.65 + 0.35 * v, 0.0, 1.0) * vein
			img.set_pixel(x, y, Color(shade, shade * 1.08, shade * 0.75, 1.0 if inside else 0.0))
	return ImageTexture.create_from_image(img)
