extends PrismBenchmarkRig

var atrium_sun: DirectionalLight3D
var dynamic_lights: Array[OmniLight3D] = []
var center_banner: MeshInstance3D

func _init() -> void:
	base_camera_position = Vector3(0.0, 2.2, 8.2)
	camera_look_target = Vector3(0.0, 2.4, -2.0)
	camera_fov = 52.0
	orbit_amplitude = Vector3(0.45, 0.15, 0.35)
	orbit_frequency_hz = 0.25

func _build_benchmark_scene() -> void:
	var stone_limestone := create_pbr_material(Color(0.74, 0.69, 0.60), 0.78, 0.0)
	var stone_trim := create_pbr_material(Color(0.56, 0.51, 0.44), 0.68, 0.0)
	var terracotta_floor := create_pbr_material(Color(0.52, 0.34, 0.26), 0.42, 0.05)
	var crimson_silk := create_pbr_material(Color(0.68, 0.12, 0.14), 0.62, 0.0)
	var cobalt_silk := create_pbr_material(Color(0.12, 0.26, 0.66), 0.62, 0.0)

	add_box(self, "CourtyardFloor", Vector3(12.0, 0.2, 20.0), Vector3(0.0, -0.1, 0.0), terracotta_floor)
	add_box(self, "LeftGalleryFloor", Vector3(3.0, 0.25, 20.0), Vector3(-4.5, 3.6, 0.0), stone_trim)
	add_box(self, "RightGalleryFloor", Vector3(3.0, 0.25, 20.0), Vector3(4.5, 3.6, 0.0), stone_trim)
	add_box(self, "LeftOuterWall", Vector3(0.4, 7.5, 20.0), Vector3(-6.0, 3.75, 0.0), stone_limestone)
	add_box(self, "RightOuterWall", Vector3(0.4, 7.5, 20.0), Vector3(6.0, 3.75, 0.0), stone_limestone)
	add_box(self, "EndFacadeWall", Vector3(12.4, 7.5, 0.5), Vector3(0.0, 3.75, -9.8), stone_limestone)

	for row in range(6):
		var z_pos := -7.5 + float(row) * 3.0
		for side: float in [-1.0, 1.0]:
			var x_pos := side * 3.2
			add_box(self, "LowerCol_%d_%d" % [row, int(side)], Vector3(0.55, 3.4, 0.55), Vector3(x_pos, 1.7, z_pos), stone_limestone)
			add_box(self, "LowerCap_%d_%d" % [row, int(side)], Vector3(0.75, 0.22, 0.75), Vector3(x_pos, 3.45, z_pos), stone_trim)
			add_box(self, "UpperCol_%d_%d" % [row, int(side)], Vector3(0.45, 3.2, 0.45), Vector3(x_pos, 5.3, z_pos), stone_limestone)

		var banner_mat := crimson_silk if (row % 2 == 0) else cobalt_silk
		var banner := add_box(self, "AtriumBanner_%d" % row, Vector3(0.04, 2.6, 1.05), Vector3(-2.6 if row % 2 == 0 else 2.6, 2.2, z_pos), banner_mat)
		if row == 2:
			center_banner = banner

	add_box(self, "LeftUpperLintel", Vector3(0.7, 0.5, 19.5), Vector3(-3.2, 7.1, 0.0), stone_trim)
	add_box(self, "RightUpperLintel", Vector3(0.7, 0.5, 19.5), Vector3(3.2, 7.1, 0.0), stone_trim)

	atrium_sun = DirectionalLight3D.new()
	atrium_sun.name = "AtriumSun"
	atrium_sun.rotation_degrees = Vector3(-56.0, -34.0, 0.0)
	atrium_sun.light_color = Color(1.0, 0.94, 0.84)
	atrium_sun.light_energy = 3.4
	atrium_sun.shadow_enabled = true
	add_child(atrium_sun)

	var light_colors := [
		Color(1.0, 0.35, 0.12),
		Color(0.18, 0.65, 1.0),
		Color(0.95, 0.85, 0.35),
	]
	for i in range(3):
		var omni := OmniLight3D.new()
		omni.name = "DynamicAtriumLight_%d" % i
		omni.light_color = light_colors[i]
		omni.light_energy = 3.2
		omni.omni_range = 8.5
		omni.shadow_enabled = (i == 0)
		add_child(omni)
		dynamic_lights.append(omni)

func _step_benchmark_frame(frame: int) -> void:
	var t := float(frame) / 60.0
	if atrium_sun != null:
		atrium_sun.rotation_degrees.y = -34.0 + sin(t * 0.9) * 14.0

	for i in range(dynamic_lights.size()):
		var phase := t * 1.6 + float(i) * (TAU / 3.0)
		dynamic_lights[i].position = Vector3(
			cos(phase) * 2.35,
			1.65 + sin(phase * 1.5) * 0.55,
			-1.5 + sin(phase) * 4.2
		)

	if center_banner != null:
		center_banner.rotation_degrees.z = sin(t * 3.2) * 4.5
