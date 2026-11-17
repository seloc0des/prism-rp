extends PrismBenchmarkRig

var hero_cube: MeshInstance3D
var glossy_sphere: MeshInstance3D

func _init() -> void:
	base_camera_position = Vector3(0.0, 1.5, 4.35)
	camera_look_target = Vector3(0.0, 1.45, 0.0)
	camera_fov = 45.0
	orbit_amplitude = Vector3(0.18, 0.06, 0.12)
	orbit_frequency_hz = 0.25

func _build_benchmark_scene() -> void:
	var white_wall := create_pbr_material(Color(0.76, 0.75, 0.72), 0.85, 0.0)
	var red_wall := create_pbr_material(Color(0.68, 0.11, 0.09), 0.85, 0.0)
	var green_wall := create_pbr_material(Color(0.12, 0.58, 0.16), 0.85, 0.0)
	var chrome_mat := create_pbr_material(Color(0.92, 0.93, 0.95), 0.08, 1.0)
	var gold_gloss := create_pbr_material(Color(0.95, 0.72, 0.24), 0.22, 1.0)
	var ceiling_panel := create_pbr_material(Color.WHITE, 0.9, 0.0, Color(1.0, 0.94, 0.84), 8.5)

	add_box(self, "Floor", Vector3(3.0, 0.1, 3.0), Vector3(0.0, -0.05, 0.0), white_wall)
	add_box(self, "Ceiling", Vector3(3.0, 0.1, 3.0), Vector3(0.0, 3.05, 0.0), white_wall)
	add_box(self, "BackWall", Vector3(3.0, 3.0, 0.1), Vector3(0.0, 1.5, -1.55), white_wall)
	add_box(self, "LeftWallRed", Vector3(0.1, 3.0, 3.0), Vector3(-1.55, 1.5, 0.0), red_wall)
	add_box(self, "RightWallGreen", Vector3(0.1, 3.0, 3.0), Vector3(1.55, 1.5, 0.0), green_wall)
	add_box(self, "CeilingEmitter", Vector3(0.95, 0.04, 0.95), Vector3(0.0, 2.98, 0.0), ceiling_panel)

	add_box(self, "TallBlock", Vector3(0.78, 1.55, 0.78), Vector3(-0.55, 0.775, -0.45), white_wall, Vector3(0.0, 22.0, 0.0))
	hero_cube = add_box(self, "HeroCube", Vector3(0.68, 0.68, 0.68), Vector3(0.52, 0.34, 0.35), gold_gloss, Vector3(0.0, -18.0, 0.0))
	glossy_sphere = add_sphere(self, "ChromeSphere", 0.38, Vector3(-0.52, 1.93, -0.45), chrome_mat)

	var key_light := OmniLight3D.new()
	key_light.name = "CornellCeilingLight"
	key_light.position = Vector3(0.0, 2.78, 0.0)
	key_light.light_color = Color(1.0, 0.95, 0.88)
	key_light.light_energy = 4.2
	key_light.omni_range = 7.5
	key_light.omni_attenuation = 1.2
	key_light.shadow_enabled = true
	add_child(key_light)

func _step_benchmark_frame(frame: int) -> void:
	var t := float(frame) / 60.0
	if hero_cube != null:
		hero_cube.rotation_degrees.y = -18.0 + sin(t * 2.4) * 14.0
	if glossy_sphere != null:
		glossy_sphere.position.y = 1.93 + sin(t * 3.0) * 0.05
