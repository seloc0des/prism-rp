extends PrismBenchmarkRig

var rotating_louver_rig: Node3D

func _init() -> void:
	base_camera_position = Vector3(0.0, 1.6, 5.6)
	camera_look_target = Vector3(0.0, 1.5, 0.0)
	camera_fov = 42.0
	orbit_amplitude = Vector3(0.28, 0.08, 0.22)
	orbit_frequency_hz = 0.30

func _build_benchmark_scene() -> void:
	var dark_floor := create_pbr_material(Color(0.07, 0.08, 0.10), 0.35, 0.2)
	var steel_wire := create_pbr_material(Color(0.88, 0.90, 0.94), 0.06, 1.0)
	var copper_slat := create_pbr_material(Color(0.94, 0.56, 0.34), 0.10, 1.0)

	add_box(self, "TestDeck", Vector3(14.0, 0.2, 14.0), Vector3(0.0, -0.1, 0.0), dark_floor)

	var grazing_sun := DirectionalLight3D.new()
	grazing_sun.name = "GrazingSpecularSun"
	grazing_sun.rotation_degrees = Vector3(-18.0, -142.0, 0.0)
	grazing_sun.light_color = Color(1.0, 0.97, 0.92)
	grazing_sun.light_energy = 4.5
	grazing_sun.shadow_enabled = true
	add_child(grazing_sun)

	var wire_cage := Node3D.new()
	wire_cage.name = "SubPixelWireGrid"
	wire_cage.position = Vector3(-1.35, 1.5, 0.2)
	add_child(wire_cage)

	for i in range(36):
		var offset := -1.1 + float(i) * (2.2 / 35.0)
		add_box(wire_cage, "VWire_%d" % i, Vector3(0.005, 2.4, 0.005), Vector3(offset, 0.0, 0.0), steel_wire)
		add_box(wire_cage, "HWire_%d" % i, Vector3(2.4, 0.005, 0.005), Vector3(0.0, offset, 0.02), steel_wire)

	rotating_louver_rig = Node3D.new()
	rotating_louver_rig.name = "SpecularLouverArray"
	rotating_louver_rig.position = Vector3(1.35, 1.5, -0.1)
	add_child(rotating_louver_rig)

	for i in range(44):
		var y_pos := -1.15 + float(i) * (2.3 / 43.0)
		add_box(
			rotating_louver_rig,
			"Louver_%d" % i,
			Vector3(2.2, 0.008, 0.08),
			Vector3(0.0, y_pos, 0.0),
			copper_slat,
			Vector3(28.0, 0.0, 0.0)
		)

	for c in range(16):
		var t := float(c) / 15.0
		add_box(
			self,
			"TensionCable_%d" % c,
			Vector3(0.004, 4.6, 0.004),
			Vector3(lerpf(-2.6, 2.6, t), 1.6, -1.6),
			steel_wire,
			Vector3(0.0, 0.0, lerpf(-34.0, 34.0, t))
		)

func _step_benchmark_frame(frame: int) -> void:
	if rotating_louver_rig != null:
		var t := float(frame) / 60.0
		rotating_louver_rig.rotation_degrees.y = sin(t * 1.8) * 12.0
