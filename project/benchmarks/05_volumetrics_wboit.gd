extends PrismBenchmarkRig

var spark_emitter: GPUParticles3D
var emissive_orb: MeshInstance3D

func _init() -> void:
	base_camera_position = Vector3(0.0, 1.75, 5.4)
	camera_look_target = Vector3(0.0, 1.55, 0.0)
	camera_fov = 48.0
	orbit_amplitude = Vector3(0.32, 0.10, 0.25)
	orbit_frequency_hz = 0.35

func _build_benchmark_scene() -> void:
	if world_environment != null and world_environment.environment != null:
		var env := world_environment.environment
		env.volumetric_fog_enabled = true
		env.volumetric_fog_density = 0.045
		env.volumetric_fog_albedo = Color(0.86, 0.90, 0.96)
		env.volumetric_fog_emission = Color(0.02, 0.03, 0.05)
		env.volumetric_fog_anisotropy = 0.62

	var obsidian_stone := create_pbr_material(Color(0.12, 0.13, 0.16), 0.32, 0.15)
	add_box(self, "SanctuaryFloor", Vector3(14.0, 0.2, 14.0), Vector3(0.0, -0.1, 0.0), obsidian_stone)
	add_box(self, "BackArchWall", Vector3(10.0, 6.0, 0.4), Vector3(0.0, 3.0, -3.2), obsidian_stone)

	for b in range(7):
		var x_pos := -3.6 + float(b) * 1.2
		add_box(self, "RoofBeam_%d" % b, Vector3(0.65, 0.35, 8.0), Vector3(x_pos, 4.4, 0.0), obsidian_stone)

	var shaft_sun := DirectionalLight3D.new()
	shaft_sun.name = "VolumetricShaftSun"
	shaft_sun.rotation_degrees = Vector3(-58.0, 24.0, 0.0)
	shaft_sun.light_color = Color(1.0, 0.93, 0.80)
	shaft_sun.light_energy = 5.2
	shaft_sun.light_volumetric_fog_energy = 2.4
	shaft_sun.shadow_enabled = true
	add_child(shaft_sun)

	var glass_specs := [
		{ "name": "GlassAmber", "color": Color(1.0, 0.58, 0.14, 0.42), "pos": Vector3(-0.45, 1.55, 0.95), "rot": 8.0 },
		{ "name": "GlassCyan",  "color": Color(0.14, 0.82, 0.98, 0.42), "pos": Vector3(0.05, 1.55, 0.35),  "rot": -6.0 },
		{ "name": "GlassRuby",  "color": Color(0.96, 0.16, 0.28, 0.42), "pos": Vector3(0.45, 1.55, -0.25), "rot": 10.0 },
	]
	for spec in glass_specs:
		var glass_mat := StandardMaterial3D.new()
		glass_mat.albedo_color = spec["color"]
		glass_mat.roughness = 0.08
		glass_mat.metallic = 0.05
		glass_mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
		glass_mat.cull_mode = BaseMaterial3D.CULL_DISABLED
		add_box(self, spec["name"], Vector3(1.45, 2.2, 0.03), spec["pos"], glass_mat, Vector3(0.0, spec["rot"], 0.0))

	var core_mat := create_pbr_material(Color.WHITE, 0.1, 0.0, Color(1.0, 0.42, 0.12), 18.0)
	emissive_orb = add_sphere(self, "EmissiveHalationCore", 0.22, Vector3(0.0, 1.45, 1.55), core_mat)

	var proc_mat := ParticleProcessMaterial.new()
	proc_mat.direction = Vector3(0.0, 1.0, 0.0)
	proc_mat.spread = 38.0
	proc_mat.initial_velocity_min = 3.2
	proc_mat.initial_velocity_max = 5.4
	proc_mat.gravity = Vector3(0.0, -6.5, 0.0)
	proc_mat.scale_min = 0.025
	proc_mat.scale_max = 0.055

	var spark_mesh := QuadMesh.new()
	spark_mesh.size = Vector2(0.06, 0.06)
	var spark_mat := StandardMaterial3D.new()
	# BILLBOARD_PARTICLES keeps the per-particle scale from the process material.
	spark_mat.billboard_mode = BaseMaterial3D.BILLBOARD_PARTICLES
	# HDR sparks via emission (unshaded materials ignore emission; scaling the albedo
	# Color by 14 also scaled its alpha).
	spark_mat.albedo_color = Color.BLACK
	spark_mat.emission_enabled = true
	spark_mat.emission = Color(1.0, 0.65, 0.22)
	spark_mat.emission_energy_multiplier = 14.0

	spark_emitter = GPUParticles3D.new()
	spark_emitter.name = "HighVelocitySparkFountain"
	spark_emitter.position = Vector3(0.0, 0.35, 1.55)
	spark_emitter.amount = 240
	spark_emitter.lifetime = 1.2
	spark_emitter.fixed_fps = 60
	spark_emitter.interpolate = false
	spark_emitter.randomness = 0.0
	# Same particles in every run: Gate 1 compares a supersampled reference capture with a test
	# capture of the same frame, and differently scattered sparks would dominate the FLIP score.
	spark_emitter.use_fixed_seed = true
	spark_emitter.seed = 1337
	spark_emitter.process_material = proc_mat
	spark_emitter.draw_pass_1 = spark_mesh
	spark_emitter.material_override = spark_mat
	add_child(spark_emitter)

func _step_benchmark_frame(frame: int) -> void:
	if emissive_orb != null:
		var t := float(frame) / 60.0
		emissive_orb.position.x = sin(t * 2.6) * 0.85
		emissive_orb.position.y = 1.45 + cos(t * 3.4) * 0.22
