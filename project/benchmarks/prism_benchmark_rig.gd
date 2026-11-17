class_name PrismBenchmarkRig
extends Node3D

@export var base_camera_position := Vector3(0.0, 1.6, 4.5)
@export var camera_look_target := Vector3(0.0, 1.2, 0.0)
@export var camera_fov := 50.0
@export var orbit_amplitude := Vector3(0.35, 0.12, 0.25)
@export var orbit_frequency_hz := 0.35

var camera: Camera3D
var world_environment: WorldEnvironment
var chronos_effect: CompositorEffect

## Sub-pixel camera jitter for Chronos TAA. Godot only jitters when its own TAA/FSR2 is on,
## and those run after compositor effects, so the rig jitters through the camera frustum and
## Chronos reads the offset back from the projection matrix.
@export var jitter_enabled := true
const JITTER_PHASES := 8

var frame_index: int = 0
var active_tier: String = "tier2"
var render_scale: float = 0.75
var qa_capture_frame: int = -1
var qa_telemetry_out: String = ""
var qa_screenshot_out: String = ""
var quit_after_frame: int = -1

func _ready() -> void:
	_parse_cmdline_args()
	_configure_viewport_scaling()
	_setup_environment_and_chronos_vpu()
	_setup_camera_rig()
	_build_benchmark_scene()
	_update_deterministic_camera(0)

func _process(_delta: float) -> void:
	_update_deterministic_camera(frame_index)
	_step_benchmark_frame(frame_index)

	if frame_index == qa_capture_frame:
		_capture_qa_frame()

	if quit_after_frame > 0 and frame_index >= quit_after_frame:
		if qa_capture_frame < 0:
			_export_telemetry()
		get_tree().quit(0)
		return

	frame_index += 1

func _get_telemetry() -> Object:
	if Engine.has_singleton(&"PrismTelemetry"):
		return Engine.get_singleton(&"PrismTelemetry")
	return null

func _export_telemetry() -> void:
	var telemetry := _get_telemetry()
	if telemetry != null and not qa_telemetry_out.is_empty():
		telemetry.call(&"export_json", qa_telemetry_out)

## Saves the displayed frame and the GPU telemetry once the capture frame has been drawn.
## With RenderDoc attached, it also captures the next frame via the in-app API.
func _capture_qa_frame() -> void:
	var telemetry := _get_telemetry()
	if telemetry != null and telemetry.call(&"is_renderdoc_attached"):
		telemetry.call(&"trigger_renderdoc_capture")

	await RenderingServer.frame_post_draw

	if not qa_screenshot_out.is_empty():
		var img := get_viewport().get_texture().get_image()
		DirAccess.make_dir_recursive_absolute(qa_screenshot_out.get_base_dir())
		var err := img.save_png(qa_screenshot_out)
		if err != OK:
			push_error("[Prism RP QA] Failed to save screenshot to %s (error %d)" % [qa_screenshot_out, err])
	_export_telemetry()

func _build_benchmark_scene() -> void:
	pass

func _step_benchmark_frame(_frame: int) -> void:
	pass

func _parse_cmdline_args() -> void:
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--prism-tier="):
			active_tier = arg.get_slice("=", 1)
		elif arg.begins_with("--prism-render-scale="):
			render_scale = clampf(arg.get_slice("=", 1).to_float(), 0.25, 2.0)
		elif arg.begins_with("--qa-capture-frame="):
			qa_capture_frame = arg.get_slice("=", 1).to_int()
		elif arg.begins_with("--qa-telemetry-out="):
			qa_telemetry_out = arg.get_slice("=", 1)
		elif arg.begins_with("--qa-screenshot-out="):
			qa_screenshot_out = arg.get_slice("=", 1)
		elif arg.begins_with("--quit-after="):
			quit_after_frame = arg.get_slice("=", 1).to_int()
		elif arg == "--prism-no-jitter":
			jitter_enabled = false

func _configure_viewport_scaling() -> void:
	var vp := get_viewport()
	if vp == null:
		return
	vp.scaling_3d_mode = Viewport.SCALING_3D_MODE_BILINEAR
	vp.scaling_3d_scale = render_scale
	vp.use_taa = false

func _setup_environment_and_chronos_vpu() -> void:
	var env := Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color(0.015, 0.018, 0.024)
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color = Color(0.08, 0.09, 0.11)
	env.ambient_light_energy = 0.35
	# Chronos' SDR look writes display-referred linear values; Godot must then only apply
	# the sRGB encode, which is what the Linear tonemapper at exposure 1.0 does.
	env.tonemap_mode = Environment.TONE_MAPPER_LINEAR
	env.tonemap_exposure = 1.0
	env.glow_enabled = false
	env.sdfgi_enabled = true
	env.sdfgi_use_occlusion = true

	var compositor := Compositor.new()
	if ClassDB.class_exists(&"ChronosVPUCompositorEffect"):
		var instance: Variant = ClassDB.instantiate(&"ChronosVPUCompositorEffect")
		if instance is CompositorEffect:
			chronos_effect = instance as CompositorEffect
			chronos_effect.set(&"dilate_velocity_enabled", true)
			chronos_effect.set(&"reactive_mask_enabled", true)
			chronos_effect.set(&"taau_enabled", true)
			chronos_effect.set(&"fft_bloom_enabled", true)
			chronos_effect.set(&"aces_odt_enabled", true)
			chronos_effect.set(&"output_display_mode", 0)
			compositor.compositor_effects = [chronos_effect]
	else:
		push_warning("[Prism RP QA] ChronosVPUCompositorEffect class not loaded; running baseline Forward+.")

	world_environment = WorldEnvironment.new()
	world_environment.name = "BenchmarkWorldEnvironment"
	world_environment.environment = env
	world_environment.compositor = compositor
	add_child(world_environment)

func _setup_camera_rig() -> void:
	camera = Camera3D.new()
	camera.name = "BenchmarkCamera"
	camera.fov = camera_fov
	camera.near = 0.05
	camera.far = 250.0
	camera.current = true
	if jitter_enabled and chronos_effect != null:
		# Frustum projection so frustum_offset can carry the sub-pixel jitter. With the default
		# KEEP_HEIGHT, `size` is the near-plane height; matching it to the FOV keeps the framing.
		camera.projection = Camera3D.PROJECTION_FRUSTUM
		camera.size = 2.0 * camera.near * tan(deg_to_rad(camera_fov) * 0.5)
	add_child(camera)

func _update_deterministic_camera(frame: int) -> void:
	if camera == null:
		return
	var t := float(frame) / 60.0
	var phase := TAU * orbit_frequency_hz * t
	var offset := Vector3(
		sin(phase) * orbit_amplitude.x,
		sin(phase * 0.5) * orbit_amplitude.y,
		(cos(phase) - 1.0) * orbit_amplitude.z
	)
	camera.position = base_camera_position + offset
	camera.look_at(camera_look_target, Vector3.UP)
	if camera.projection == Camera3D.PROJECTION_FRUSTUM:
		camera.frustum_offset = _jitter_frustum_offset(frame)

## Halton(2, 3) sub-pixel jitter in internal pixels, converted to near-plane world units.
func _jitter_frustum_offset(frame: int) -> Vector2:
	var phase := (frame % JITTER_PHASES) + 1
	var jitter_px := Vector2(_halton(phase, 2) - 0.5, _halton(phase, 3) - 0.5)
	var internal_height := maxf(get_viewport().get_visible_rect().size.y * render_scale, 1.0)
	return jitter_px * (camera.size / internal_height)

func _halton(index: int, base: int) -> float:
	var result := 0.0
	var f := 1.0
	var i := index
	while i > 0:
		f /= float(base)
		result += f * float(i % base)
		@warning_ignore("integer_division")
		i = i / base
	return result

func create_pbr_material(
	albedo: Color,
	roughness: float = 0.5,
	metallic: float = 0.0,
	emission: Color = Color.BLACK,
	emission_energy: float = 0.0
) -> StandardMaterial3D:
	var mat := StandardMaterial3D.new()
	mat.albedo_color = albedo
	mat.roughness = roughness
	mat.metallic = metallic
	if emission_energy > 0.0:
		mat.emission_enabled = true
		mat.emission = emission
		mat.emission_energy_multiplier = emission_energy
	return mat

func add_box(
	parent: Node3D,
	box_name: String,
	size: Vector3,
	pos: Vector3,
	mat: Material,
	rot_deg: Vector3 = Vector3.ZERO
) -> MeshInstance3D:
	var mesh := BoxMesh.new()
	mesh.size = size
	var mi := MeshInstance3D.new()
	mi.name = box_name
	mi.mesh = mesh
	mi.material_override = mat
	mi.position = pos
	mi.rotation_degrees = rot_deg
	parent.add_child(mi)
	return mi

func add_sphere(
	parent: Node3D,
	sphere_name: String,
	radius: float,
	pos: Vector3,
	mat: Material,
	rings: int = 32,
	segments: int = 64
) -> MeshInstance3D:
	var mesh := SphereMesh.new()
	mesh.radius = radius
	mesh.height = radius * 2.0
	mesh.rings = rings
	mesh.radial_segments = segments
	var mi := MeshInstance3D.new()
	mi.name = sphere_name
	mi.mesh = mesh
	mi.material_override = mat
	mi.position = pos
	parent.add_child(mi)
	return mi
