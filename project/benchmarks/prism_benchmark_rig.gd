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
var lumina_effect: CompositorEffect
var opaque_capture_effect: CompositorEffect

## Which Prism RP effects to build (QA flags: --prism-chronos=off, --prism-lumina=off,
## --prism-opaque-capture=off, --prism-sdfgi=on|off). SDFGI defaults to off when Lumina runs,
## so the two GIs don't stack.
var use_chronos := true
var use_lumina := true
var use_opaque_capture := true
var sdfgi_override := ""
## --prism-godot-taa=on enables Godot's own TAA, to exercise Chronos' double-TAA guard.
var godot_taa := false
## Property overrides: --prism-set=chronos.taau_enabled=false,lumina.debug_view_mode=2
var property_overrides: Array[Dictionary] = []

## Sub-pixel camera jitter for Chronos TAA, supplied by the addon's ChronosJitterCamera.
@export var jitter_enabled := true
const JITTER_CAMERA_SCRIPT := "res://addons/prism_rp/scripts/chronos_jitter_camera.gd"

var frame_index: int = 0
var active_tier: String = "tier2"
var render_scale: float = 0.75
var qa_capture_frame: int = -1
var qa_telemetry_out: String = ""
var qa_screenshot_out: String = ""
## --qa-scene-export=<dir>: write the rendered scene (triangles, materials, lights, camera) for the
## Gate 2 Cycles reference. --prism-ambient=off zeroes ambient light (indirect-only comparisons).
var qa_scene_export: String = ""
var ambient_enabled := true
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
	if not qa_scene_export.is_empty():
		var size := Vector2i(get_viewport().get_visible_rect().size)
		var export_err := PrismSceneExport.export_scene(self, camera, world_environment.environment, qa_scene_export, size)
		if export_err != OK:
			push_error("[Prism RP QA] Scene export to %s failed (error %d)" % [qa_scene_export, export_err])
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
		elif arg.begins_with("--qa-scene-export="):
			qa_scene_export = arg.get_slice("=", 1)
		elif arg.begins_with("--prism-ambient="):
			ambient_enabled = _flag_value(arg)
		elif arg.begins_with("--quit-after="):
			quit_after_frame = arg.get_slice("=", 1).to_int()
		elif arg == "--prism-no-jitter":
			jitter_enabled = false
		elif arg.begins_with("--prism-chronos="):
			use_chronos = _flag_value(arg)
		elif arg.begins_with("--prism-lumina="):
			use_lumina = _flag_value(arg)
		elif arg.begins_with("--prism-opaque-capture="):
			use_opaque_capture = _flag_value(arg)
		elif arg.begins_with("--prism-godot-taa="):
			godot_taa = _flag_value(arg)
		elif arg.begins_with("--prism-sdfgi="):
			sdfgi_override = arg.get_slice("=", 1).to_lower()
		elif arg.begins_with("--prism-set="):
			_parse_property_overrides(arg.substr("--prism-set=".length()))

func _flag_value(arg: String) -> bool:
	var value := arg.get_slice("=", 1).to_lower()
	return not (value == "off" or value == "0" or value == "false")

## "chronos.taau_enabled=false,lumina.intensity=1.5" -> [{target, property, value}, ...]
func _parse_property_overrides(spec: String) -> void:
	for entry in spec.split(",", false):
		var eq := entry.find("=")
		var dot := entry.find(".")
		if eq < 0 or dot < 0 or dot > eq:
			push_error("[Prism RP QA] Bad --prism-set entry '%s' (want effect.property=value)" % entry)
			continue
		property_overrides.append({
			"target": entry.substr(0, dot),
			"property": entry.substr(dot + 1, eq - dot - 1),
			"value": _parse_value(entry.substr(eq + 1)),
		})

func _parse_value(text: String) -> Variant:
	var lower := text.to_lower()
	if lower == "true" or lower == "on":
		return true
	if lower == "false" or lower == "off":
		return false
	if text.is_valid_int():
		return text.to_int()
	if text.is_valid_float():
		return text.to_float()
	return text

func _apply_property_overrides() -> void:
	var targets := {
		"chronos": chronos_effect,
		"lumina": lumina_effect,
		"capture": opaque_capture_effect,
	}
	for o in property_overrides:
		var target: Object = targets.get(o["target"])
		if target == null:
			push_warning("[Prism RP QA] --prism-set target '%s' is not active" % o["target"])
			continue
		var prop: StringName = o["property"]
		if not (prop in target):
			push_error("[Prism RP QA] %s has no property '%s'" % [o["target"], prop])
			continue
		target.set(prop, o["value"])

func _configure_viewport_scaling() -> void:
	var vp := get_viewport()
	if vp == null:
		return
	vp.scaling_3d_mode = Viewport.SCALING_3D_MODE_BILINEAR
	vp.scaling_3d_scale = render_scale
	vp.use_taa = godot_taa

func _setup_environment_and_chronos_vpu() -> void:
	var env := Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color(0.015, 0.018, 0.024)
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color = Color(0.08, 0.09, 0.11)
	env.ambient_light_energy = 0.35 if ambient_enabled else 0.0
	# Chronos' SDR look writes display-referred linear values; Godot must then only apply
	# the sRGB encode, which is what the Linear tonemapper at exposure 1.0 does.
	env.tonemap_mode = Environment.TONE_MAPPER_LINEAR
	env.tonemap_exposure = 1.0
	env.glow_enabled = false

	var effects: Array[CompositorEffect] = []
	if not ClassDB.class_exists(&"ChronosVPUCompositorEffect"):
		push_warning("[Prism RP QA] Prism RP extension not loaded; running baseline Forward+.")
	else:
		# Callback types order the effects (Lumina POST_OPAQUE, capture PRE_TRANSPARENT,
		# Chronos POST_TRANSPARENT); array order doesn't matter.
		if use_lumina:
			lumina_effect = ClassDB.instantiate(&"LuminaCascadesCompositorEffect") as CompositorEffect
			effects.append(lumina_effect)
		if use_chronos:
			chronos_effect = ClassDB.instantiate(&"ChronosVPUCompositorEffect") as CompositorEffect
			chronos_effect.set(&"dilate_velocity_enabled", true)
			chronos_effect.set(&"reactive_mask_enabled", true)
			chronos_effect.set(&"taau_enabled", true)
			chronos_effect.set(&"fft_bloom_enabled", true)
			chronos_effect.set(&"aces_odt_enabled", true)
			chronos_effect.set(&"output_display_mode", 0)
			effects.append(chronos_effect)
			# Opaque-only capture lets Chronos flag particles and glass (no depth or motion
			# vectors) in its reactive mask.
			if use_opaque_capture:
				opaque_capture_effect = ClassDB.instantiate(&"ChronosOpaqueCaptureEffect") as CompositorEffect
				effects.append(opaque_capture_effect)
		_apply_property_overrides()

	var sdfgi_on := lumina_effect == null
	if sdfgi_override == "on":
		sdfgi_on = true
	elif sdfgi_override == "off":
		sdfgi_on = false
	env.sdfgi_enabled = sdfgi_on
	env.sdfgi_use_occlusion = true

	var compositor := Compositor.new()
	compositor.compositor_effects = effects

	world_environment = WorldEnvironment.new()
	world_environment.name = "BenchmarkWorldEnvironment"
	world_environment.environment = env
	world_environment.compositor = compositor
	add_child(world_environment)

func _setup_camera_rig() -> void:
	var use_jitter := jitter_enabled and chronos_effect != null and ResourceLoader.exists(JITTER_CAMERA_SCRIPT)
	if use_jitter:
		camera = (load(JITTER_CAMERA_SCRIPT) as GDScript).new() as Camera3D
	else:
		camera = Camera3D.new()
	camera.name = "BenchmarkCamera"
	camera.fov = camera_fov
	camera.near = 0.05
	camera.far = 250.0
	camera.current = true
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
