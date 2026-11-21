extends SceneTree
# Checks Lumina's ray-traced world cache update (lumina_ws_update_rt) on the custom engine build: the cache has to
# light the view with light that was never on screen, taken from the scene mirror's lights and emission. Run through
# tests/engine/run_world_cache_rt.sh, from the repository root with the addon linked into project/ and imported.
#
# The camera looks down at a floor no light reaches. Behind it, out of view, a wall faces it. Lumina's "World Cascade
# Only" debug view shows the irradiance the world cache gives the floor. The depth-buffer march never sees the wall,
# so only the traced update can light the floor:
#   A: a directional light shines on the wall. The floor has to be lit.
#   B: the light's bake mode becomes Disabled, which takes it out of GI. The floor has to go dark, so the light came
#      from the mirror's light table.
#   C: the wall's material starts to emit. The floor has to be lit again, from the mirror's emission.
#   D: the wall stops emitting, the light comes back into GI turned around, so it lights only a card that faces the
#      wall: the wall is lit by the card alone. The floor has to be lit, by light that bounced twice off screen.
#   E: world_multi_bounce_enabled goes off. The wall goes dark and so does the floor: the switch reaches the passes.

const SIZE := Vector2i(320, 180)
const SETTLE_FRAMES := 60
const LIT := 0.2           # mean floor brightness (0..1 in the 8-bit capture) that counts as lit
const DARK_SHARE := 0.15   # phase B's floor at most this share of phase A's, phase E's of phase D's
const BOUNCE_LIT := 0.1    # phase D's floor, lit only by the second bounce

enum Step { SETTLE_A, SETTLE_B, SETTLE_C, SETTLE_D, SETTLE_E }

var failures := 0
var step := Step.SETTLE_A
var frames := 0
var effect: CompositorEffect
var sun: DirectionalLight3D
var wall_material: StandardMaterial3D
var lit_a := 0.0
var lit_d := 0.0


func _initialize() -> void:
	root.size = SIZE
	effect = ClassDB.instantiate("LuminaCascadesCompositorEffect") as CompositorEffect
	if effect == null:
		_fail("LuminaCascadesCompositorEffect is missing: is the Prism RP extension loaded?")
		quit(1)
		return
	if not RenderingServer.get_rendering_device().has_feature(RenderingDevice.SUPPORTS_RAY_QUERY):
		_fail("the GPU has no ray queries, so the traced world cache update can't run")
		effect = null
		quit(1)
		return
	effect.set("debug_view_mode", 7)   # World Cascade Only
	_build_scene()


func _process(_delta: float) -> bool:
	if effect == null:
		return true
	frames += 1
	if frames < SETTLE_FRAMES:
		return false
	var floor_brightness := _floor_brightness()
	match step:
		Step.SETTLE_A:
			lit_a = floor_brightness
			_check(floor_brightness >= LIT, "A: the lit wall behind the camera lights the floor (mean %.4f, want at least %.4f)" % [floor_brightness, LIT])
			sun.light_bake_mode = Light3D.BAKE_DISABLED
			_next(Step.SETTLE_B)
		Step.SETTLE_B:
			var most := lit_a * DARK_SHARE
			_check(floor_brightness <= most, "B: with the light out of GI the floor goes dark (mean %.4f, want at most %.4f)" % [floor_brightness, most])
			wall_material.emission_enabled = true
			_next(Step.SETTLE_C)
		Step.SETTLE_C:
			_check(floor_brightness >= LIT, "C: the glowing wall lights the floor (mean %.4f, want at least %.4f)" % [floor_brightness, LIT])
			wall_material.emission_enabled = false
			sun.light_bake_mode = Light3D.BAKE_DYNAMIC
			sun.rotation_degrees = Vector3.ZERO   # shines along -Z: onto the card's wall side only
			sun.shadow_enabled = false            # the default, stated: the wall stands between the sun and the card
			_add_mesh(_box(Vector3(8.0, 2.0, 0.2)), Vector3(0.0, 4.5, 1.5), _grey())
			_next(Step.SETTLE_D)
		Step.SETTLE_D:
			lit_d = floor_brightness
			_check(floor_brightness >= BOUNCE_LIT, "D: the wall the card lights lights the floor (mean %.4f, want at least %.4f)" % [floor_brightness, BOUNCE_LIT])
			effect.set("world_multi_bounce_enabled", false)
			_next(Step.SETTLE_E)
		Step.SETTLE_E:
			var most := lit_d * DARK_SHARE
			_check(floor_brightness <= most, "E: with one bounce the card's light no longer reaches the floor (mean %.4f, want at most %.4f)" % [floor_brightness, most])
			print("WORLD_CACHE_RT_TEST: ", "PASS" if failures == 0 else "FAIL")
			quit(0 if failures == 0 else 1)
	return false


func _next(next_step: Step) -> void:
	step = next_step
	frames = 0


# Mean brightness of the bottom third of the view, all floor.
func _floor_brightness() -> float:
	var image := root.get_texture().get_image()
	image.convert(Image.FORMAT_RGB8)
	var sum := 0.0
	var count := 0
	for y in range(SIZE.y * 2 / 3, SIZE.y):
		for x in SIZE.x:
			sum += image.get_pixel(x, y).get_luminance()
			count += 1
	return sum / count


func _build_scene() -> void:
	var environment := Environment.new()
	environment.background_mode = Environment.BG_COLOR
	environment.background_color = Color.BLACK
	environment.ambient_light_source = Environment.AMBIENT_SOURCE_DISABLED
	environment.tonemap_mode = Environment.TONE_MAPPER_LINEAR
	var compositor := Compositor.new()
	compositor.compositor_effects = [effect]
	var world_environment := WorldEnvironment.new()
	world_environment.environment = environment
	world_environment.compositor = compositor
	root.add_child(world_environment)

	var camera := Camera3D.new()
	root.add_child(camera)
	camera.look_at_from_position(Vector3(0.0, 1.5, 0.0), Vector3(0.0, 0.0, -6.0))

	var plane := PlaneMesh.new()
	plane.size = Vector2(16.0, 16.0)
	_add_mesh(plane, Vector3.ZERO, _grey())

	wall_material = _grey()
	wall_material.emission = Color.WHITE
	wall_material.emission_energy_multiplier = 1.0
	_add_mesh(_box(Vector3(12.0, 6.0, 0.4)), Vector3(0.0, 3.0, 3.0), wall_material)

	# Turned to shine along +Z: onto the wall's camera side, and along the floor, which it doesn't light.
	sun = DirectionalLight3D.new()
	sun.light_energy = 3.0
	sun.rotation_degrees = Vector3(0.0, 180.0, 0.0)
	root.add_child(sun)


func _grey() -> StandardMaterial3D:
	var material := StandardMaterial3D.new()
	material.albedo_color = Color(0.8, 0.8, 0.8)
	return material


func _box(size: Vector3) -> BoxMesh:
	var box := BoxMesh.new()
	box.size = size
	return box


func _add_mesh(mesh: Mesh, position: Vector3, material: Material) -> MeshInstance3D:
	var instance := MeshInstance3D.new()
	instance.mesh = mesh
	instance.position = position
	instance.set_surface_override_material(0, material)
	root.add_child(instance)
	return instance


func _check(condition: bool, what: String) -> void:
	if condition:
		print("ok: ", what)
	else:
		_fail(what)


func _fail(what: String) -> void:
	failures += 1
	printerr("FAILED: ", what)
