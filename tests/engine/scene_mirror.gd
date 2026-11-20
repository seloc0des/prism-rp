extends SceneTree
# Checks Lumina's scene mirror (LuminaSceneMirror) on the custom engine build. Run through
# tests/engine/run_scene_mirror.sh, from the repository root with the addon linked into project/ and imported.
#
# Lumina's "Ray-Traced Albedo" debug view traces one camera ray per pixel through the mirror and writes the albedo
# it hits, so it must match Godot's own unshaded view of the same scene pixel for pixel. That covers what the mirror
# holds (meshes, transforms, UVs, textures, colours, materials and their overrides, culling, alpha scissor) and what
# it leaves out (hidden and shadow-only meshes). Meshes with gi_mode disabled are left out of the mirror on purpose:
# where the unshaded view draws one, the ray-traced view keeps Godot's lit pixel instead of writing its albedo.
#
# Phase A captures both views of the scene as built. Phase B moves and resizes the box, frees the sphere, and adds a
# cylinder and a dense sphere big enough to make the mirror grow its buffers while the other meshes are live, then
# captures both again: the mirror has to follow the scene tree. Phase C leaves the effect off long enough for the
# mirror to release everything, turns it back on and compares with phase B's unshaded capture: the mirror has to
# start again from nothing.

const SIZE := Vector2i(640, 360)
const SETTLE_FRAMES := 8
const TOLERANCE := 6            # per 8-bit channel
const MAX_MISMATCH := 0.0003    # share of compared pixels allowed to differ (silhouette rounding), about 69
const SPOT_TOLERANCE := 4
const IDLE_FRAMES := 150        # more than LuminaSceneMirror::kIdleFramesBeforeRelease (120)

enum Step { SETTLE_RT_A, SETTLE_UNSHADED_A, SETTLE_RT_B, SETTLE_UNSHADED_B, IDLE, SETTLE_RT_C }

var failures := 0
var step := Step.SETTLE_RT_A
var frames := 0
var effect: CompositorEffect
var camera: Camera3D
var box: MeshInstance3D
var sphere: MeshInstance3D
var gi_disabled: MeshInstance3D
var sphere_pixel: Vector2i
var ray_traced: Image
var unshaded_b: Image


func _initialize() -> void:
	root.size = SIZE
	effect = ClassDB.instantiate("LuminaCascadesCompositorEffect") as CompositorEffect
	if effect == null:
		_fail("LuminaCascadesCompositorEffect is missing: is the Prism RP extension loaded?")
		quit(1)
		return
	effect.set("debug_view_mode", 8)
	_build_scene()


func _process(_delta: float) -> bool:
	if effect == null:
		return true
	frames += 1
	if step == Step.SETTLE_RT_A and frames >= SETTLE_FRAMES:
		ray_traced = _capture()
		_show_unshaded(true)
		_next(Step.SETTLE_UNSHADED_A)
	elif step == Step.SETTLE_UNSHADED_A and frames >= SETTLE_FRAMES:
		_compare("A", ray_traced, _capture())
		sphere_pixel = Vector2i(camera.unproject_position(sphere.global_position))
		_change_scene()
		_show_unshaded(false)
		_next(Step.SETTLE_RT_B)
	elif step == Step.SETTLE_RT_B and frames >= SETTLE_FRAMES:
		ray_traced = _capture()
		_show_unshaded(true)
		_next(Step.SETTLE_UNSHADED_B)
	elif step == Step.SETTLE_UNSHADED_B and frames >= SETTLE_FRAMES:
		unshaded_b = _capture()
		_compare("B", ray_traced, unshaded_b)
		_next(Step.IDLE)
	elif step == Step.IDLE and frames >= IDLE_FRAMES:
		_show_unshaded(false)
		_next(Step.SETTLE_RT_C)
	elif step == Step.SETTLE_RT_C and frames >= SETTLE_FRAMES:
		_compare("C", _capture(), unshaded_b)
		print("SCENE_MIRROR_TEST: ", "PASS" if failures == 0 else "FAIL")
		quit(0 if failures == 0 else 1)
	return false


func _next(next_step: Step) -> void:
	step = next_step
	frames = 0


func _show_unshaded(unshaded: bool) -> void:
	effect.enabled = not unshaded
	root.debug_draw = Viewport.DEBUG_DRAW_UNSHADED if unshaded else Viewport.DEBUG_DRAW_DISABLED


func _capture() -> Image:
	var image := root.get_texture().get_image()
	image.convert(Image.FORMAT_RGB8)
	return image


func _build_scene() -> void:
	var environment := Environment.new()
	environment.background_mode = Environment.BG_COLOR
	environment.background_color = Color.BLACK
	environment.tonemap_mode = Environment.TONE_MAPPER_LINEAR
	var compositor := Compositor.new()
	compositor.compositor_effects = [effect]
	var world_environment := WorldEnvironment.new()
	world_environment.environment = environment
	world_environment.compositor = compositor
	root.add_child(world_environment)

	# Lit, the frame differs from the unshaded one, so a debug view that never ran can't pass.
	var sun := DirectionalLight3D.new()
	sun.light_energy = 0.3
	sun.rotation_degrees = Vector3(-50.0, 30.0, 0.0)
	root.add_child(sun)

	camera = Camera3D.new()
	root.add_child(camera)
	camera.look_at_from_position(Vector3(0.0, 4.0, 7.0), Vector3(0.0, 0.8, 0.0))

	var floor_material := StandardMaterial3D.new()
	floor_material.albedo_texture = _checker_texture()
	floor_material.uv1_scale = Vector3(2.0, 2.0, 1.0)
	var plane := PlaneMesh.new()
	plane.size = Vector2(8.0, 8.0)
	_add_mesh(plane, Vector3.ZERO, floor_material)

	box = _add_mesh(BoxMesh.new(), Vector3(-2.0, 0.6, 0.0), _color_material(Color(0.8, 0.3, 0.1)))
	box.rotation_degrees.y = 30.0
	box.scale = Vector3(1.0, 1.2, 1.0)

	sphere = _add_mesh(SphereMesh.new(), Vector3(2.0, 0.8, 0.0), null)
	sphere.material_override = _color_material(Color(0.2, 0.7, 0.3))

	var two_surfaces := _add_mesh(_two_surface_mesh(), Vector3(0.0, 1.6, -2.0), null)
	two_surfaces.set_surface_override_material(1, _color_material(Color(0.15, 0.25, 0.9)))

	# Quads face +Z, toward the camera; turned around, only the double-sided one shows its back.
	var one_sided := _add_mesh(QuadMesh.new(), Vector3(-1.2, 2.6, 1.0), _color_material(Color(0.9, 0.9, 0.1)))
	one_sided.rotation_degrees.y = 180.0
	var two_sided_material := _color_material(Color(0.8, 0.1, 0.7))
	two_sided_material.cull_mode = BaseMaterial3D.CULL_DISABLED
	var two_sided := _add_mesh(QuadMesh.new(), Vector3(1.2, 2.6, 1.0), two_sided_material)
	two_sided.rotation_degrees.y = 180.0

	var cutout_material := _color_material(Color(1.0, 0.55, 0.1))
	cutout_material.albedo_texture = _stripe_alpha_texture()
	cutout_material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA_SCISSOR
	cutout_material.alpha_scissor_threshold = 0.5
	_add_mesh(QuadMesh.new(), Vector3(0.0, 2.6, 1.0), cutout_material)

	var hidden := _add_mesh(BoxMesh.new(), Vector3(0.0, 0.4, 1.5), _color_material(Color.WHITE))
	hidden.visible = false
	var shadow_only := _add_mesh(BoxMesh.new(), Vector3(-0.8, 0.4, 2.0), _color_material(Color.WHITE))
	shadow_only.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_SHADOWS_ONLY

	gi_disabled = _add_mesh(BoxMesh.new(), Vector3(3.0, 2.4, -1.5), _color_material(Color(0.9, 0.9, 0.9)))
	gi_disabled.scale = Vector3(0.6, 0.6, 0.6)
	gi_disabled.gi_mode = GeometryInstance3D.GI_MODE_DISABLED


func _change_scene() -> void:
	box.position = Vector3(-1.5, 0.6, 1.0)
	(box.mesh as BoxMesh).size = Vector3(1.4, 1.0, 1.0)   # its "changed" signal: the mirror reads it again
	sphere.queue_free()
	var cylinder := CylinderMesh.new()
	cylinder.top_radius = 0.4
	cylinder.bottom_radius = 0.6
	_add_mesh(cylinder, Vector3(2.4, 1.0, 1.8), _color_material(Color(0.9, 0.85, 0.2)))
	# About 197k triangle corners, ten times what the mirror's first buffers hold.
	var dense := SphereMesh.new()
	dense.radial_segments = 256
	dense.rings = 128
	_add_mesh(dense, Vector3(-2.8, 0.5, -1.5), _color_material(Color(0.6, 0.4, 0.8)))


func _add_mesh(mesh: Mesh, position: Vector3, material: Material) -> MeshInstance3D:
	var instance := MeshInstance3D.new()
	instance.mesh = mesh
	instance.position = position
	if material != null:
		instance.set_surface_override_material(0, material)
	root.add_child(instance)
	return instance


func _color_material(color: Color) -> StandardMaterial3D:
	var material := StandardMaterial3D.new()
	material.albedo_color = color
	return material


# Two side-by-side quads, one surface each. The left one has vertex colours and a material that uses them.
func _two_surface_mesh() -> ArrayMesh:
	var mesh := ArrayMesh.new()
	var colors: Array[Color] = [Color(1.0, 0.2, 0.2), Color(0.2, 1.0, 0.2), Color(0.2, 0.2, 1.0), Color(1.0, 1.0, 0.2)]
	for side in 2:
		var tool := SurfaceTool.new()
		tool.begin(Mesh.PRIMITIVE_TRIANGLES)
		var x0 := -1.0 + side * 1.05
		var corners: Array[Vector3] = [Vector3(x0, -0.5, 0.0), Vector3(x0 + 0.95, -0.5, 0.0), Vector3(x0 + 0.95, 0.5, 0.0), Vector3(x0, 0.5, 0.0)]
		# Clockwise seen from +Z, Godot's front-face winding.
		for corner in [0, 3, 2, 0, 2, 1]:
			tool.set_normal(Vector3.BACK)
			tool.set_color(colors[corner])
			tool.add_vertex(corners[corner])
		tool.commit(mesh)
	var vertex_colored := StandardMaterial3D.new()
	vertex_colored.vertex_color_use_as_albedo = true
	mesh.surface_set_material(0, vertex_colored)
	return mesh


func _checker_texture() -> ImageTexture:
	var image := Image.create_empty(8, 8, false, Image.FORMAT_RGBA8)
	for y in 8:
		for x in 8:
			image.set_pixel(x, y, Color8(200, 60, 40) if (x + y) % 2 == 0 else Color8(40, 120, 220))
	image.generate_mipmaps()
	return ImageTexture.create_from_image(image)


# Vertical stripes two texels wide, alternately opaque and clear.
func _stripe_alpha_texture() -> ImageTexture:
	var image := Image.create_empty(8, 8, false, Image.FORMAT_RGBA8)
	for y in 8:
		for x in 8:
			image.set_pixel(x, y, Color(1.0, 1.0, 1.0, 1.0 if ((x >> 1) & 1) == 0 else 0.0))
	image.generate_mipmaps()
	return ImageTexture.create_from_image(image)


func _compare(phase: String, traced: Image, unshaded: Image) -> void:
	var exclude := _screen_rect(gi_disabled)
	var compared := 0
	var mismatched := 0
	var first_bad: Array[String] = []
	for y in traced.get_height():
		for x in traced.get_width():
			if exclude.has_point(Vector2i(x, y)):
				continue
			compared += 1
			var a := traced.get_pixel(x, y)
			var b := unshaded.get_pixel(x, y)
			var diff := maxf(maxf(absf(a.r - b.r), absf(a.g - b.g)), absf(a.b - b.b)) * 255.0
			if diff > TOLERANCE:
				mismatched += 1
				if first_bad.size() < 8:
					first_bad.push_back("(%d, %d) traced %s, unshaded %s" % [x, y, a.to_html(false), b.to_html(false)])
	var share := float(mismatched) / float(maxi(compared, 1))
	_check(share <= MAX_MISMATCH, "phase %s: the ray-traced albedo matches Godot's unshaded view (%d of %d pixels differ)" % [phase, mismatched, compared])
	for line in first_bad:
		printerr("  ", line)

	var box_pixel := Vector2i(camera.unproject_position(box.global_position))
	_check_color("phase %s: the box shows its sRGB albedo" % phase, traced, box_pixel, Color(0.8, 0.3, 0.1))
	var gi_pixel := Vector2i(camera.unproject_position(gi_disabled.global_position))
	var lit := traced.get_pixelv(gi_pixel)
	_check(absf(lit.r - 0.9) * 255.0 > 8.0 * SPOT_TOLERANCE,
			"phase %s: the gi_mode-disabled box is left out of the mirror (pixel %s keeps Godot's lit %s)" % [phase, gi_pixel, lit.to_html(false)])
	_check_color("phase %s: the unshaded view draws the gi_mode-disabled box" % phase, unshaded, gi_pixel, Color(0.9, 0.9, 0.9))
	if phase != "A":
		var where := traced.get_pixelv(sphere_pixel)
		_check(absf(where.g - 0.7) * 255.0 > 8.0 * SPOT_TOLERANCE,
				"phase %s: the freed sphere is gone (pixel %s shows %s)" % [phase, sphere_pixel, where.to_html(false)])
	_check_save_on_failure(phase, traced, unshaded)


func _check_color(what: String, image: Image, pixel: Vector2i, expected: Color) -> void:
	var got := image.get_pixelv(pixel)
	var diff := maxf(maxf(absf(got.r - expected.r), absf(got.g - expected.g)), absf(got.b - expected.b)) * 255.0
	_check(diff <= SPOT_TOLERANCE, "%s (pixel %s: %s, want %s)" % [what, pixel, got.to_html(false), expected.to_html(false)])


# Screen-space bounds of an instance's AABB, grown by two pixels.
func _screen_rect(instance: MeshInstance3D) -> Rect2i:
	var aabb := instance.global_transform * instance.get_aabb()
	var rect := Rect2(camera.unproject_position(aabb.position), Vector2.ZERO)
	for i in 8:
		rect = rect.expand(camera.unproject_position(aabb.get_endpoint(i)))
	return Rect2i(rect.grow(2.0))


func _check_save_on_failure(phase: String, traced: Image, unshaded: Image) -> void:
	var dir := OS.get_environment("SCENE_MIRROR_OUT")
	if failures > 0 and dir != "":
		traced.save_png(dir.path_join("ray_traced_%s.png" % phase))
		unshaded.save_png(dir.path_join("unshaded_%s.png" % phase))


func _check(condition: bool, what: String) -> void:
	if condition:
		print("ok: ", what)
	else:
		_fail(what)


func _fail(what: String) -> void:
	failures += 1
	printerr("FAILED: ", what)
