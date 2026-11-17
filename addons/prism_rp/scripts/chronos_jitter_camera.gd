class_name ChronosJitterCamera
extends Camera3D
## Drop-in Camera3D that supplies the sub-pixel jitter Chronos TAA needs.
##
## Godot only jitters the projection when its own TAA or FSR2 is on, and those run after
## compositor effects. This camera jitters through the frustum instead: at runtime it switches
## to PROJECTION_FRUSTUM, keeps the framing of the `fov` you set in the editor, and moves
## `frustum_offset` along a Halton(2, 3) sequence. Chronos reads the offset back from the
## projection matrix of the frame being rendered, so no extra wiring is needed.
##
## Keep Viewport.use_taa off and use a non-temporal scaling_3d_mode (Bilinear or FSR 1);
## Chronos bypasses its TAA while a Godot temporal mode is active.

## Turn jitter on or off at runtime. Off restores a normal perspective projection.
@export var jitter_enabled := true:
	set(value):
		jitter_enabled = value
		if is_inside_tree():
			_apply_projection_mode()

## Length of the Halton sequence. 8 is a good default; longer sequences converge slower.
@export_range(1, 64) var jitter_phases := 8

## Jitter amplitude in internal (render-resolution) pixels. 1.0 spans one full pixel.
@export_range(0.0, 2.0, 0.05) var jitter_scale := 1.0

var _frame := 0

func _ready() -> void:
	# Run after gameplay scripts so this frame's fov/near changes are already in.
	process_priority = 1000
	_apply_projection_mode()
	_update_jitter()

func _process(_delta: float) -> void:
	_frame += 1
	_update_jitter()

## Restart the sequence (e.g. after a camera cut). Pair with ChronosVPUCompositorEffect.reset_taau_history().
func reset_jitter_sequence() -> void:
	_frame = 0

## Current jitter in internal pixels, for debugging or HUDs.
func get_jitter_pixels() -> Vector2:
	if not jitter_enabled:
		return Vector2.ZERO
	var phase := (_frame % maxi(jitter_phases, 1)) + 1
	return Vector2(_halton(phase, 2) - 0.5, _halton(phase, 3) - 0.5) * jitter_scale

func _apply_projection_mode() -> void:
	if jitter_enabled:
		projection = Camera3D.PROJECTION_FRUSTUM
	else:
		projection = Camera3D.PROJECTION_PERSPECTIVE
		frustum_offset = Vector2.ZERO

func _update_jitter() -> void:
	if not jitter_enabled:
		return

	# Match the perspective framing of `fov`. Projection.set_frustum treats `size` as the
	# near-plane height for KEEP_HEIGHT and as the width for KEEP_WIDTH; both are 2*n*tan(fov/2).
	size = 2.0 * near * tan(deg_to_rad(fov) * 0.5)

	var vp := get_viewport()
	if vp == null:
		return
	var view_size := vp.get_visible_rect().size
	var internal_size := view_size * vp.scaling_3d_scale
	if internal_size.x < 1.0 or internal_size.y < 1.0:
		return

	var near_plane_height := size
	if keep_aspect == Camera3D.KEEP_WIDTH:
		near_plane_height = size * (view_size.y / view_size.x)

	# World units per internal pixel at the near plane (square pixels).
	frustum_offset = get_jitter_pixels() * (near_plane_height / internal_size.y)

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
