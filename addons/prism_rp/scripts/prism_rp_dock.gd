@tool
extends VBoxContainer
## Prism RP editor dock: quick access to the Chronos / Lumina effects in the edited scene.
## Stage toggles and tuning live in the Inspector (select the Compositor resource); this dock
## covers the things you flip constantly while looking at the viewport.

const EFFECT_CLASSES: Array[StringName] = [&"ChronosVPUCompositorEffect", &"LuminaCascadesCompositorEffect"]

var _effects: Array[CompositorEffect] = []
var _effect_picker: OptionButton
var _debug_picker: OptionButton
var _telemetry_label: RichTextLabel
var _timer: Timer

func _ready() -> void:
	name = "Prism RP"
	custom_minimum_size = Vector2(220, 0)

	var refresh := Button.new()
	refresh.text = "Find effects in scene"
	refresh.pressed.connect(refresh_effects)
	add_child(refresh)

	_effect_picker = OptionButton.new()
	_effect_picker.item_selected.connect(_on_effect_selected)
	add_child(_labeled("Effect", _effect_picker))

	_debug_picker = OptionButton.new()
	_debug_picker.item_selected.connect(_on_debug_view_selected)
	add_child(_labeled("Debug view", _debug_picker))

	var buttons := HBoxContainer.new()
	var reload := Button.new()
	reload.text = "Reload shaders"
	reload.pressed.connect(_on_reload_pressed)
	buttons.add_child(reload)
	var reset := Button.new()
	reset.text = "Reset history"
	reset.pressed.connect(_on_reset_pressed)
	buttons.add_child(reset)
	add_child(buttons)

	add_child(HSeparator.new())
	_telemetry_label = RichTextLabel.new()
	_telemetry_label.fit_content = true
	_telemetry_label.bbcode_enabled = true
	_telemetry_label.custom_minimum_size = Vector2(0, 160)
	add_child(_telemetry_label)

	_timer = Timer.new()
	_timer.wait_time = 0.5
	_timer.autostart = true
	_timer.timeout.connect(_update_telemetry)
	add_child(_timer)

	refresh_effects()

func _labeled(text: String, control: Control) -> Control:
	var row := HBoxContainer.new()
	var label := Label.new()
	label.text = text
	label.custom_minimum_size = Vector2(80, 0)
	row.add_child(label)
	control.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	row.add_child(control)
	return row

## Collects Prism RP effects from every WorldEnvironment / Camera3D compositor in the edited scene.
func refresh_effects() -> void:
	_effects.clear()
	_effect_picker.clear()
	var root := EditorInterface.get_edited_scene_root()
	if root != null:
		_collect_effects(root)
	for effect in _effects:
		var owner_name: String = effect.get_meta(&"_prism_owner", "")
		_effect_picker.add_item("%s (%s)" % [effect.get_class().trim_suffix("CompositorEffect"), owner_name])
	if _effects.is_empty():
		_effect_picker.add_item("No Prism RP effects in scene")
		_effect_picker.disabled = true
	else:
		_effect_picker.disabled = false
	_on_effect_selected(0)

func _collect_effects(node: Node) -> void:
	var compositor: Compositor = null
	if node is WorldEnvironment:
		compositor = (node as WorldEnvironment).compositor
	elif node is Camera3D:
		compositor = (node as Camera3D).compositor
	if compositor != null:
		for effect in compositor.compositor_effects:
			if effect != null and EFFECT_CLASSES.has(StringName(effect.get_class())) and not _effects.has(effect):
				effect.set_meta(&"_prism_owner", node.name)
				_effects.append(effect)
	for child in node.get_children():
		_collect_effects(child)

func _selected_effect() -> CompositorEffect:
	var i := _effect_picker.selected
	if i < 0 or i >= _effects.size():
		return null
	return _effects[i]

func _on_effect_selected(_index: int) -> void:
	_debug_picker.clear()
	var effect := _selected_effect()
	_debug_picker.disabled = effect == null
	if effect == null:
		return
	# Build the list from the effect's own enum hint so it never drifts from the C++ side.
	for prop in effect.get_property_list():
		if prop["name"] == "debug_view_mode":
			for entry in String(prop["hint_string"]).split(","):
				var parts := entry.split(":")
				_debug_picker.add_item(parts[0], int(parts[1]) if parts.size() > 1 else _debug_picker.item_count)
			break
	var current: int = effect.get(&"debug_view_mode")
	var idx := _debug_picker.get_item_index(current)
	if idx >= 0:
		_debug_picker.select(idx)

func _on_debug_view_selected(index: int) -> void:
	var effect := _selected_effect()
	if effect != null:
		effect.set(&"debug_view_mode", _debug_picker.get_item_id(index))

func _on_reload_pressed() -> void:
	var effect := _selected_effect()
	if effect != null and effect.has_method(&"reload_shaders"):
		effect.call(&"reload_shaders")

func _on_reset_pressed() -> void:
	var effect := _selected_effect()
	if effect == null:
		return
	if effect.has_method(&"reset_taau_history"):
		effect.call(&"reset_taau_history")
	elif effect.has_method(&"reset_history"):
		effect.call(&"reset_history")

func _update_telemetry() -> void:
	if not is_visible_in_tree():
		return
	if not Engine.has_singleton(&"PrismTelemetry"):
		_telemetry_label.text = "[i]PrismTelemetry not loaded.[/i]"
		return
	var telemetry := Engine.get_singleton(&"PrismTelemetry")
	if not telemetry.call(&"is_enabled"):
		_telemetry_label.text = "[i]Telemetry disabled (release build?).[/i]"
		return
	var data: Dictionary = telemetry.call(&"get_telemetry_dictionary")
	var window := int(data.get("median_window_frames", 0))
	var lines := PackedStringArray(["[b]GPU ms (each pass's median over the last %d frames)[/b]" % window])
	var stages: Dictionary = data.get("pass_timings_median_ms", {})
	for stage in stages:
		lines.append("%s: %.3f" % [stage, float(stages[stage])])
	var subpasses: Dictionary = data.get("subpass_timings", {})
	if not subpasses.is_empty():
		lines.append("[b]Passes[/b]")
		for sub in subpasses:
			lines.append("  %s: %.3f" % [sub, float(subpasses[sub].get("median_gpu_ms", 0.0))])
	_telemetry_label.text = "\n".join(lines)
