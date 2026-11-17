@tool
extends EditorPlugin

const DockScript := preload("res://addons/prism_rp/scripts/prism_rp_dock.gd")

var _dock: Control

func _enter_tree() -> void:
	_dock = DockScript.new()
	add_control_to_dock(DOCK_SLOT_RIGHT_UL, _dock)
	scene_changed.connect(_on_scene_changed)

func _exit_tree() -> void:
	if scene_changed.is_connected(_on_scene_changed):
		scene_changed.disconnect(_on_scene_changed)
	if _dock != null:
		remove_control_from_docks(_dock)
		_dock.queue_free()
		_dock = null

func _on_scene_changed(_root: Node) -> void:
	if _dock != null:
		_dock.call(&"refresh_effects")
