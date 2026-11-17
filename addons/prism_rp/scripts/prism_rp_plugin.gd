@tool
extends EditorPlugin

func _enter_tree() -> void:
	print("[Prism RP] EditorPlugin enabled.")

func _exit_tree() -> void:
	print("[Prism RP] EditorPlugin disabled.")
