extends SceneTree
# Checks the Track B project setting the engine module registers. Run through tests/engine/run_track_b.sh.

const SETTING := "rendering/prism_rp/track_b"


func _init() -> void:
	var failures := 0

	if not ProjectSettings.has_setting(SETTING):
		printerr("FAILED: the setting %s is not registered" % SETTING)
		failures += 1
	else:
		var value: int = ProjectSettings.get_setting(SETTING)
		if value != 0:
			printerr("FAILED: the default of %s is %d, expected 0 (Off)" % [SETTING, value])
			failures += 1

		var info := {}
		for property in ProjectSettings.get_property_list():
			if property["name"] == SETTING:
				info = property
		if info.get("hint_string", "") != "Off,Auto":
			printerr("FAILED: the options of %s are '%s', expected 'Off,Auto'" % [SETTING, info.get("hint_string", "")])
			failures += 1

	print("TRACK_B_SETTING_TEST: ", "PASS" if failures == 0 else "FAIL")
	quit(0 if failures == 0 else 1)
