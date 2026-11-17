import pytest

import check_gdscript_api as api


@pytest.mark.skipif(not api.API_JSON.exists(), reason="godot-cpp submodule not checked out")
def test_gdscript_uses_only_known_godot_names():
    problems = api.run()
    assert not problems, "\n".join(problems)


@pytest.mark.skipif(not api.API_JSON.exists(), reason="godot-cpp submodule not checked out")
def test_checker_catches_renamed_property_and_typos(tmp_path):
    gd = tmp_path / "bad.gd"
    gd.write_text(
        "extends Node\n"
        "func _ready() -> void:\n"
        "\tvar sun := DirectionalLight3D.new()\n"
        "\tsun.volumetric_fog_energy = 2.0\n"          # renamed in Godot 4 (light_volumetric_fog_energy)
        "\tsun.look_att(Vector3.ZERO)\n"               # typo'd method
        "\tvar m := BaseMaterial3D.BILLBOARD_NOPE\n"   # unknown constant
        "\tprint(\"a.b = c\")\n"                       # inside a string: must be ignored
    )
    problems = api.check_file(gd, api.load_api_names())
    assert any("volumetric_fog_energy" in p for p in problems)
    assert any("look_att" in p for p in problems)
    assert any("BILLBOARD_NOPE" in p for p in problems)
    assert len(problems) == 3
