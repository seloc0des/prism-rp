"""tests/check_export.py pieces that don't need Godot: presets, the run verdict, version agreement."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import check_export as ce  # noqa: E402
import gpu_session as gs  # noqa: E402


def test_presets_export_both_platforms_with_the_shaders():
    cfg = ce.presets_cfg(["linux", "windows"])
    assert '[preset.0]' in cfg and 'name="Linux"' in cfg and 'platform="Linux"' in cfg
    assert '[preset.1]' in cfg and 'platform="Windows Desktop"' in cfg
    assert cfg.count('include_filter="*.glsl"') == 2           # the .glsl files Prism RP reads at runtime
    assert "application/modify_resources=false" in cfg          # no rcedit needed on Linux


def test_verdict_needs_the_extension_a_frame_and_both_effects(tmp_path):
    shot, tele = tmp_path / "captured.png", tmp_path / "telemetry.json"
    log = "Godot Engine v4.7.2\n[Prism RP] Registered PrismTelemetry, ChronosVPUCompositorEffect ...\n"
    shot.write_bytes(b"png")
    tele.write_text(json.dumps({"pass_timings_ms": {"chronos_vpu": 0.31, "lumina_cascades_gi": 1.2, "aces2_optics": 0.0}}))
    ok, detail = ce.verify_run(log, shot, tele)
    assert ok and "chronos_vpu 0.31 ms" in detail and "aces2_optics" not in detail

    ok, detail = ce.verify_run("ERROR: Can't open dynamic library: libprism_rp...", shot, tele)
    assert not ok and "didn't register" in detail and "Can't open dynamic library" in detail

    tele.write_text(json.dumps({"pass_timings_ms": {"chronos_vpu": 0.3}}))
    ok, detail = ce.verify_run(log, shot, tele)
    assert not ok and "no Lumina timings" in detail


def test_godot_version_matches_the_session_script():
    assert ce.GODOT_VERSION == gs.GODOT_VERSION
