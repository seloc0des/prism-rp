import re

import numpy as np
from PIL import Image

from conftest import REPO
import run_gpu_validation as gv
import run_visual_regression as rvr


def bound_properties(cpp_name: str) -> set[str]:
    src = (REPO / "addons/prism_rp/src" / cpp_name).read_text()
    return set(re.findall(r'PropertyInfo\([^,]+,\s*"(\w+)"', src))


TARGET_PROPERTIES = {
    "chronos": bound_properties("chronos_vpu_effect.cpp"),
    "lumina": bound_properties("lumina_cascades_effect.cpp"),
}
RIG = (REPO / "project/benchmarks/prism_benchmark_rig.gd").read_text()


def test_config_names_unique():
    names = [c.name for c in gv.CONFIGS]
    assert len(names) == len(set(names))


def test_every_override_targets_a_bound_property():
    for cfg in gv.CONFIGS:
        for arg in cfg.args:
            if not arg.startswith("--prism-set="):
                continue
            for entry in arg.removeprefix("--prism-set=").split(","):
                target, rest = entry.split(".", 1)
                prop = rest.split("=", 1)[0]
                assert prop in TARGET_PROPERTIES[target], f"{cfg.name}: {target} has no property {prop}"


def test_every_rig_flag_is_parsed_by_the_rig():
    for cfg in gv.CONFIGS:
        for arg in cfg.args:
            flag = arg.split("=", 1)[0] + ("=" if "=" in arg else "")
            assert f'"{flag}' in RIG, f"{cfg.name}: rig doesn't parse {flag}"


def test_report_end_to_end_with_fake_captures(tmp_path, monkeypatch):
    def fake_capture(godot_bin, renderdoccmd, project_dir, scene_name, scene_res_path, tier, capture_frame,
                     artifacts_dir, extra_args=None, render_scale=None, output_dir=None):
        output_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((12, 16, 3), 100, np.uint8)).save(output_dir / "captured.png")
        (output_dir / "godot.log").write_text("[Prism RP] Registered\n")
        errors = ["VUID-fake"] if "--prism-godot-taa=on" in (extra_args or []) else []
        return output_dir / "captured.png", {"chronos_vpu": 1.0}, errors

    monkeypatch.setattr(rvr, "capture_scene", fake_capture)
    tier = rvr.HARDWARE_TIERS["tier2"]
    results = [gv.run_config("godot", tmp_path, tmp_path, tier, "01_cornell_box", c, 48) for c in gv.CONFIGS]
    gate = [gv.run_gate1("godot", tmp_path, tmp_path, tier, "01_cornell_box", 48, 0.08)]
    report = gv.write_report(tmp_path, tier, results, gate, 0.08).read_text()

    assert sum(not r.ok for r in results) == 1
    assert "chronos_godot_taa_guard**: VUID-fake" in report
    assert gate[0].passed and gate[0].mean_flip == 0.0
    assert report.count("| **") == len(gv.CONFIGS)
    assert (tmp_path / "results.json").exists()


def test_errors_shared_with_baseline_are_labeled(tmp_path):
    tier = rvr.HARDWARE_TIERS["tier2"]
    engine = "Validation Error: [ VUID-vkQueueSubmit-pSignalSemaphores-00067 ] swapchain"
    results = [
        gv.RunResult(scene="01_cornell_box", config="baseline_godot", ok=False, errors=[engine]),
        gv.RunResult(scene="01_cornell_box", config="chronos_default", ok=False, errors=[engine]),
        gv.RunResult(scene="01_cornell_box", config="lumina_default", ok=False,
                     errors=["Validation Error: [ VUID-vkCmdDispatch-None-08114 ] ours"]),
    ]
    report = gv.write_report(tmp_path, tier, results, [], 0.08).read_text()
    chronos_line = next(l for l in report.splitlines() if "chronos_default**" in l)
    lumina_line = next(l for l in report.splitlines() if "lumina_default**" in l)
    assert "also in baseline_godot" in chronos_line
    assert "also in baseline_godot" not in lumina_line
