from pathlib import Path
import re

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
import run_visual_regression as rvr


def test_tier_pass_budgets_sum_to_pipeline_budget():
    for tier in rvr.HARDWARE_TIERS.values():
        assert sum(tier.pass_budgets_ms.values()) == pytest.approx(tier.prism_pipeline_budget_ms, abs=1e-6), tier.name
        assert tier.prism_pipeline_budget_ms < tier.total_frame_budget_ms


@pytest.mark.parametrize("name, stage", [
    ("chronos_aces2_odt", "aces2_optics"),
    ("aces2_lut", "aces2_optics"),
    ("chronos_taau", "chronos_vpu"),
    ("chronos_fft_bloom_forward_rows", "chronos_vpu"),
    ("lumina_rc_trace_l0", "lumina_cascades_gi"),
    ("svgf_atrous", "lumina_cascades_gi"),
    ("meshlet_cull", "gpu_meshlet_cull"),
    ("tile_shade_pbr", "tile_material_shade"),
    ("vcsp_page_fault", "virtual_shadow_pages"),
    ("froxel_inject", "froxel_volumetrics"),
    ("Opaque Pass", ""),
])
def test_canonical_stage_mapping(name, stage):
    assert rvr.map_to_canonical_stage(name) == stage


def test_python_stage_mapping_mirrors_cpp():
    """Every prefix the C++ telemetry maps must map the same way in Python."""
    cpp = (REPO / "addons/prism_rp/src/prism_telemetry.cpp").read_text()
    body = cpp[cpp.index("PrismTelemetry::_map_to_canonical_stage"):]
    body = body[: body.index("\n}\n")]
    for block in re.finditer(r"if \((.*?)\) \{\s*return \"(\w+)\";", body, re.S):
        stage = block.group(2)
        for prefix in re.findall(r'begins_with\("(\w+)"\)', block.group(1)):
            assert rvr.map_to_canonical_stage(prefix + "x") == stage, prefix
        for exact in re.findall(r'== "(\w+)"', block.group(1)):
            assert rvr.map_to_canonical_stage(exact) == stage, exact


def test_baseline_csv_roundtrip(tmp_path):
    results = [rvr.SceneEvalResult(scene="01_cornell_box", tier="tier2", passed=True,
                                   pass_timings_ms={"chronos_vpu": 1.23456, "aces2_optics": 0.5})]
    csv_path = tmp_path / "baseline.csv"
    rvr.save_baseline_timings(csv_path, results)
    loaded = rvr.load_baseline_timings(csv_path)
    assert loaded == {"01_cornell_box": {"aces2_optics": 0.5, "chronos_vpu": 1.2346}}


def test_missing_baseline_is_empty(tmp_path):
    assert rvr.load_baseline_timings(tmp_path / "nope.csv") == {}


def _write(path, arr):
    Image.fromarray(arr.astype(np.uint8)).save(path)
    return path


def test_flip_identical_images_is_zero(tmp_path):
    img = (np.random.default_rng(1).random((24, 32, 3)) * 255)
    a = _write(tmp_path / "a.png", img)
    b = _write(tmp_path / "b.png", img)
    mean, peak = rvr.evaluate_flip_diff(a, b, tmp_path / "heat.png")
    assert mean == pytest.approx(0.0, abs=1e-6) and peak == pytest.approx(0.0, abs=1e-6)
    heat = Image.open(tmp_path / "heat.png")
    assert heat.size == (32, 24) and heat.mode == "RGB"


def test_flip_detects_difference(tmp_path):
    img = np.random.default_rng(2).random((24, 32, 3)) * 200
    a = _write(tmp_path / "a.png", img)
    b = _write(tmp_path / "b.png", img + 40)
    mean, peak = rvr.evaluate_flip_diff(a, b, tmp_path / "heat.png")
    assert 0.0 < mean <= peak <= 1.0


def test_flip_rejects_resolution_mismatch(tmp_path):
    a = _write(tmp_path / "a.png", np.zeros((24, 32, 3)))
    b = _write(tmp_path / "b.png", np.zeros((12, 16, 3)))
    with pytest.raises(ValueError, match="Resolution mismatch"):
        rvr.evaluate_flip_diff(a, b, tmp_path / "heat.png")


def test_heatmap_accepts_scalar_maps(tmp_path):
    rvr.save_flip_magma_heatmap(np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8, 1), tmp_path / "h.png")
    assert Image.open(tmp_path / "h.png").size == (8, 8)
