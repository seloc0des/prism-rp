"""Runs the real Cycles pipeline on a synthetic Cornell box (skipped without Blender)."""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exr  # noqa: E402
import synthetic  # noqa: E402


def render(scene_path: Path, out: Path, samples: int) -> None:
    proc = subprocess.run(["blender", "-b", "--factory-startup", "--python-exit-code", "1",
                           "--python", str(HERE / "cycles_reference.py"), "--",
                           "--scene", str(scene_path), "--out", str(out), "--samples", str(samples)],
                          capture_output=True, text=True)
    log = f"--- stdout ---\n{proc.stdout[-4000:]}\n--- stderr ---\n{proc.stderr[-4000:]}"
    assert proc.returncode == 0 and Path(str(out) + "_full.exr").exists() and Path(str(out) + "_direct.exr").exists(), log


@pytest.mark.skipif(shutil.which("blender") is None, reason="Blender not installed")
def test_cycles_diffuse_indirect_shows_red_bleed(tmp_path):
    scene = synthetic.write(tmp_path / "scene.json")
    out = tmp_path / "ref"
    render(scene, out, 128)
    ind = exr.load_indirect(out)
    assert ind.shape == (72, 96, 3) and np.isfinite(ind).all()
    h, w = ind.shape[:2]
    floor = ind[int(h * 0.92)]
    red_excess = floor[:, 0] - floor[:, 1]
    # Floor next to the red (left) wall gets red bounce light; it fades toward the white right wall.
    assert red_excess[w // 6: w // 3].mean() > 0.03
    assert red_excess[w // 6: w // 3].mean() > 1.3 * red_excess[2 * w // 3: 5 * w // 6].mean()
    assert ind[..., :].max() > 0.1                     # there is bounce light at all


@pytest.mark.skipif(shutil.which("blender") is None, reason="Blender not installed")
def test_authored_colours_render_decoded_like_godot(tmp_path):
    # Godot decodes an authored sRGB 0.5 to linear 0.214 for the background and the ambient light. The
    # reference must render the same light: a white floor under that ambient, and the background above it.
    scene = synthetic.cornell()
    scene["meshes"] = [m for m in scene["meshes"] if m["name"] == "floor"]
    scene["lights"] = []
    scene["environment"] = {"background": [0.5, 0.5, 0.5], "ambient": [0.5, 0.5, 0.5], "ambient_energy": 1.0}
    out = tmp_path / "ref"
    render(synthetic.write(tmp_path / "scene.json", scene), out, 64)
    img = exr.load_rgb(str(out) + "_full.exr")
    h, w = img.shape[:2]
    assert img[: h // 4].mean() == pytest.approx(0.214, abs=0.005)                          # background
    assert img[h - 3:, int(w * 0.4): int(w * 0.6)].mean() == pytest.approx(0.214, abs=0.03)  # floor, lit by the ambient
