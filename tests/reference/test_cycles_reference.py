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


@pytest.mark.skipif(shutil.which("blender") is None, reason="Blender not installed")
def test_cycles_diffuse_indirect_shows_red_bleed(tmp_path):
    scene = synthetic.write(tmp_path / "scene.json")
    out = tmp_path / "ref"
    proc = subprocess.run(["blender", "-b", "--factory-startup", "--python-exit-code", "1",
                           "--python", str(HERE / "cycles_reference.py"), "--",
                           "--scene", str(scene), "--out", str(out), "--samples", "128"], capture_output=True, text=True)
    log = f"--- stdout ---\n{proc.stdout[-4000:]}\n--- stderr ---\n{proc.stderr[-4000:]}"
    assert proc.returncode == 0 and Path(str(out) + "_full.exr").exists() and Path(str(out) + "_direct.exr").exists(), log
    ind = exr.load_indirect(out)
    assert ind.shape == (72, 96, 3) and np.isfinite(ind).all()
    h, w = ind.shape[:2]
    floor = ind[int(h * 0.92)]
    red_excess = floor[:, 0] - floor[:, 1]
    # Floor next to the red (left) wall gets red bounce light; it fades toward the white right wall.
    assert red_excess[w // 6: w // 3].mean() > 0.03
    assert red_excess[w // 6: w // 3].mean() > 1.3 * red_excess[2 * w // 3: 5 * w // 6].mean()
    assert ind[..., :].max() > 0.1                     # there is bounce light at all
