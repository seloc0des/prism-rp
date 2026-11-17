#!/usr/bin/env python3
"""Gate 2: Lumina's indirect light vs a Cycles path-traced reference.

For each scene: Godot renders Lumina's "Denoised Irradiance" debug view (bounced diffuse light
for white albedo; Chronos off, ambient off, no jitter, full resolution) and exports the scene
(project/benchmarks/prism_scene_export.gd). Blender re-renders that export with Cycles and writes
its Diffuse Indirect pass, limited to the same single diffuse bounce. The two are compared
in linear light:

  psnr_db         raw PSNR
  psnr_fitted_db  PSNR after a least-squares brightness fit; Godot and Blender light units
                  don't map exactly, so this measures whether the light is in the right places
  fitted_scale    how far off the units are (1.0 = perfect calibration)

Gate 2 passes when psnr_fitted_db > 36 (ROADMAP.md). Needs a GPU for Godot, Blender 4.x or 5.x, and
`pip install OpenEXR`.

    python3 tests/run_gate2.py --scenes 01_cornell_box 04_dynamic_sponza
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "reference"))

import run_visual_regression as rvr  # noqa: E402
from exr import load_pass  # noqa: E402
from metrics import gate2_metrics, srgb_to_linear  # noqa: E402

GATE2_SCENES = ["01_cornell_box", "04_dynamic_sponza"]
CYCLES_SCRIPT = HERE / "reference" / "cycles_reference.py"


def lumina_capture_args(export_dir: Path) -> list[str]:
    return [
        "--prism-chronos=off", "--prism-no-jitter", "--prism-ambient=off",
        "--prism-set=lumina.debug_view_mode=2",
        f"--qa-scene-export={export_dir}",
    ]


def render_reference(blender: str, scene_json: Path, out_prefix: Path, samples: int, bounces: int) -> Path:
    cmd = [blender, "-b", "--factory-startup", "--python-exit-code", "1", "--python", str(CYCLES_SCRIPT), "--",
           "--scene", str(scene_json), "--out", str(out_prefix), "--samples", str(samples), "--diffuse-bounces", str(bounces)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (out_prefix.parent / "blender.log").write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    exr_path = Path(str(out_prefix) + "_passes.exr")
    if proc.returncode != 0 or not exr_path.exists():
        raise RuntimeError(f"Blender failed (exit {proc.returncode}); see {out_prefix.parent / 'blender.log'}")
    return exr_path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--scenes", nargs="+", default=GATE2_SCENES, choices=list(rvr.CANONICAL_SCENES))
    p.add_argument("--tier", choices=list(rvr.HARDWARE_TIERS), default="tier2")
    p.add_argument("--godot-bin", default=None)
    p.add_argument("--blender-bin", default=None)
    p.add_argument("--samples", type=int, default=512)
    p.add_argument("--diffuse-bounces", type=int, default=1, help="Lumina v1 is single-bounce.")
    p.add_argument("--capture-frame", type=int, default=48)
    p.add_argument("--min-psnr", type=float, default=36.0)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    repo = HERE.parent
    out_root = Path(args.out_dir) if args.out_dir else repo / "tests" / "artifacts" / "gate2" / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)
    godot = rvr.resolve_executable(args.godot_bin, ["godot", "godot4", "Godot_v4.4.1-stable_linux.x86_64"], "Godot 4 executable")
    blender = rvr.resolve_executable(args.blender_bin, ["blender", "blender.exe"], "Blender 4.x")
    tier = rvr.HARDWARE_TIERS[args.tier]

    results = []
    for scene in args.scenes:
        scene_dir = out_root / scene
        export_dir = scene_dir / "export"
        print(f"[{scene}] Godot (Lumina irradiance) ...")
        png, _, errors = rvr.capture_scene(godot, None, repo / "project", scene, rvr.CANONICAL_SCENES[scene], tier,
                                           args.capture_frame, out_root, lumina_capture_args(export_dir), 1.0, scene_dir / "godot")
        print(f"[{scene}] Cycles reference ({args.samples} spp) ...")
        ref = load_pass(render_reference(blender, export_dir / "scene.json", scene_dir / "reference", args.samples, args.diffuse_bounces), "DiffInd")
        test = srgb_to_linear(np.asarray(Image.open(png).convert("RGB"), np.float64) / 255.0)
        if test.shape != ref.shape:
            raise SystemExit(f"{scene}: Godot capture {test.shape} vs reference {ref.shape}")
        m = gate2_metrics(test, ref)
        m.update(scene=scene, passed=m["psnr_fitted_db"] > args.min_psnr, godot_errors=errors)
        results.append(m)
        print(f"  PSNR {m['psnr_db']:.2f} dB raw, {m['psnr_fitted_db']:.2f} dB fitted (scale {m['fitted_scale']:.3f})"
              f" -> {'PASS' if m['passed'] else 'FAIL'}")

    (out_root / "gate2.json").write_text(json.dumps(results, indent=2))
    print(f"\nResults: {out_root / 'gate2.json'}")
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
