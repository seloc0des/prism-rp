#!/usr/bin/env python3
"""Gate 2: Lumina's indirect light vs a Cycles path-traced reference.

For each scene: Godot renders Lumina's "Denoised Irradiance" debug view (bounced diffuse light
for white albedo; Chronos off, ambient off, no jitter, full resolution) and exports the scene
(project/benchmarks/prism_scene_export.gd). Blender re-renders that export with Cycles, once with
the same single diffuse bounce and once without; the difference is the bounced light. Every Godot
capture uses the same matte scene (--prism-diffuse-only: no metals, no specular), so the export,
Lumina and SDFGI share one set of materials. Godot also captures its unshaded view (each visible
surface's albedo), and both irradiances are multiplied by it: Gate 2 compares bounce light as a
viewer sees it, in linear light:

  psnr_db         raw PSNR
  psnr_fitted_db  PSNR after a least-squares brightness fit; Godot and Blender light units
                  don't map exactly, so this measures whether the light is in the right places
  fitted_scale    how far off the units are (1.0 = perfect calibration)

Also psnr_center_db (fitted, central 70% of the frame: how much the screen edges cost, where
screen-space rays run out) and the mean levels. Each scene directory gets compare.png (reference |
Lumina | error, at one exposure) and gate2.json collects the numbers.

Emission is off in every capture (and so in the export Cycles renders): Gate 2 compares the bounce
light of the scene's lights. Emissive surfaces are often stand-ins drawn next to a real light, like
the Cornell box's ceiling panel, and counting their light as Lumina's would light the scene twice.

Stock Godot's own GI is measured the same way (on by default; --no-sdfgi skips it): two more Godot
captures of the final image with SDFGI on and off, whose difference is SDFGI's bounce light as seen.
Not Godot's lighting-only debug view: it swaps every material, including the ones SDFGI captures its
colours from, for a grey default, so SDFGI would be measured in a colourless scene.
compare_sdfgi.png shows it.

--noise-check renders the reference a second time with another seed. reference_ceiling_db is the
best fitted PSNR anything could score against this reference: its self-PSNR plus 3 dB (a perfect
estimate differs from one noisy render by that noise, two renders by twice its variance).

Gate 2 passes when Lumina beats SDFGI and its fitted PSNR exceeds --min-psnr (ROADMAP.md);
--report-only exits 0 anyway (CI on lavapipe).
Needs a GPU for Godot (or lavapipe), Blender 4.x or 5.x, and `pip install OpenEXR`.

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
from exr import load_indirect, load_rgb  # noqa: E402
from metrics import gate2_metrics, psnr  # noqa: E402

GATE2_SCENES = ["01_cornell_box", "04_dynamic_sponza"]
CYCLES_SCRIPT = HERE / "reference" / "cycles_reference.py"


# Every Gate 2 capture renders the same scene the reference does: matte (no metals, no specular), no emission, no
# environment light, and one bounce of GI, so the exported scene, Lumina and SDFGI share one set of materials and
# the comparison is about a single diffuse bounce of the scene's lights.
GATE2_SCENE = ["--prism-chronos=off", "--prism-no-jitter", "--prism-ambient=off", "--prism-ssr=off",
               "--prism-emission=off", "--prism-diffuse-only", "--prism-gi-single-bounce"]


def lumina_capture_args(export_dir: Path, exr_path: Path) -> list[str]:
    return [
        *GATE2_SCENE, "--prism-set=lumina.debug_view_mode=2",
        f"--qa-scene-export={export_dir}",
        f"--qa-exr-out={exr_path}",   # float capture: irradiance above 1.0 isn't clipped
    ]


def sdfgi_capture_args(exr_path: Path, sdfgi_on: bool) -> list[str]:
    """The final image with SDFGI on or off; the difference is SDFGI's bounce light as seen. Not Godot's
    lighting-only view: it swaps every material for a grey default, also where SDFGI captures colours."""
    return [*GATE2_SCENE, "--prism-lumina=off", f"--prism-sdfgi={'on' if sdfgi_on else 'off'}", f"--qa-exr-out={exr_path}"]


def albedo_capture_args(exr_path: Path) -> list[str]:
    """Godot's unshaded view: each visible surface's albedo, to weight the bounce light as a viewer sees it."""
    return [*GATE2_SCENE, "--prism-lumina=off", "--prism-sdfgi=off", "--prism-debug-draw=unshaded", f"--qa-exr-out={exr_path}"]


def jsonable(value):
    """numpy scalars (bool_, float64, ...) and containers of them -> plain Python for json.dumps."""
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value.item() if isinstance(value, np.generic) else value


def to_srgb8(linear: np.ndarray, exposure: float) -> np.ndarray:
    c = np.clip(np.asarray(linear, np.float64) * exposure, 0.0, 1.0)
    return (np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055) * 255.0 + 0.5).astype(np.uint8)


def write_comparison(path: Path, reference: np.ndarray, test: np.ndarray, scale: float) -> None:
    """reference | Lumina (brightness-fitted) | |difference| x4, all at the reference's exposure."""
    exposure = 1.0 / max(float(np.percentile(reference, 99.5)), 1e-4)
    fitted = test * scale
    error = np.abs(fitted - reference).mean(axis=-1, keepdims=True).repeat(3, axis=-1) * 4.0
    row = np.concatenate([to_srgb8(reference, exposure), to_srgb8(fitted, exposure), to_srgb8(error, exposure)], axis=1)
    Image.fromarray(row).save(path)


def render_reference(blender: str, scene_json: Path, out_prefix: Path, samples: int, bounces: int, seed: int = 1) -> Path:
    cmd = [blender, "-b", "--factory-startup", "--python-exit-code", "1", "--python", str(CYCLES_SCRIPT), "--",
           "--scene", str(scene_json), "--out", str(out_prefix), "--samples", str(samples), "--diffuse-bounces", str(bounces),
           "--seed", str(seed)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log = out_prefix.parent / ("blender.log" if seed == 1 else f"blender_seed{seed}.log")
    log.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0 or not Path(str(out_prefix) + "_direct.exr").exists():
        raise RuntimeError(f"Blender failed (exit {proc.returncode}); see {log}")
    return out_prefix


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--scenes", nargs="+", default=GATE2_SCENES, choices=list(rvr.CANONICAL_SCENES))
    p.add_argument("--tier", choices=list(rvr.HARDWARE_TIERS), default="tier2")
    p.add_argument("--godot-bin", default=None)
    p.add_argument("--blender-bin", default=None)
    p.add_argument("--samples", type=int, default=512)
    p.add_argument("--diffuse-bounces", type=int, default=1, help="Lumina v1 is single-bounce.")
    p.add_argument("--capture-frame", type=int, default=48)
    p.add_argument("--min-psnr", type=float, default=25.0,
                   help="Interim floor of fitted PSNR for screen-space Lumina (docs/NEXT_STEPS.md); 36 is the ray-query target.")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--report-only", action="store_true", help="Exit 0 even below --min-psnr (crashes still fail).")
    p.add_argument("--no-sdfgi", action="store_true", help="Skip the comparison with Godot's SDFGI.")
    p.add_argument("--noise-check", action="store_true", help="Render the reference twice to measure its noise ceiling.")
    args = p.parse_args()

    repo = HERE.parent
    # Absolute: Godot runs with --path project, so it would resolve a relative path against project/.
    out_root = (Path(args.out_dir) if args.out_dir else repo / "tests" / "artifacts" / "gate2" / dt.datetime.now().strftime("%Y%m%d-%H%M%S")).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    godot = rvr.resolve_executable(args.godot_bin, ["godot", "godot4", "Godot_v4.7.2-stable_linux.x86_64"], "Godot 4 executable")
    blender = rvr.resolve_executable(args.blender_bin, ["blender", "blender.exe"], "Blender 4.x")
    tier = rvr.HARDWARE_TIERS[args.tier]

    results = []
    for scene in args.scenes:
        scene_dir = out_root / scene
        export_dir = scene_dir / "export"
        print(f"[{scene}] Godot (Lumina irradiance) ...")
        exr_path = scene_dir / "godot" / "irradiance.exr"
        albedo_exr = scene_dir / "albedo" / "albedo.exr"
        for stale in (exr_path, albedo_exr):   # capture_scene clears a stale PNG, not these
            stale.unlink(missing_ok=True)
        _, _, errors = rvr.capture_scene(godot, None, repo / "project", scene, rvr.CANONICAL_SCENES[scene], tier,
                                         args.capture_frame, out_root, lumina_capture_args(export_dir, exr_path),
                                         1.0, scene_dir / "godot")
        print(f"[{scene}] Godot (albedo, unshaded view) ...")
        _, _, albedo_errors = rvr.capture_scene(godot, None, repo / "project", scene, rvr.CANONICAL_SCENES[scene], tier,
                                                args.capture_frame, out_root, albedo_capture_args(albedo_exr), 1.0,
                                                albedo_exr.parent)
        errors = [*errors, *albedo_errors]
        for exr in (exr_path, albedo_exr):
            if not exr.exists():
                raise SystemExit(f"{scene}: Godot wrote no {exr} (see the godot.log beside it)")
        test, albedo = load_rgb(exr_path).astype(np.float64), load_rgb(albedo_exr).astype(np.float64)
        print(f"[{scene}] Cycles reference ({args.samples} spp) ...")
        ref = load_indirect(render_reference(blender, export_dir / "scene.json", scene_dir / "reference", args.samples, args.diffuse_bounces))
        if not (test.shape == ref.shape == albedo.shape):
            raise SystemExit(f"{scene}: Godot capture {test.shape}, albedo {albedo.shape}, reference {ref.shape}")
        # Bounce light as a viewer sees it: irradiance (E / pi, the white-albedo quantity) times the surface's albedo.
        ref_seen = ref * albedo
        m = gate2_metrics(test * albedo, ref_seen, clipped_test=False)
        m.update(scene=scene, godot_errors=errors)
        write_comparison(scene_dir / "compare.png", ref_seen, test * albedo, m["fitted_scale"])
        print(f"  Lumina: {m['psnr_db']:.2f} dB raw, {m['psnr_fitted_db']:.2f} dB fitted (scale {m['fitted_scale']:.3f}), "
              f"{m['psnr_center_db']:.2f} dB in the centre; mean level {m['mean_test']:.4f} vs reference {m['mean_reference']:.4f}")

        if not args.no_sdfgi:
            print(f"[{scene}] Godot SDFGI, on and off ...")
            frames = {}
            for on in (True, False):
                exr = scene_dir / f"sdfgi_{'on' if on else 'off'}" / "frame.exr"
                exr.unlink(missing_ok=True)
                _, _, sdfgi_errors = rvr.capture_scene(godot, None, repo / "project", scene, rvr.CANONICAL_SCENES[scene], tier,
                                                       args.capture_frame, out_root, sdfgi_capture_args(exr, on), 1.0, exr.parent)
                m["godot_errors"] = [*m["godot_errors"], *sdfgi_errors]
                if not exr.exists():
                    raise SystemExit(f"{scene}: Godot wrote no {exr} (see the godot.log beside it)")
                frames[on] = load_rgb(exr).astype(np.float64)
            sdfgi = np.maximum(frames[True] - frames[False], 0.0)   # already as seen: albedo times SDFGI's irradiance
            ms = gate2_metrics(sdfgi, ref_seen, clipped_test=False)
            m["sdfgi"] = ms
            m["beats_sdfgi"] = m["psnr_fitted_db"] > ms["psnr_fitted_db"]
            write_comparison(scene_dir / "compare_sdfgi.png", ref_seen, sdfgi, ms["fitted_scale"])
            print(f"  SDFGI:  {ms['psnr_db']:.2f} dB raw, {ms['psnr_fitted_db']:.2f} dB fitted (scale {ms['fitted_scale']:.3f}), "
                  f"{ms['psnr_center_db']:.2f} dB in the centre -> Lumina {'beats' if m['beats_sdfgi'] else 'does not beat'} SDFGI")

        if args.noise_check:
            print(f"[{scene}] Cycles reference again, second seed ...")
            ref_b = load_indirect(render_reference(blender, export_dir / "scene.json", scene_dir / "reference_seed2",
                                                   args.samples, args.diffuse_bounces, seed=2))
            m["reference_self_psnr_db"] = psnr(ref_b * albedo, ref_seen)
            m["reference_ceiling_db"] = m["reference_self_psnr_db"] + 3.01
            print(f"  Reference noise: two seeds agree to {m['reference_self_psnr_db']:.2f} dB, so nothing can score "
                  f"above about {m['reference_ceiling_db']:.2f} dB at {args.samples} samples")

        m["passed"] = m["psnr_fitted_db"] > args.min_psnr and m.get("beats_sdfgi", True)
        print(f"  -> {'PASS' if m['passed'] else 'FAIL'} (needs > {args.min_psnr} dB fitted"
              f"{' and beating SDFGI' if 'beats_sdfgi' in m else ''})")
        results.append(jsonable(m))

    (out_root / "gate2.json").write_text(json.dumps(results, indent=2))
    print(f"\nResults: {out_root / 'gate2.json'}")
    return 0 if args.report_only or all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
