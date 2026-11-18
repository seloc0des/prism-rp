#!/usr/bin/env python3
"""Gate 2's settling check: after a light switches on, Lumina's indirect light settles within 6 frames.

Scene 01 is frozen with a static camera (the rig's motion test at 0 degrees per frame), Chronos is off
and Lumina shows its denoised irradiance. The scene's lights stay off until --switch-frame, so the
bounce light they cause starts from nothing. The frames after the switch and a much later, settled
frame are saved as float EXRs. The settling error k frames after the switch is

    mean |I(switch + k) - I(settled)| / mean I(settled)

and the light has settled at the first k where that drops below --tolerance (10% by default). The
check passes when that happens within --max-frames (6, ROADMAP.md Gate 2). Frame timing is fixed
(--fixed-fps 60), so the result doesn't depend on how fast the GPU or lavapipe renders.

    python3 tests/check_gi_convergence.py --godot-bin godot --out-dir tests/artifacts/gi_convergence
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "reference"))

import run_visual_regression as rvr  # noqa: E402

SCENE = "01_cornell_box"


def settling_errors(frames: list[np.ndarray], settled: np.ndarray) -> list[float]:
    level = max(float(np.mean(settled)), 1e-6)
    return [float(np.mean(np.abs(f - settled))) / level for f in frames]


def frames_to_settle(errors: list[float], tolerance: float) -> int | None:
    """First k (0 = the switch frame itself) with an error below tolerance, or None."""
    return next((k for k, e in enumerate(errors) if e < tolerance), None)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--godot-bin", default=None)
    p.add_argument("--project-dir", default="project")
    p.add_argument("--tier", choices=list(rvr.HARDWARE_TIERS), default="tier1")
    p.add_argument("--out-dir", default="tests/artifacts/gi_convergence")
    p.add_argument("--switch-frame", type=int, default=30, help="Frame the lights come on.")
    p.add_argument("--frames", type=int, default=12, help="Frames after the switch to save.")
    p.add_argument("--settled-after", type=int, default=90, help="Frames after the switch taken as settled.")
    p.add_argument("--tolerance", type=float, default=0.10)
    p.add_argument("--max-frames", type=int, default=6)
    p.add_argument("--report-only", action="store_true", help="Exit 0 even when it settles too slowly.")
    args = p.parse_args()

    import exr

    repo = Path(__file__).resolve().parent.parent
    out_dir = Path(args.out_dir) if Path(args.out_dir).is_absolute() else (repo / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tier = rvr.HARDWARE_TIERS[args.tier]
    godot = rvr.resolve_executable(args.godot_bin, ["godot", "godot4", "Godot_v4.7.2-stable_linux.x86_64"], "Godot 4 executable")

    f0 = args.switch_frame
    settled_frame = f0 + args.settled_after
    wanted = list(range(f0, f0 + args.frames + 1)) + [settled_frame]
    exr_dir = out_dir / "frames"
    extra = ["--prism-chronos=off", "--prism-no-jitter", "--prism-ambient=off", "--prism-ssr=off", "--prism-motion-test=0",
             "--prism-set=lumina.debug_view_mode=2", f"--prism-light-on-frame={f0}",
             f"--qa-exr-frames={','.join(str(n) for n in wanted)}", f"--qa-exr-dir={exr_dir}"]
    rvr.capture_scene(godot, None, (repo / args.project_dir).resolve(), SCENE, rvr.CANONICAL_SCENES[SCENE], tier,
                      settled_frame, out_dir, extra_args=extra, output_dir=out_dir)

    missing = [n for n in wanted if not (exr_dir / f"frame_{n}.exr").exists()]
    if missing:
        print(f"GI convergence: no EXR for frames {missing}; see {out_dir / 'godot.log'}")
        return 2
    settled = exr.load_rgb(exr_dir / f"frame_{settled_frame}.exr").astype(np.float64)
    frames = [exr.load_rgb(exr_dir / f"frame_{n}.exr").astype(np.float64) for n in range(f0, f0 + args.frames + 1)]
    errors = settling_errors(frames, settled)
    k = frames_to_settle(errors, args.tolerance)
    passed = k is not None and k <= args.max_frames
    report = {"scene": SCENE, "tier": tier.name, "switch_frame": f0, "settled_frame": settled_frame,
              "tolerance": args.tolerance, "max_frames": args.max_frames, "errors": errors,
              "frames_to_settle": k, "settled_level": float(np.mean(settled)), "passed": passed}
    (out_dir / "gi_convergence.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Settling error after k frames: " + ", ".join(f"{i}:{e:.3f}" for i, e in enumerate(errors)))
    settle_text = f"settles after {k} frames" if k is not None else f"not settled within {args.frames} frames"
    print(f"GI convergence: {settle_text} (within {100 * args.tolerance:.0f}%; needs <= {args.max_frames}) -> "
          f"{'PASS' if passed else 'FAIL'}")
    return 0 if passed or args.report_only else 1


if __name__ == "__main__":
    sys.exit(main())
