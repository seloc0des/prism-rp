#!/usr/bin/env python3
"""Check the motion vectors Chronos uses against ones computed from the camera.

Godot produces the motion vectors; Chronos reads them (convention: velocity = prev_uv - curr_uv, uv with y
down) and removes the camera jitter it adds through ChronosJitterCamera. A change in Godot's convention, or
broken jitter compensation, shows up here as a sign, scale or sub-pixel offset error.

The benchmark rig's motion test freezes the scene and yaws the camera at a constant rate. Under pure rotation
the expected motion vector of every pixel depends only on the two camera orientations and the field of view,
not on depth, so it can be computed exactly. Chronos' signed velocity debug view (mode 5: rg = 0.5 + v * 10)
is captured as a float EXR. Three runs:

  calibration   no motion, no jitter: every pixel must read zero (also detects an sRGB-encoded capture)
  yaw           camera rotating, no jitter
  yaw_jitter    camera rotating with ChronosJitterCamera's jitter: compensation must leave the same field

    python3 tests/check_motion_vectors.py --godot-bin godot --out-dir tests/artifacts/motion_check
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "reference"))

import run_visual_regression as rvr  # noqa: E402

SCENE = "01_cornell_box"
SIGNED_VELOCITY_SCALE = 10.0  # chronos_aces2_odt.comp.glsl, debug view 5
CAMERA_RE = re.compile(r"motion-test camera fov=([\d.]+) keep_aspect=(\d+) curr=\[([^\]]+)\] prev=\[([^\]]+)\]")
KEEP_WIDTH = 0  # Camera3D.KEEP_WIDTH: fov is horizontal; KEEP_HEIGHT (1, the default): vertical


def parse_basis(text: str) -> np.ndarray:
    """'x.x,x.y,x.z;y.x,...' (the rig prints the basis columns) -> 3x3 matrix with those columns."""
    cols = [[float(v) for v in col.split(",")] for col in text.split(";")]
    return np.array(cols, dtype=np.float64).T


def expected_velocity(width: int, height: int, fov_deg: float, keep_aspect: int,
                      curr: np.ndarray, prev: np.ndarray) -> np.ndarray:
    """(H, W, 2) motion vectors in uv units (prev - curr, y down) for a camera that only rotated."""
    aspect = width / height
    t = np.tan(np.radians(fov_deg) / 2)
    tx, ty = (t, t / aspect) if keep_aspect == KEEP_WIDTH else (t * aspect, t)
    u = (np.arange(width) + 0.5) / width
    v = (np.arange(height) + 0.5) / height
    uu, vv = np.meshgrid(u, v)
    # View-space ray through each pixel (camera looks down -Z, +Y up, uv y down), to world, to previous view.
    ray_view = np.stack([(2 * uu - 1) * tx, -(2 * vv - 1) * ty, -np.ones_like(uu)], axis=-1)
    ray_world = ray_view @ curr.T
    ray_prev = ray_world @ prev  # prev.T applied to each ray: world -> previous view
    depth = -ray_prev[..., 2]
    prev_u = (ray_prev[..., 0] / depth / tx) * 0.5 + 0.5
    prev_v = (-ray_prev[..., 1] / depth / ty) * 0.5 + 0.5
    return np.stack([prev_u - uu, prev_v - vv], axis=-1)


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def capture(godot_bin: str, project: Path, out_dir: Path, tier, name: str, yaw: float, jitter: bool,
            frame: int) -> tuple[np.ndarray, str]:
    import exr

    run_dir = out_dir / name
    exr_path = run_dir / "velocity.exr"
    args = ["--prism-lumina=off", f"--prism-motion-test={yaw}", f"--qa-exr-out={exr_path}",
            "--prism-set=chronos.debug_view_mode=5"]
    if not jitter:
        args.append("--prism-no-jitter")
    rvr.capture_scene(godot_bin, None, project, SCENE, rvr.CANONICAL_SCENES[SCENE], tier, frame, out_dir,
                      extra_args=args, output_dir=run_dir)
    if not exr_path.exists():
        raise RuntimeError(f"{name}: no EXR written; see {run_dir / 'godot.log'}")
    return exr.load_rgb(exr_path), (run_dir / "godot.log").read_text(encoding="utf-8", errors="replace")


def evaluate(measured: np.ndarray, expected: np.ndarray, internal_width: int, tolerance_px: float) -> dict:
    """Compare in the central region (the box fills it; the clear-colour background has no motion vectors)."""
    h, w = expected.shape[:2]
    y0, y1, x0, x1 = int(h * 0.35), int(h * 0.65), int(w * 0.35), int(w * 0.65)
    err_px = np.linalg.norm(measured[y0:y1, x0:x1] - expected[y0:y1, x0:x1], axis=-1) * internal_width
    exp_px = np.linalg.norm(expected[y0:y1, x0:x1], axis=-1) * internal_width
    return {
        "median_error_px": float(np.median(err_px)),
        "p90_error_px": float(np.percentile(err_px, 90)),
        "within_tolerance": float(np.mean(err_px <= tolerance_px)),
        "expected_motion_px": float(np.median(exp_px)),
        "mean_measured_uv": [float(x) for x in measured[y0:y1, x0:x1].reshape(-1, 2).mean(axis=0)],
        "mean_expected_uv": [float(x) for x in expected[y0:y1, x0:x1].reshape(-1, 2).mean(axis=0)],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--godot-bin", default=None)
    p.add_argument("--project-dir", default="project")
    p.add_argument("--tier", choices=list(rvr.HARDWARE_TIERS), default="tier2")
    p.add_argument("--out-dir", default="tests/artifacts/motion_check")
    p.add_argument("--yaw-deg", type=float, default=0.5, help="Camera yaw per frame.")
    p.add_argument("--frame", type=int, default=24, help="Capture frame.")
    p.add_argument("--tolerance-px", type=float, default=0.25, help="Per-pixel error allowed, in internal pixels.")
    args = p.parse_args()

    repo = Path(__file__).resolve().parent.parent
    project = (repo / args.project_dir).resolve()
    out_dir = (repo / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tier = rvr.HARDWARE_TIERS[args.tier]
    godot = rvr.resolve_executable(args.godot_bin, ["godot", "godot4", "Godot_v4.7.2-stable_linux.x86_64"], "Godot 4 executable")
    internal_width = round(tier.internal_width)

    report: dict = {"scene": SCENE, "tier": tier.name, "yaw_deg_per_frame": args.yaw_deg, "tolerance_px": args.tolerance_px}
    failures: list[str] = []

    # 1. Calibration: zero motion must decode to zero. Detect an sRGB-encoded capture from the 0.5 offset.
    cal, _ = capture(godot, project, out_dir, tier, "calibration", 0.0, False, args.frame)
    centre = cal[cal.shape[0] // 3: 2 * cal.shape[0] // 3, cal.shape[1] // 3: 2 * cal.shape[1] // 3, :2]
    raw_offset = float(np.median(centre))
    srgb = abs(raw_offset - 0.7354) < abs(raw_offset - 0.5)
    report["capture_encoding"] = "srgb" if srgb else "linear"

    def decode(img: np.ndarray) -> np.ndarray:
        rg = img[..., :2].astype(np.float64)
        if srgb:
            rg = srgb_to_linear(rg)
        return (rg - 0.5) / SIGNED_VELOCITY_SCALE

    cal_px = np.abs(decode(cal)[cal.shape[0] // 3: 2 * cal.shape[0] // 3, cal.shape[1] // 3: 2 * cal.shape[1] // 3]) * internal_width
    report["calibration"] = {"raw_offset": raw_offset, "median_abs_px": float(np.median(cal_px))}
    if np.median(cal_px) > args.tolerance_px:
        failures.append(f"calibration: a static camera reads {np.median(cal_px):.3f} px of motion")

    # 2-3. Rotating camera, without and with jitter.
    for name, jitter in (("yaw", False), ("yaw_jitter", True)):
        img, log = capture(godot, project, out_dir, tier, name, args.yaw_deg, jitter, args.frame)
        m = CAMERA_RE.search(log)
        if not m:
            failures.append(f"{name}: the rig didn't log the motion-test camera")
            continue
        fov, keep_aspect = float(m.group(1)), int(m.group(2))
        expected = expected_velocity(img.shape[1], img.shape[0], fov, keep_aspect, parse_basis(m.group(3)), parse_basis(m.group(4)))
        result = evaluate(decode(img), expected, internal_width, args.tolerance_px)
        report[name] = result
        if result["median_error_px"] > args.tolerance_px or result["within_tolerance"] < 0.9:
            failures.append(f"{name}: median error {result['median_error_px']:.3f} px, "
                            f"{100 * result['within_tolerance']:.0f}% within {args.tolerance_px} px "
                            f"(measured {result['mean_measured_uv']}, expected {result['mean_expected_uv']})")

    report["passed"] = not failures
    report["failures"] = failures
    (out_dir / "motion_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Motion vectors: " + ("PASS" if not failures else "FAIL\n  " + "\n  ".join(failures)))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
