#!/usr/bin/env python3
"""
Prism RP Automated Visual Regression & Frame-Budget Gatekeeper

Runs the 5 canonical Prism RP benchmark scenes deterministically (--fixed-fps 60), has the
benchmark rig save the displayed frame and PrismTelemetry GPU timings at the capture frame
(each stage the sum of its passes' medians over the frames before it),
evaluates perceptual image error with NVIDIA FLIP (`flip-evaluator`), and enforces
Tier 1 / Tier 2 / Tier 3 frame-time budgets.

With --renderdoc, Godot runs under `renderdoccmd capture`; the rig triggers a capture through
RenderDoc's in-app API and, if RenderDoc's Python module is importable, per-pass GPU durations
are read from the capture and take precedence over the in-engine timestamps.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import flip_evaluator as flip
except ImportError as exc:
    raise SystemExit(
        "ERROR: 'flip-evaluator' is required. Install it via: pip install flip-evaluator"
    ) from exc


CANONICAL_SCENES: dict[str, str] = {
    "01_cornell_box": "res://benchmarks/01_cornell_box.tscn",
    "02_foliage_overdraw": "res://benchmarks/02_foliage_overdraw.tscn",
    "03_thin_geometry_aliasing": "res://benchmarks/03_thin_geometry_aliasing.tscn",
    "04_dynamic_sponza": "res://benchmarks/04_dynamic_sponza.tscn",
    "05_volumetrics_wboit": "res://benchmarks/05_volumetrics_wboit.tscn",
}


@dataclass(frozen=True)
class HardwareTierSpec:
    name: str
    description: str
    internal_width: int
    internal_height: int
    target_width: int
    target_height: int
    total_frame_budget_ms: float
    prism_pipeline_budget_ms: float
    pass_budgets_ms: dict[str, float]


HARDWARE_TIERS: dict[str, HardwareTierSpec] = {
    "tier1": HardwareTierSpec(
        name="tier1",
        description="Steam Deck / RDNA2 Handheld (720p -> 1080p, 60 FPS target)",
        internal_width=1280,
        internal_height=720,
        target_width=1920,
        target_height=1080,
        total_frame_budget_ms=16.60,
        prism_pipeline_budget_ms=13.50,
        pass_budgets_ms={
            "gpu_meshlet_cull": 0.90,
            "tile_material_shade": 2.80,
            "virtual_shadow_pages": 1.65,
            "lumina_cascades_gi": 3.55,
            "froxel_volumetrics": 1.40,
            "chronos_vpu": 2.45,
            "aces2_optics": 0.75,
        },
    ),
    "tier2": HardwareTierSpec(
        name="tier2",
        description="Mid-Range Desktop RTX 4060 Ti / RX 6700 (1080p -> 1440p, 90 FPS target)",
        internal_width=1920,
        internal_height=1080,
        target_width=2560,
        target_height=1440,
        total_frame_budget_ms=11.10,
        prism_pipeline_budget_ms=9.15,
        pass_budgets_ms={
            "gpu_meshlet_cull": 0.65,
            "tile_material_shade": 1.85,
            "virtual_shadow_pages": 1.10,
            "lumina_cascades_gi": 2.40,
            "froxel_volumetrics": 0.95,
            "chronos_vpu": 1.65,
            "aces2_optics": 0.55,
        },
    ),
    "tier3": HardwareTierSpec(
        name="tier3",
        description="Enthusiast RT GPU (1440p -> 4K, 120 FPS target)",
        internal_width=2560,
        internal_height=1440,
        target_width=3840,
        target_height=2160,
        total_frame_budget_ms=8.30,
        prism_pipeline_budget_ms=6.80,
        pass_budgets_ms={
            "gpu_meshlet_cull": 0.50,
            "tile_material_shade": 1.35,
            "virtual_shadow_pages": 0.85,
            "lumina_cascades_gi": 1.75,
            "froxel_volumetrics": 0.70,
            "chronos_vpu": 1.20,
            "aces2_optics": 0.45,
        },
    ),
}


@dataclass
class SceneEvalResult:
    scene: str
    tier: str
    passed: bool
    mean_flip_error: float | None = None
    max_flip_error: float | None = None
    total_prism_ms: float | None = None
    pass_timings_ms: dict[str, float] = field(default_factory=dict)
    vulkan_validation_errors: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    captured_png: str = ""
    diff_heatmap_png: str = ""


def resolve_executable(explicit_path: str | None, candidates: list[str], label: str) -> str:
    if explicit_path:
        resolved = shutil.which(explicit_path) or explicit_path
        if Path(resolved).exists():
            return str(Path(resolved).resolve())
        raise FileNotFoundError(f"{label} not found at explicit path: {explicit_path}")

    for candidate in candidates:
        found = shutil.which(candidate)
        if found:
            return found

    raise FileNotFoundError(
        f"Could not locate {label} on PATH (searched: {', '.join(candidates)})."
    )


def try_import_renderdoc_module() -> Any | None:
    py_path = os.environ.get("RENDERDOC_PYTHON_PATH")
    if py_path and py_path not in sys.path:
        sys.path.insert(0, py_path)
    try:
        import renderdoc as rd  # type: ignore[import-not-found]
        return rd
    except ImportError:
        return None


def map_to_canonical_stage(subpass_name: str) -> str:
    """Mirror of PrismTelemetry::_map_to_canonical_stage (prism_telemetry.cpp)."""
    name = subpass_name.lower()
    if name == "chronos_aces2_odt" or name.startswith("aces2_"):
        return "aces2_optics"
    prefixes = [
        (("chronos_",), "chronos_vpu"),
        (("lumina_", "svgf_"), "lumina_cascades_gi"),
        (("meshlet_", "visbuf_"), "gpu_meshlet_cull"),
        (("tile_shade_", "material_"), "tile_material_shade"),
        (("vcsp_", "shadow_"), "virtual_shadow_pages"),
        (("froxel_", "wboit_"), "froxel_volumetrics"),
    ]
    for starts, stage in prefixes:
        if name.startswith(starts):
            return stage
    return ""


def extract_timings_from_rdc(rd_mod: Any, rdc_path: Path) -> dict[str, float]:
    cap = rd_mod.OpenCaptureFile()
    status = cap.OpenFile(str(rdc_path), "", None)
    if status != rd_mod.ResultCode.Succeeded:
        raise RuntimeError(f"RenderDoc failed to open {rdc_path}: {status}")

    if not cap.LocalReplaySupport():
        cap.Shutdown()
        raise RuntimeError(f"Local GPU replay unsupported for capture: {rdc_path}")

    status, controller = cap.OpenCapture(rd_mod.ReplayOptions(), None)
    if status != rd_mod.ResultCode.Succeeded:
        cap.Shutdown()
        raise RuntimeError(f"RenderDoc failed to initialize replay controller: {status}")

    pass_timings_ms: dict[str, float] = {}
    try:
        available_counters = controller.EnumerateCounters()
        if rd_mod.GPUCounter.EventGPUDuration in available_counters:
            counter_results = controller.FetchCounters([rd_mod.GPUCounter.EventGPUDuration])
            eid_to_ms: dict[int, float] = {
                sample.eventId: sample.value.d * 1000.0
                for sample in counter_results
            }

            def walk_actions(actions: list[Any], active_region: str | None = None) -> None:
                for action in actions:
                    # Debug labels come from PrismTelemetry::begin_pass (e.g. "chronos_taau").
                    name = action.customName if action.customName else ""
                    region = map_to_canonical_stage(name) or active_region

                    if region and action.eventId in eid_to_ms:
                        pass_timings_ms[region] = pass_timings_ms.get(region, 0.0) + eid_to_ms[action.eventId]

                    if action.children:
                        walk_actions(action.children, region)

            walk_actions(controller.GetRootActions())

    finally:
        controller.Shutdown()
        cap.Shutdown()

    return pass_timings_ms


def telemetry_pass_timings(telemetry: dict[str, Any]) -> dict[str, float]:
    """Each stage's GPU ms from a telemetry export: the median over its recent frames, or for an
    export from before the median existed, the last frame's."""
    timings = telemetry.get("pass_timings_median_ms") or telemetry.get("pass_timings_ms", {})
    return {name: float(ms) for name, ms in timings.items()}


def capture_scene(
    godot_bin: str,
    renderdoccmd: str | None,
    project_dir: Path,
    scene_name: str,
    scene_res_path: str,
    tier: HardwareTierSpec,
    capture_frame: int,
    artifacts_dir: Path,
    extra_args: list[str] | None = None,
    render_scale: float | None = None,
    output_dir: Path | None = None,
) -> tuple[Path, dict[str, float], list[str]]:
    """Run one scene and return (screenshot, per-stage GPU ms, validation/script error lines).

    extra_args are appended to the rig's user args (after `--`); render_scale overrides the tier's
    internal/target ratio; output_dir overrides <artifacts>/<tier>/<scene>.
    """
    scene_art_dir = output_dir if output_dir is not None else artifacts_dir / tier.name / scene_name
    scene_art_dir.mkdir(parents=True, exist_ok=True)

    telemetry_json = scene_art_dir / "telemetry.json"
    captured_png = scene_art_dir / "captured.png"
    for stale in [telemetry_json, captured_png, *scene_art_dir.glob("*.rdc")]:
        stale.unlink(missing_ok=True)

    if render_scale is None:
        render_scale = round(tier.internal_width / float(tier.target_width), 4)
    cmd: list[str] = [
        godot_bin,
        "--rendering-driver", "vulkan",
        "--resolution", f"{tier.target_width}x{tier.target_height}",
        # Deterministic timing: fixed delta for physics/particles, no vsync pacing.
        "--fixed-fps", "60",
        "--disable-vsync",
        "--path", str(project_dir),
        scene_res_path,
        "--",
        f"--prism-tier={tier.name}",
        f"--prism-render-scale={render_scale}",
        f"--qa-capture-frame={capture_frame}",
        f"--qa-telemetry-out={telemetry_json}",
        f"--qa-screenshot-out={captured_png}",
        f"--quit-after={capture_frame + 3}",
        *(extra_args or []),
    ]

    if renderdoccmd:
        cmd = [
            renderdoccmd,
            "capture",
            "--wait-for-exit",
            "--capture-file", str(scene_art_dir / scene_name),
            *cmd,
        ]

    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        xvfb = shutil.which("xvfb-run")
        if xvfb:
            cmd = [
                xvfb,
                "-a",
                "-s",
                f"-screen 0 {tier.target_width}x{tier.target_height}x24",
                *cmd,
            ]

    env = os.environ.copy()
    env.setdefault("VK_INSTANCE_LAYERS", "VK_LAYER_KHRONOS_validation")

    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    combined_log = f"{proc.stdout}\n{proc.stderr}"
    log_path = scene_art_dir / "godot.log"
    log_path.write_text(combined_log, encoding="utf-8")

    validation_errors = [
        line.strip()
        for line in combined_log.splitlines()
        if re.search(r"(VUID-|Validation Error|SYNC-HAZARD|Can't open dynamic library|SCRIPT ERROR|Parse Error)", line)
    ]

    if not captured_png.exists():
        raise RuntimeError(
            f"No screenshot was written for {scene_name} (exit={proc.returncode}). Inspect {log_path}."
        )

    pass_timings_ms: dict[str, float] = {}
    if renderdoccmd:
        rdc_files = sorted(scene_art_dir.glob(f"{scene_name}*.rdc"))
        rd_mod = try_import_renderdoc_module()
        if not rdc_files:
            print(f"  !! RenderDoc produced no capture for {scene_name}; using in-engine timestamps.", file=sys.stderr)
        elif rd_mod is None:
            print("  !! RenderDoc Python module not importable (set RENDERDOC_PYTHON_PATH); using in-engine timestamps.", file=sys.stderr)
        else:
            pass_timings_ms = extract_timings_from_rdc(rd_mod, rdc_files[-1])

    if telemetry_json.exists():
        telemetry_data = json.loads(telemetry_json.read_text(encoding="utf-8"))
        for pass_name, ms_val in telemetry_pass_timings(telemetry_data).items():
            pass_timings_ms.setdefault(pass_name, ms_val)

    return captured_png, pass_timings_ms, validation_errors


def save_flip_magma_heatmap(error_map: np.ndarray, out_path: Path) -> None:
    e = np.clip(np.squeeze(error_map), 0.0, 1.0)
    r = np.clip(1.8 * e - 0.15, 0.0, 1.0)
    g = np.clip(2.2 * (e - 0.45), 0.0, 1.0)
    b = np.clip(np.sin(e * np.pi) * 0.85 + np.maximum(0.0, (e - 0.75) * 3.0), 0.0, 1.0)
    rgb_u8 = (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)
    Image.fromarray(rgb_u8).save(out_path)


def evaluate_flip_diff(
    golden_png: Path,
    captured_png: Path,
    diff_heatmap_png: Path,
) -> tuple[float, float]:
    with Image.open(golden_png) as g_img, Image.open(captured_png) as c_img:
        if g_img.size != c_img.size:
            raise ValueError(
                f"Resolution mismatch: golden {g_img.size} vs captured {c_img.size}. "
                "Re-run with --update-golden if the tier resolution changed."
            )

    dynamic_range = "HDR" if golden_png.suffix.lower() == ".exr" else "LDR"
    # applyMagma=False returns the scalar per-pixel error (HxW); the default is an RGB heatmap.
    flip_error_map, mean_flip_error, _params = flip.evaluate(
        str(golden_png),
        str(captured_png),
        dynamic_range,
        applyMagma=False,
    )

    error_arr = np.squeeze(np.asarray(flip_error_map, dtype=np.float32))
    max_flip_error = float(np.max(error_arr)) if error_arr.size > 0 else 0.0
    save_flip_magma_heatmap(error_arr, diff_heatmap_png)

    return float(mean_flip_error), max_flip_error


def load_baseline_timings(csv_path: Path) -> dict[str, dict[str, float]]:
    if not csv_path.exists():
        return {}
    baselines: dict[str, dict[str, float]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scene = row["scene"]
            pass_name = row["pass_name"]
            baselines.setdefault(scene, {})[pass_name] = float(row["gpu_ms"])
    return baselines


def save_baseline_timings(csv_path: Path, results: list[SceneEvalResult]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scene", "pass_name", "gpu_ms"])
        for res in results:
            for pass_name, ms in sorted(res.pass_timings_ms.items()):
                writer.writerow([res.scene, pass_name, f"{ms:.4f}"])


def run_regression_suite(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    project_dir = (repo_root / args.project_dir).resolve()
    golden_dir = (repo_root / args.golden_dir).resolve()
    artifacts_dir = (repo_root / args.artifacts_dir).resolve()

    golden_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    tier = HARDWARE_TIERS[args.tier]
    godot_bin = resolve_executable(
        args.godot_bin,
        ["godot", "godot4", "Godot_v4.7.2-stable_linux.x86_64", "Godot_v4.7.2-stable_win64.exe", "godot.windows.editor.x86_64.exe"],
        "Godot 4 executable",
    )
    renderdoccmd: str | None = None
    if args.renderdoc:
        renderdoccmd = resolve_executable(
            args.renderdoc_bin,
            ["renderdoccmd", "renderdoccmd.exe"],
            "RenderDoc CLI (renderdoccmd)",
        )

    selected_scenes = args.scenes if args.scenes else list(CANONICAL_SCENES.keys())
    for s in selected_scenes:
        if s not in CANONICAL_SCENES:
            raise ValueError(f"Unknown benchmark scene '{s}'. Valid: {list(CANONICAL_SCENES.keys())}")

    baseline_csv = golden_dir / f"baseline_timings_{tier.name}.csv"
    baseline_timings = load_baseline_timings(baseline_csv)

    print(f"=== Prism RP Visual & Frame-Budget QA ({tier.name.upper()}: {tier.description}) ===")
    print(f"Godot Binary : {godot_bin}")
    print(f"RenderDoc CLI: {renderdoccmd or 'disabled (pass --renderdoc to enable)'}")
    print(f"FLIP Gate    : mean <= {args.max_flip:.4f} | Pass Regression Gate: <= +{args.max_pass_delta_ms:.2f} ms\n")

    results: list[SceneEvalResult] = []

    for scene_name in selected_scenes:
        scene_res_path = CANONICAL_SCENES[scene_name]
        print(f"[{scene_name}] Capturing frame {args.capture_frame}...")

        captured_png, pass_timings_ms, validation_errors = capture_scene(
            godot_bin=godot_bin,
            renderdoccmd=renderdoccmd,
            project_dir=project_dir,
            scene_name=scene_name,
            scene_res_path=scene_res_path,
            tier=tier,
            capture_frame=args.capture_frame,
            artifacts_dir=artifacts_dir,
        )

        golden_png = golden_dir / f"{scene_name}_{tier.name}.png"
        diff_heatmap_png = artifacts_dir / tier.name / scene_name / "flip_diff_magma.png"

        eval_res = SceneEvalResult(
            scene=scene_name,
            tier=tier.name,
            passed=True,
            pass_timings_ms=pass_timings_ms,
            vulkan_validation_errors=validation_errors,
            captured_png=str(captured_png),
            diff_heatmap_png=str(diff_heatmap_png),
        )

        if validation_errors:
            eval_res.failures.append(
                f"Vulkan Validation / GDExtension errors detected ({len(validation_errors)}): {validation_errors[0]}"
            )

        if args.update_golden:
            shutil.copy2(captured_png, golden_png)
            print(f"  -> Updated golden reference: {golden_png.name}")

        if not golden_png.exists():
            eval_res.failures.append(
                f"Missing golden reference image: {golden_png}. Run with --update-golden to initialize."
            )
        else:
            try:
                mean_flip, max_flip = evaluate_flip_diff(golden_png, captured_png, diff_heatmap_png)
            except ValueError as exc:
                eval_res.failures.append(str(exc))
                mean_flip, max_flip = None, None
            if mean_flip is not None:
                eval_res.mean_flip_error = round(mean_flip, 6)
                eval_res.max_flip_error = round(max_flip, 6)
            if mean_flip is not None and mean_flip > args.max_flip:
                eval_res.failures.append(
                    f"Mean FLIP error {mean_flip:.5f} exceeds threshold {args.max_flip:.5f} "
                    f"(heatmap: {diff_heatmap_png})"
                )

        if pass_timings_ms:
            total_ms = sum(pass_timings_ms.values())
            eval_res.total_prism_ms = round(total_ms, 4)

            if total_ms > tier.prism_pipeline_budget_ms:
                eval_res.failures.append(
                    f"Total Prism RP GPU time {total_ms:.2f} ms exceeds {tier.name} budget "
                    f"{tier.prism_pipeline_budget_ms:.2f} ms"
                )

            scene_base = baseline_timings.get(scene_name, {})
            for pass_name, measured_ms in pass_timings_ms.items():
                hard_cap = tier.pass_budgets_ms.get(pass_name)
                if hard_cap is not None and measured_ms > hard_cap:
                    eval_res.failures.append(
                        f"Pass '{pass_name}' took {measured_ms:.2f} ms (tier cap: {hard_cap:.2f} ms)"
                    )

                if not args.update_golden and pass_name in scene_base:
                    delta_ms = measured_ms - scene_base[pass_name]
                    if delta_ms > args.max_pass_delta_ms:
                        eval_res.failures.append(
                            f"Pass '{pass_name}' regressed by +{delta_ms:.2f} ms "
                            f"(baseline {scene_base[pass_name]:.2f} ms -> {measured_ms:.2f} ms)"
                        )

        eval_res.passed = len(eval_res.failures) == 0
        results.append(eval_res)

        status_badge = "PASS" if eval_res.passed else "FAIL"
        flip_str = f"{eval_res.mean_flip_error:.5f}" if eval_res.mean_flip_error is not None else "N/A"
        gpu_str = f"{eval_res.total_prism_ms:.2f} ms" if eval_res.total_prism_ms is not None else "N/A"
        print(f"  -> [{status_badge}] Mean FLIP: {flip_str} | Total Prism GPU: {gpu_str}")
        for fail_msg in eval_res.failures:
            print(f"     * {fail_msg}", file=sys.stderr)

    if args.update_golden:
        save_baseline_timings(baseline_csv, results)
        print(f"\nSaved baseline pass timings to {baseline_csv}")

    report_path = artifacts_dir / f"qa_report_{tier.name}.json"
    report_payload = {
        "tier": asdict(tier),
        "max_flip_threshold": args.max_flip,
        "max_pass_delta_ms": args.max_pass_delta_ms,
        "all_passed": all(r.passed for r in results),
        "scenes": [asdict(r) for r in results],
    }
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    print(f"\nWrote machine-readable QA report: {report_path}")

    return 0 if all(r.passed for r in results) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Prism RP RenderDoc + NVIDIA FLIP visual and performance regression gates."
    )
    parser.add_argument(
        "--tier",
        choices=list(HARDWARE_TIERS.keys()),
        default="tier2",
        help="Target hardware performance tier (default: tier2).",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=list(CANONICAL_SCENES.keys()),
        help="Subset of canonical benchmark scenes to run (default: all 5).",
    )
    parser.add_argument(
        "--godot-bin",
        default=None,
        help="Path to Godot 4 binary (auto-detected from PATH if omitted).",
    )
    parser.add_argument(
        "--renderdoc-bin",
        default=None,
        help="Path to renderdoccmd binary (auto-detected from PATH if omitted; used with --renderdoc).",
    )
    parser.add_argument(
        "--renderdoc",
        action="store_true",
        help="Run Godot under renderdoccmd and read per-pass GPU durations from the capture.",
    )
    parser.add_argument(
        "--project-dir",
        default="project",
        help="Relative path to Godot QA project directory (default: project).",
    )
    parser.add_argument(
        "--golden-dir",
        default="tests/golden",
        help="Relative path to approved golden PNGs and baseline CSVs (default: tests/golden).",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="tests/artifacts",
        help="Relative path for screenshots, .rdc captures, FLIP heatmaps, and JSON reports.",
    )
    parser.add_argument(
        "--capture-frame",
        type=int,
        default=16,
        help="Frame index to capture after temporal history warm-up (default: 16).",
    )
    parser.add_argument(
        "--max-flip",
        type=float,
        default=0.015,
        help="Maximum allowed mean NVIDIA FLIP error vs. golden PNG (default: 0.015).",
    )
    parser.add_argument(
        "--max-pass-delta-ms",
        type=float,
        default=0.25,
        help="Maximum allowed single-pass GPU time increase vs. baseline CSV (default: 0.25 ms).",
    )
    parser.add_argument(
        "--update-golden",
        action="store_true",
        help="Overwrite golden reference PNGs and baseline timing CSVs with current run outputs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(run_regression_suite(parse_args()))
