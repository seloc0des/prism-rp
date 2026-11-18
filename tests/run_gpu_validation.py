#!/usr/bin/env python3
"""
Prism RP one-shot GPU validation.

Runs every benchmark scene through a ladder of configurations (Godot baseline, each Chronos and
Lumina stage, every debug view, guards and fallbacks), then the Gate 1 reference comparison, and
writes a single markdown report with screenshots, GPU timings and every validation / script /
Prism RP warning line. Meant to be run once on a machine with a real GPU; see docs/GPU_TEST_PLAN.md
for what to look at in each image.

    python3 tests/run_gpu_validation.py --tier tier2
    python3 tests/run_gpu_validation.py --scenes 05_volumetrics_wboit --configs chronos_default lumina_default
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_visual_regression as rvr  # noqa: E402


@dataclass(frozen=True)
class RunConfig:
    name: str
    group: str
    description: str
    args: tuple[str, ...]
    render_scale: float | None = None
    scenes: tuple[str, ...] = ()  # only these scenes (empty: every scene)


CHRONOS_ONLY = ("--prism-lumina=off",)
LUMINA_ONLY = ("--prism-chronos=off",)


def _set(*pairs: str) -> str:
    return "--prism-set=" + ",".join(pairs)


CONFIGS: list[RunConfig] = [
    RunConfig("baseline_godot", "baseline", "Stock Forward+ with SDFGI, no Prism RP effects.",
              ("--prism-chronos=off", "--prism-lumina=off", "--prism-sdfgi=on")),

    RunConfig("chronos_default", "chronos", "Chronos with every stage on, jittered camera.", CHRONOS_ONLY),
    RunConfig("chronos_no_jitter", "chronos", "Same, camera jitter off (TAA becomes a temporal filter only).",
              (*CHRONOS_ONLY, "--prism-no-jitter")),
    RunConfig("chronos_taa_only", "chronos", "TAA only: bloom and SDR look off (output is linear, Godot tonemaps).",
              (*CHRONOS_ONLY, _set("chronos.fft_bloom_enabled=false", "chronos.aces_odt_enabled=false"))),
    RunConfig("chronos_no_taa", "chronos", "Bloom + SDR look without TAA (reference for aliasing).",
              (*CHRONOS_ONLY, _set("chronos.taau_enabled=false"))),
    RunConfig("chronos_no_capture", "chronos", "Opaque capture off: no transparency reactive input.",
              (*CHRONOS_ONLY, "--prism-opaque-capture=off")),
    RunConfig("chronos_godot_taa_guard", "chronos", "Godot TAA on: Chronos must warn once and bypass its TAA.",
              (*CHRONOS_ONLY, "--prism-godot-taa=on")),
    RunConfig("chronos_foliage_alpha_hash", "chronos", "Leaves alpha-hashed instead of alpha-scissored: Chronos must "
              "resolve the hash noise into soft, stable edges (compare with chronos_default).",
              (*CHRONOS_ONLY, "--prism-foliage-alpha=hash"), scenes=("02_foliage_overdraw",)),
    RunConfig("chronos_hdr", "chronos", "HDR output requested: the log shows whether Godot enabled it and the peak "
              "Chronos maps highlights to. The PNG is clamped to SDR; judge HDR on the monitor (gpu_session.py).",
              (*CHRONOS_ONLY, "--prism-hdr=on"), scenes=("01_cornell_box", "05_volumetrics_wboit")),
    RunConfig("chronos_debug_velocity", "chronos-debug", "Dilated velocity (static scene: near black; motion: smooth).",
              (*CHRONOS_ONLY, _set("chronos.debug_view_mode=1"))),
    RunConfig("chronos_debug_reactive", "chronos-debug", "Reactive mask (bright on sparks, glass, disocclusions).",
              (*CHRONOS_ONLY, _set("chronos.debug_view_mode=2"))),
    RunConfig("chronos_debug_bloom", "chronos-debug", "FFT bloom and halation only.",
              (*CHRONOS_ONLY, _set("chronos.debug_view_mode=3"))),
    RunConfig("chronos_debug_confidence", "chronos-debug", "TAA history confidence (green = stable history).",
              (*CHRONOS_ONLY, _set("chronos.debug_view_mode=4"))),

    RunConfig("lumina_default", "lumina", "Lumina GI with SVGF; Chronos off; SDFGI off.", LUMINA_ONLY),
    RunConfig("lumina_white_albedo", "lumina", "GI with white albedo: judges the lighting without the albedo proxy.",
              (*LUMINA_ONLY, _set("lumina.albedo_mode=1"))),
    RunConfig("lumina_no_denoise", "lumina", "Temporal off, 0 a-trous iterations: raw noise level.",
              (*LUMINA_ONLY, _set("lumina.temporal_enabled=false", "lumina.atrous_iterations=0"))),
    RunConfig("lumina_one_cascade", "lumina", "Only cascade 0: near-field GI, shows what upper levels add.",
              (*LUMINA_ONLY, _set("lumina.cascade_count=1"))),
    RunConfig("lumina_world_cache", "lumina", "World-space radiance cache on (off by default).",
              (*LUMINA_ONLY, _set("lumina.world_cascades_enabled=true"))),
    RunConfig("lumina_debug_world_only", "lumina-debug", "Irradiance from the world cache alone.",
              (*LUMINA_ONLY, _set("lumina.world_cascades_enabled=true", "lumina.debug_view_mode=7"))),
    RunConfig("lumina_debug_raw", "lumina-debug", "Raw integrated irradiance (noisy, no albedo).",
              (*LUMINA_ONLY, _set("lumina.debug_view_mode=1"))),
    RunConfig("lumina_debug_denoised", "lumina-debug", "Denoised irradiance.",
              (*LUMINA_ONLY, _set("lumina.debug_view_mode=2"))),
    RunConfig("lumina_debug_history", "lumina-debug", "SVGF history length (green = long history).",
              (*LUMINA_ONLY, _set("lumina.debug_view_mode=3"))),
    RunConfig("lumina_debug_variance", "lumina-debug", "SVGF variance estimate.",
              (*LUMINA_ONLY, _set("lumina.debug_view_mode=4"))),
    RunConfig("lumina_debug_normals", "lumina-debug", "Decoded view-space normals (x red, y green, z blue).",
              (*LUMINA_ONLY, _set("lumina.debug_view_mode=5"))),
    RunConfig("lumina_debug_albedo", "lumina-debug", "Albedo proxy.",
              (*LUMINA_ONLY, _set("lumina.debug_view_mode=6"))),

    RunConfig("full_pipeline", "full", "Lumina + Chronos + opaque capture together.", ()),
]

CONFIGS_BY_NAME = {c.name: c for c in CONFIGS}


@dataclass
class RunResult:
    scene: str
    config: str
    ok: bool
    screenshot: str = ""
    log: str = ""
    gpu_ms: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    prism_lines: list[str] = field(default_factory=list)
    exception: str = ""


@dataclass
class Gate1Result:
    scene: str
    mean_flip: float | None
    max_flip: float | None
    passed: bool
    reference_png: str = ""
    test_png: str = ""
    heatmap_png: str = ""
    note: str = ""


def run_config(godot_bin: str, project_dir: Path, out_root: Path, tier: rvr.HardwareTierSpec,
               scene: str, cfg: RunConfig, capture_frame: int) -> RunResult:
    out_dir = out_root / scene / cfg.name
    result = RunResult(scene=scene, config=cfg.name, ok=False)
    try:
        png, timings, errors = rvr.capture_scene(
            godot_bin=godot_bin,
            renderdoccmd=None,
            project_dir=project_dir,
            scene_name=scene,
            scene_res_path=rvr.CANONICAL_SCENES[scene],
            tier=tier,
            capture_frame=capture_frame,
            artifacts_dir=out_root,
            extra_args=list(cfg.args),
            render_scale=cfg.render_scale,
            output_dir=out_dir,
        )
        result.screenshot = str(png.relative_to(out_root))
        result.gpu_ms = timings
        result.errors = errors
    except Exception as exc:  # keep going: one broken config must not hide the rest
        result.exception = f"{type(exc).__name__}: {exc}"

    log_path = out_dir / "godot.log"
    if log_path.exists():
        result.log = str(log_path.relative_to(out_root))
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        result.prism_lines = [l.strip() for l in lines if "[Prism RP" in l][:20]
    result.ok = not result.exception and not result.errors
    return result


def run_gate1(godot_bin: str, project_dir: Path, out_root: Path, tier: rvr.HardwareTierSpec,
              scene: str, capture_frame: int, threshold: float) -> Gate1Result:
    """Gate 1: Chronos at 50% render scale vs a 2x supersampled, non-temporal reference.

    Both sides keep bloom and the SDR look so only resolution + TAA differ. Godot caps
    scaling_3d_scale at 2.0, so the reference is 2x2 supersampled at the tier's output size
    (the ROADMAP's "4K supersampled" for tier 2's 1440p output is approximated by this).
    """
    base = out_root / scene / "gate1"
    common = (*CHRONOS_ONLY,)
    ref = RunConfig("gate1_reference", "gate1", "", (*common, "--prism-no-jitter", _set("chronos.taau_enabled=false")), 2.0)
    test = RunConfig("gate1_test", "gate1", "", common, 0.5)
    res = Gate1Result(scene=scene, mean_flip=None, max_flip=None, passed=False)
    try:
        ref_png, _, _ = rvr.capture_scene(godot_bin, None, project_dir, scene, rvr.CANONICAL_SCENES[scene], tier,
                                          capture_frame, out_root, list(ref.args), ref.render_scale, base / "reference")
        test_png, _, _ = rvr.capture_scene(godot_bin, None, project_dir, scene, rvr.CANONICAL_SCENES[scene], tier,
                                           capture_frame, out_root, list(test.args), test.render_scale, base / "test")
        heat = base / "flip_magma.png"
        mean_flip, max_flip = rvr.evaluate_flip_diff(ref_png, test_png, heat)
        res.mean_flip, res.max_flip = round(mean_flip, 5), round(max_flip, 5)
        res.passed = mean_flip < threshold
        res.reference_png = str(ref_png.relative_to(out_root))
        res.test_png = str(test_png.relative_to(out_root))
        res.heatmap_png = str(heat.relative_to(out_root))
    except Exception as exc:
        res.note = f"{type(exc).__name__}: {exc}"
    return res


VALIDATION_ID = re.compile(r"(VUID-[A-Za-z0-9_-]+|SYNC-HAZARD-[A-Z_-]+)")


def finding_keys(lines: list[str]) -> set[str]:
    """Validation IDs where present, else the whole line: what to compare against the baseline."""
    keys = set()
    for line in lines:
        m = VALIDATION_ID.search(line)
        keys.add(m.group(1) if m else line.strip())
    return keys


def stage_total(gpu_ms: dict[str, float], stage: str) -> str:
    return f"{gpu_ms[stage]:.3f}" if stage in gpu_ms else "-"


def write_report(out_root: Path, tier: rvr.HardwareTierSpec, results: list[RunResult],
                 gate1: list[Gate1Result], gate1_threshold: float) -> Path:
    lines: list[str] = []
    lines.append(f"# Prism RP GPU validation ({tier.name})")
    lines.append("")
    lines.append(f"Generated {dt.datetime.now().isoformat(timespec='seconds')}. {tier.description}.")
    lines.append("What to look for in each image: `docs/GPU_TEST_PLAN.md`.")
    lines.append("")

    failed = [r for r in results if not r.ok]
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Runs: {len(results)}, with errors: {len(failed)}")
    if gate1:
        g_pass = sum(1 for g in gate1 if g.passed)
        lines.append(f"- Gate 1 (mean FLIP < {gate1_threshold}): {g_pass}/{len(gate1)} scenes")
    budget = tier.pass_budgets_ms
    lines.append(f"- Budgets ({tier.name}): chronos_vpu {budget['chronos_vpu']} ms, "
                 f"aces2_optics {budget['aces2_optics']} ms, lumina_cascades_gi {budget['lumina_cascades_gi']} ms")
    lines.append("")

    # Findings that also appear in the scene's stock-Godot run come from the engine, driver or
    # validation layers, not from Prism RP.
    baseline_keys = {r.scene: finding_keys(r.errors) for r in results if r.config == "baseline_godot"}
    if failed:
        lines.append("## Errors")
        lines.append("")
        for r in failed:
            new = finding_keys(r.errors) - baseline_keys.get(r.scene, set())
            tag = "" if r.exception or new or r.config == "baseline_godot" else " *(all also in baseline_godot: engine/driver, not Prism RP)*"
            lines.append(f"- **{r.scene} / {r.config}**: {r.exception or r.errors[0]}{tag}" + (f" ([log]({r.log}))" if r.log else ""))
        lines.append("")

    for scene in dict.fromkeys(r.scene for r in results):
        lines.append(f"## {scene}")
        lines.append("")
        lines.append("| Config | Image | Chronos ms | Optics ms | Lumina ms | Errors | Prism RP messages |")
        lines.append("| :--- | :--- | ---: | ---: | ---: | ---: | :--- |")
        for r in (x for x in results if x.scene == scene):
            cfg = CONFIGS_BY_NAME[r.config]
            img = f"[![{r.config}]({r.screenshot})]({r.screenshot})" if r.screenshot else "missing"
            msgs = "<br>".join(m.replace("|", "/") for m in r.prism_lines[:3]) or ""
            lines.append(
                f"| **{r.config}**<br>{cfg.description} | {img} | {stage_total(r.gpu_ms, 'chronos_vpu')} | "
                f"{stage_total(r.gpu_ms, 'aces2_optics')} | {stage_total(r.gpu_ms, 'lumina_cascades_gi')} | "
                f"{len(r.errors) + (1 if r.exception else 0)} | {msgs} |"
            )
        lines.append("")

    if gate1:
        lines.append("## Gate 1: Chronos at 50% vs 2x supersampled reference")
        lines.append("")
        lines.append("| Scene | Mean FLIP | Max FLIP | Pass | Reference | Chronos 50% | Error map |")
        lines.append("| :--- | ---: | ---: | :---: | :--- | :--- | :--- |")
        for g in gate1:
            if g.note:
                lines.append(f"| {g.scene} | - | - | no | {g.note} | | |")
                continue
            lines.append(f"| {g.scene} | {g.mean_flip} | {g.max_flip} | {'yes' if g.passed else 'no'} | "
                         f"[ref]({g.reference_png}) | [test]({g.test_png}) | [flip]({g.heatmap_png}) |")
        lines.append("")

    report = out_root / "report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_root / "results.json").write_text(json.dumps({
        "tier": tier.name,
        "runs": [asdict(r) for r in results],
        "gate1": [asdict(g) for g in gate1],
    }, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run every Prism RP stage on a real GPU and write one report.")
    p.add_argument("--tier", choices=list(rvr.HARDWARE_TIERS), default="tier2")
    p.add_argument("--scenes", nargs="+", choices=list(rvr.CANONICAL_SCENES))
    p.add_argument("--configs", nargs="+", choices=list(CONFIGS_BY_NAME))
    p.add_argument("--godot-bin", default=None)
    p.add_argument("--project-dir", default="project")
    p.add_argument("--out-dir", default=None, help="Default: tests/artifacts/gpu_validation/<timestamp>")
    p.add_argument("--capture-frame", type=int, default=48, help="Frame to capture (default 48, enough TAA/SVGF warm-up).")
    p.add_argument("--skip-gate1", action="store_true")
    p.add_argument("--gate1-max-flip", type=float, default=0.08)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    project_dir = (repo_root / args.project_dir).resolve()
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = Path(args.out_dir).resolve() if args.out_dir else repo_root / "tests" / "artifacts" / "gpu_validation" / stamp
    out_root.mkdir(parents=True, exist_ok=True)

    tier = rvr.HARDWARE_TIERS[args.tier]
    godot_bin = rvr.resolve_executable(
        args.godot_bin, ["godot", "godot4", "Godot_v4.7.2-stable_linux.x86_64", "Godot_v4.7.2-stable_win64.exe", "godot.windows.editor.x86_64.exe"],
        "Godot 4 executable")
    scenes = args.scenes or list(rvr.CANONICAL_SCENES)
    configs = [CONFIGS_BY_NAME[c] for c in args.configs] if args.configs else CONFIGS

    print(f"=== Prism RP GPU validation: {len(scenes)} scenes x {len(configs)} configs ({tier.name}) -> {out_root}")
    results: list[RunResult] = []
    for scene in scenes:
        for cfg in configs:
            if cfg.scenes and scene not in cfg.scenes:
                continue
            r = run_config(godot_bin, project_dir, out_root, tier, scene, cfg, args.capture_frame)
            status = "ok" if r.ok else "ERRORS"
            print(f"  [{status:6}] {scene} / {cfg.name}" + (f"  ({r.exception or r.errors[0]})" if not r.ok else ""))
            results.append(r)

    gate1: list[Gate1Result] = []
    if not args.skip_gate1:
        for scene in scenes:
            g = run_gate1(godot_bin, project_dir, out_root, tier, scene, args.capture_frame, args.gate1_max_flip)
            print(f"  [gate1 ] {scene}: mean FLIP {g.mean_flip} ({'pass' if g.passed else 'FAIL'}) {g.note}")
            gate1.append(g)

    report = write_report(out_root, tier, results, gate1, args.gate1_max_flip)
    print(f"\nReport: {report}")
    return 0 if all(r.ok for r in results) and all(g.passed for g in gate1) else 1


if __name__ == "__main__":
    sys.exit(main())
