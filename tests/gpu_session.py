#!/usr/bin/env python3
"""One-command GPU session for Prism RP. See docs/GPU_TEST_PLAN.md.

    python3 tests/gpu_session.py --tier tier2

Stages, each reported as it runs (a stage that can't run stops the ones that depend on it):

  system      OS, GPU, driver, display session, git revision
  python      the packages the checks need
  godot       Godot 4.7.2: --godot-bin, on PATH, cached under tests/artifacts/godot/, or downloaded
  build       the extension for this machine (template_debug)
  project     link the addon into project/ and import the project once
  motion      motion-vector check (tests/check_motion_vectors.py)
  validation  every scene x configuration, plus Gate 1 (tests/run_gpu_validation.py)
  settling    how many frames Lumina's bounce light takes to settle after a light comes on (tests/check_gi_convergence.py)
  export      your-game check: release exports of scene 01 for Linux and Windows, run (Windows through Wine
              if installed) to confirm Prism RP loads and runs in an exported game (tests/check_export.py)
  gate2       Lumina against a Blender Cycles reference, if Blender is installed (tests/run_gate2.py)
  hdr         by eye, if Godot can drive HDR here: scene 05 with HDR output, then SDR only
  package     one zip of everything (screenshots as JPEG)

--software runs on Mesa's lavapipe (CPU Vulkan) with a reduced scene / configuration set and without Gate 1,
Gate 2 or the HDR step. It tests this script and its plumbing, not the renderer's speed or looks; CI runs it.

Exit code: 0 when every stage ran and the motion check passed. Validation findings (Vulkan messages, Gate 1
FLIP) are recorded for review either way.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TESTS = REPO / "tests"
GODOT_VERSION = "4.7.2"
GODOT_RELEASE_URL = f"https://github.com/godotengine/godot/releases/download/{GODOT_VERSION}-stable/"
SOFTWARE_SCENES = ["01_cornell_box"]
SOFTWARE_CONFIGS = ["baseline_godot", "chronos_default", "lumina_default", "full_pipeline", "chronos_hdr"]
REQUIRED_MODULES = {"numpy": "numpy", "PIL": "pillow", "flip_evaluator": "flip-evaluator", "OpenEXR": "OpenEXR"}


class StageFailed(Exception):
    pass


class Session:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.stages: dict[str, dict] = {}
        self.info: dict = {}

    def run(self, name: str, fn, *args, needs: tuple[str, ...] = ()):
        missing = [n for n in needs if self.stages.get(n, {}).get("status") not in ("ok", "issues")]
        if missing:
            self.stages[name] = {"status": "skipped", "detail": f"needs {', '.join(missing)}"}
            print(f"\n== {name}: skipped (needs {', '.join(missing)})")
            return None
        print(f"\n== {name}")
        start = time.monotonic()
        try:
            status, detail, value = fn(*args)
        except StageFailed as e:
            status, detail, value = "failed", str(e), None
        except Exception as e:  # keep going: the package should still capture what happened
            status, detail, value = "failed", f"{type(e).__name__}: {e}", None
        self.stages[name] = {"status": status, "detail": detail, "seconds": round(time.monotonic() - start, 1)}
        print(f"   -> {status}: {detail}")
        return value

    def write(self) -> Path:
        path = self.out_dir / "session.json"
        path.write_text(json.dumps({"info": self.info, "stages": self.stages}, indent=2), encoding="utf-8")
        return path


def run(cmd: list[str], timeout: float | None = None, **kw) -> subprocess.CompletedProcess:
    print("   $ " + " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False, **kw)


def run_streamed(cmd: list[str], log_path: Path, timeout: float | None = None) -> subprocess.CompletedProcess:
    """Run a long step, echoing its output as it comes (so a 30-minute stage isn't silent) and saving it."""
    print("   $ " + " ".join(str(c) for c in cmd), flush=True)
    start = time.monotonic()
    lines: list[str] = []
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                          env={**os.environ, "PYTHONUNBUFFERED": "1"}) as proc, \
            open(log_path, "w", encoding="utf-8") as log:
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            lines.append(line)
            print("   | " + line.rstrip(), flush=True)
            if timeout is not None and time.monotonic() - start > timeout:
                proc.kill()
                raise StageFailed(f"timed out after {timeout:.0f} s; see {log_path.name}")
        proc.wait()
    return subprocess.CompletedProcess(cmd, proc.returncode, "".join(lines), "")


def sh_out(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False).stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


# --------------------------------------------------------------------------- stages

def stage_system(session: Session, software: bool):
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "session_type": os.environ.get("XDG_SESSION_TYPE", ""),
        "wayland_display": bool(os.environ.get("WAYLAND_DISPLAY")),
        "git_revision": sh_out(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"]),
        "git_dirty_files": len(sh_out(["git", "-C", str(REPO), "status", "--short"]).splitlines()),
        "software": software,
    }
    summary = sh_out(["vulkaninfo", "--summary"])
    devices = re.findall(r"deviceName\s*=\s*(.+)", summary)
    info["vulkan_devices"] = devices
    info["nvidia_smi"] = sh_out(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    session.info.update(info)
    (session.out_dir / "vulkaninfo_summary.txt").write_text(summary, encoding="utf-8")
    if not devices:
        return "issues", "vulkaninfo found no Vulkan device (install vulkan-tools to check; Godot may still work)", None
    gpu = info["nvidia_smi"] or devices[0]
    return "ok", f"{gpu}; display session: {info['session_type'] or 'unknown'}", None


def stage_python(_session: Session):
    missing = []
    for module, package in REQUIRED_MODULES.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        raise StageFailed(f"missing Python packages; run: {sys.executable} -m pip install {' '.join(missing)}")
    return "ok", "numpy, pillow, flip-evaluator, OpenEXR", None


def godot_version(path: str) -> str:
    if not (Path(path).exists() or shutil.which(path)):
        return ""
    lines = sh_out([path, "--version"]).splitlines()
    return lines[-1].strip() if lines else ""


def godot_asset() -> tuple[str, str]:
    """(zip name, executable inside) for this platform."""
    machine = platform.machine().lower()
    if sys.platform.startswith("linux"):
        arch = "arm64" if machine in ("aarch64", "arm64") else "x86_64"
        exe = f"Godot_v{GODOT_VERSION}-stable_linux.{arch}"
        return exe + ".zip", exe
    if sys.platform == "win32":
        exe = f"Godot_v{GODOT_VERSION}-stable_win64.exe"
        return exe + ".zip", exe
    raise StageFailed(f"no Godot download for {sys.platform}; pass --godot-bin")


def stage_godot(session: Session, explicit: str | None):
    candidates = [explicit] if explicit else ["godot", "godot4", f"Godot_v{GODOT_VERSION}-stable_linux.x86_64"]
    zip_name, exe_name = godot_asset() if not explicit else ("", "")
    cache = TESTS / "artifacts" / "godot" / GODOT_VERSION
    if exe_name:
        candidates.append(str(cache / exe_name))
    for c in candidates:
        path = shutil.which(c) or (c if Path(c).exists() else None)
        if path and godot_version(path).startswith(GODOT_VERSION):
            session.info["godot"] = {"path": path, "version": godot_version(path)}
            return "ok", f"{godot_version(path)} at {path}", path
        if path and explicit:
            raise StageFailed(f"{path} is {godot_version(path) or 'not runnable'}, need {GODOT_VERSION}")
    if explicit:
        raise StageFailed(f"--godot-bin {explicit} not found")
    cache.mkdir(parents=True, exist_ok=True)
    url = GODOT_RELEASE_URL + zip_name
    print(f"   downloading {url}")
    archive = cache / zip_name
    urllib.request.urlretrieve(url, archive)
    with zipfile.ZipFile(archive) as z:
        z.extractall(cache)
    archive.unlink()
    path = cache / exe_name
    path.chmod(0o755)
    version = godot_version(str(path))
    if not version.startswith(GODOT_VERSION):
        raise StageFailed(f"downloaded Godot reports '{version}'")
    session.info["godot"] = {"path": str(path), "version": version}
    return "ok", f"downloaded {version} to {path}", str(path)


def library_path() -> Path:
    plat = "windows" if sys.platform == "win32" else "linux"
    arch = "arm64" if platform.machine().lower() in ("aarch64", "arm64") else "x86_64"
    ext = "dll" if plat == "windows" else "so"
    return REPO / "addons" / "prism_rp" / "bin" / f"libprism_rp.{plat}.template_debug.{arch}.{ext}"


def stage_build(session: Session, skip: bool):
    lib = library_path()
    if skip:
        if not lib.exists():
            raise StageFailed(f"--skip-build, but {lib.relative_to(REPO)} doesn't exist")
        return "ok", f"using existing {lib.name}", None
    scons = shutil.which("scons")
    if not scons:
        raise StageFailed(f"SCons not found; run: {sys.executable} -m pip install scons")
    if not (REPO / "thirdparty" / "godot-cpp" / "SConstruct").exists():
        raise StageFailed("submodules missing; run: git submodule update --init --recursive")
    plat = "windows" if sys.platform == "win32" else "linux"
    proc = run([scons, f"platform={plat}", "target=template_debug", f"-j{os.cpu_count() or 4}"], cwd=REPO)
    (session.out_dir / "build.log").write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0 or not lib.exists():
        raise StageFailed(f"build failed (exit {proc.returncode}); see build.log")
    return "ok", f"built {lib.name}", None


def link_addon(link: Path, target: Path) -> str:
    """Make `link` (the QA project's addon folder) show `target`; returns how. Never trusts what is already there."""
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() and link.resolve() != target.resolve():
        try:  # dangling, or pointing at another checkout (e.g. a link committed from another machine)
            link.unlink()
        except OSError:
            os.rmdir(link)  # a directory symlink on Windows
    if link.is_symlink() or link.exists():
        how = "existing link" if link.is_symlink() else "existing folder"
        if not link.is_symlink() and link.resolve() != target.resolve():
            shutil.rmtree(link)  # a stale copy from an earlier run: refresh it
            shutil.copytree(target, link)
            how = "refreshed copy"
    else:
        try:
            os.symlink(target, link, target_is_directory=True)
            how = "symlink"
        except OSError:
            if sys.platform == "win32" and subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                                                          capture_output=True).returncode == 0:
                how = "junction"
            else:
                shutil.copytree(target, link)
                how = "copy"
    return how


def stage_project(session: Session, godot: str):
    how = link_addon(REPO / "project" / "addons" / "prism_rp", REPO / "addons" / "prism_rp")
    proc = run([godot, "--headless", "--path", str(REPO / "project"), "--import"], timeout=900)
    (session.out_dir / "import.log").write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    if not (REPO / "project" / ".godot" / "extension_list.cfg").exists():
        raise StageFailed("import didn't write .godot/extension_list.cfg (extension not found); see import.log")
    return "ok", f"addon linked ({how}), project imported", None


def stage_motion(session: Session, godot: str, tier: str):
    out = session.out_dir / "motion_check"
    proc = run_streamed([sys.executable, str(TESTS / "check_motion_vectors.py"), "--godot-bin", godot, "--tier", tier,
                         "--out-dir", str(out)], session.out_dir / "motion_check.log", timeout=1800)
    report = out / "motion_check.json"
    if not report.exists():
        raise StageFailed(f"no result (exit {proc.returncode}); see motion_check.log")
    data = json.loads(report.read_text(encoding="utf-8"))
    yaw = data.get("yaw", {})
    jit = data.get("yaw_jitter", {})
    detail = (f"median error {yaw.get('median_error_px', float('nan')):.3f} px "
              f"(with jitter {jit.get('median_error_px', float('nan')):.3f} px) "
              f"for ~{yaw.get('expected_motion_px', float('nan')):.1f} px of motion")
    if not data.get("passed"):
        raise StageFailed(detail + "; " + "; ".join(data.get("failures", [])))
    return "ok", detail, data


def stage_validation(session: Session, godot: str, tier: str, software: bool):
    out = session.out_dir / "validation"
    cmd = [sys.executable, str(TESTS / "run_gpu_validation.py"), "--tier", tier, "--godot-bin", godot, "--out-dir", str(out)]
    if software:
        cmd += ["--scenes", *SOFTWARE_SCENES, "--configs", *SOFTWARE_CONFIGS, "--skip-gate1"]
    proc = run_streamed(cmd, session.out_dir / "validation.log", timeout=6 * 3600)
    results = out / "results.json"
    if not results.exists():
        raise StageFailed(f"no results (exit {proc.returncode}); see validation.log")
    data = json.loads(results.read_text(encoding="utf-8"))
    runs = data.get("runs", [])
    crashed = [f"{r['scene']}/{r['config']}" for r in runs if r.get("exception")]
    sys.path.insert(0, str(TESTS))
    import run_gpu_validation as rgv

    with_errors = [r for r in runs if not r.get("ok")]
    prism = rgv.prism_caused_failures(runs)
    gate1 = data.get("gate1", [])
    detail = f"{len(runs)} runs, {len(prism)} with errors Prism RP causes"
    if len(with_errors) > len(prism):
        detail += f" ({len(with_errors) - len(prism)} more with engine/driver messages that plain Godot also has)"
    if gate1:
        detail += f"; Gate 1 {sum(1 for g in gate1 if g.get('passed'))}/{len(gate1)} scenes"
    if crashed:
        raise StageFailed(detail + f"; no image from: {', '.join(crashed[:6])}")
    return ("ok" if proc.returncode == 0 else "issues"), detail, None


MIN_BLENDER = (4, 0)  # tests/reference/cycles_reference.py uses node types and inputs from Blender 4


def blender_version(blender: str) -> tuple[int, int] | None:
    """(major, minor) from `blender --version`, or None if it can't be read."""
    try:
        proc = run([blender, "--version"], timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = re.search(r"Blender (\d+)\.(\d+)", proc.stdout)
    return (int(m.group(1)), int(m.group(2))) if m else None


def stage_gate2(session: Session, godot: str, tier: str, blender: str):
    version = blender_version(blender)
    if version is not None and version < MIN_BLENDER:
        return ("skipped", f"Blender {version[0]}.{version[1]} is too old: Gate 2 needs {MIN_BLENDER[0]}.x or newer "
                "(the Linux .tar.xz from blender.org runs without installing; pass its blender with --blender-bin)", None)
    out = session.out_dir / "gate2"
    proc = run_streamed([sys.executable, str(TESTS / "run_gate2.py"), "--tier", tier, "--godot-bin", godot,
                         "--blender-bin", blender, "--out-dir", str(out)], session.out_dir / "gate2.log", timeout=6 * 3600)
    report = out / "gate2.json"
    if report.exists():
        rows = json.loads(report.read_text(encoding="utf-8"))
        detail = "; ".join(f"{r['scene']}: {r['psnr_fitted_db']:.1f} dB fitted ({r.get('psnr_center_db', float('nan')):.1f} "
                           f"in the centre, scale {r['fitted_scale']:.2f})" for r in rows)
    else:
        psnr = re.findall(r"PSNR[^\d-]*([\d.]+)", proc.stdout)
        detail = f"PSNR {', '.join(psnr)} dB; see gate2.log" if psnr else "see gate2.log"
    if proc.returncode not in (0, 1):
        raise StageFailed(f"exit {proc.returncode}; {detail}")
    return ("ok" if proc.returncode == 0 else "issues"), detail, None


def stage_settling(session: Session, godot: str, tier: str):
    out = session.out_dir / "gi_convergence"
    proc = run_streamed([sys.executable, str(TESTS / "check_gi_convergence.py"), "--godot-bin", godot, "--tier", tier,
                         "--out-dir", str(out)], session.out_dir / "gi_convergence.log", timeout=1800)
    report = out / "gi_convergence.json"
    if not report.exists():
        raise StageFailed(f"no result (exit {proc.returncode}); see gi_convergence.log")
    data = json.loads(report.read_text(encoding="utf-8"))
    errors = data.get("errors", [])
    k = data.get("frames_to_settle")
    at_limit = errors[data["max_frames"]] if len(errors) > data["max_frames"] else float("nan")
    detail = (f"settles after {k} frames" if k is not None else f"not settled within {len(errors) - 1} frames") + \
        f" (error {100 * at_limit:.0f}% at {data['max_frames']} frames; needs under {100 * data['tolerance']:.0f}%)"
    return ("ok" if data.get("passed") else "issues"), detail, None


def stage_export(session: Session, godot: str):
    out = session.out_dir / "export_check"
    proc = run_streamed([sys.executable, str(TESTS / "check_export.py"), "--godot-bin", godot, "--download-templates",
                         "--out-dir", str(out)], session.out_dir / "export_check.log", timeout=3600)
    report = out / "export_check.json"
    if not report.exists():
        raise StageFailed(f"no result (exit {proc.returncode}); see export_check.log")
    platforms = json.loads(report.read_text(encoding="utf-8")).get("platforms", {})
    detail = "; ".join(f"{name}: {r['status']} ({r['detail']})" for name, r in platforms.items())
    failed = any(r["status"] == "failed" for r in platforms.values())
    return ("issues" if failed else "ok"), detail, None


def stage_hdr(session: Session, godot: str, tier: str, interactive: bool):
    sys.path.insert(0, str(TESTS))
    import run_visual_regression as rvr

    spec = rvr.HARDWARE_TIERS[tier]
    scene = "05_volumetrics_wboit"
    probe_dir = session.out_dir / "hdr" / "probe"
    try:
        rvr.capture_scene(godot, None, REPO / "project", scene, rvr.CANONICAL_SCENES[scene], spec, 30,
                          session.out_dir / "hdr", extra_args=["--prism-lumina=off", "--prism-hdr=on"], output_dir=probe_dir)
    except RuntimeError as e:
        raise StageFailed(f"HDR probe run failed: {e}") from e
    log = (probe_dir / "godot.log").read_text(encoding="utf-8", errors="replace")
    m = re.search(r"HDR output at capture: enabled=(\w+) max_linear=([\d.]+)", log)
    enabled = bool(m and m.group(1).lower() == "true")
    peak = float(m.group(2)) if m else 0.0
    session.info["hdr"] = {"enabled": enabled, "max_linear": peak}
    if not enabled:
        why = "on Linux this needs a Wayland session" if sys.platform.startswith("linux") else "check the OS display settings"
        return "skipped", f"Godot couldn't enable HDR output here ({why}; session type: {os.environ.get('XDG_SESSION_TYPE', '?')})", None
    if not interactive:
        return "ok", f"HDR output works (peak {peak:.2f}x SDR white); by-eye comparison not run (--non-interactive)", None

    def show(label: str, extra: list[str]) -> None:
        print(f"   showing {label} for about 15 s ...")
        subprocess.run([godot, "--rendering-driver", "vulkan", "--max-fps", "60", "--path", str(REPO / "project"), rvr.CANONICAL_SCENES[scene],
                        "--", "--prism-lumina=off", "--prism-hdr=on", "--quit-after=900", *extra],
                       capture_output=True, text=True, check=False)

    print(f"   Godot drives this display in HDR (peak {peak:.2f}x SDR white). Two runs of scene 05 follow:")
    print("   first with Chronos' look following the display peak, then limited to SDR. Watch the bright sparks and glass.")
    show("run 1 (HDR highlights)", [])
    show("run 2 (SDR only)", ["--prism-set=chronos.hdr_output=1"])
    answers = {
        "brighter_highlights": input("   In run 1, were highlights brighter than in run 2 without looking clipped? [y/n/unsure] ").strip(),
        "problems": input("   Anything wrong in run 1 (banding, colour shifts, flicker)? Describe, or press Enter: ").strip(),
    }
    session.info["hdr"]["answers"] = answers
    ok = answers["brighter_highlights"].lower().startswith("y") and not answers["problems"]
    return ("ok" if ok else "issues"), f"peak {peak:.2f}x; answers: {answers}", None


def stage_package(session: Session, stamp: str) -> tuple[str, str, Path]:
    from PIL import Image

    session.write()
    zip_path = session.out_dir.parent / f"gpu_session_{stamp}.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(session.out_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(session.out_dir)
            if path.suffix.lower() == ".png":
                tmp = path.with_suffix(".zip.jpg")
                Image.open(path).convert("RGB").save(tmp, quality=90)
                z.write(tmp, str(rel.with_suffix(".jpg")))
                tmp.unlink()
            elif path.suffix.lower() in (".exr", ".rdc"):
                continue
            elif path.name == "report.md":
                z.writestr(str(rel), path.read_text(encoding="utf-8").replace(".png)", ".jpg)"))
            else:
                z.write(path, str(rel))
    size_mb = zip_path.stat().st_size / 1e6
    shown = zip_path.relative_to(REPO) if zip_path.is_relative_to(REPO) else zip_path
    return "ok", f"{shown} ({size_mb:.1f} MB)", zip_path


# --------------------------------------------------------------------------- main

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tier", choices=["tier1", "tier2", "tier3"], default=None,
                   help="Hardware tier of this GPU (default tier2; tier1 with --software).")
    p.add_argument("--software", action="store_true", help="Run on lavapipe with a reduced set (CI).")
    p.add_argument("--godot-bin", default=None)
    p.add_argument("--blender-bin", default=None)
    p.add_argument("--skip-build", action="store_true", help="Use the library already in addons/prism_rp/bin.")
    p.add_argument("--skip-gate2", action="store_true")
    p.add_argument("--skip-export", action="store_true", help="Skip the exported-game check (downloads ~1 GB of templates once).")
    p.add_argument("--skip-hdr", action="store_true")
    p.add_argument("--non-interactive", action="store_true", help="Don't ask questions (HDR is checked, not judged).")
    p.add_argument("--out-dir", default=None, help="Default: tests/artifacts/gpu_session/<timestamp>")
    args = p.parse_args()

    tier = args.tier or ("tier1" if args.software else "tier2")
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else TESTS / "artifacts" / "gpu_session" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.software and sys.platform.startswith("linux") and not os.environ.get("VK_DRIVER_FILES"):
        icds = sorted(glob.glob("/usr/share/vulkan/icd.d/lvp_icd*.json"))
        if icds:
            os.environ["VK_DRIVER_FILES"] = icds[0]

    print(f"Prism RP GPU session ({'software, lavapipe' if args.software else tier}) -> {out_dir}")
    s = Session(out_dir)
    s.info["tier"] = tier
    s.run("system", stage_system, s, args.software)
    s.run("python", stage_python, s)
    godot = s.run("godot", stage_godot, s, args.godot_bin)
    s.run("build", stage_build, s, args.skip_build)
    s.run("project", stage_project, s, godot, needs=("godot", "build"))
    s.run("motion", stage_motion, s, godot, tier, needs=("python", "project"))
    s.run("validation", stage_validation, s, godot, tier, args.software, needs=("python", "project"))

    s.run("settling", stage_settling, s, godot, tier, needs=("python", "project"))

    if args.software or args.skip_export:
        why = "software mode (CI's export-check job covers exports)" if args.software else "--skip-export"
        s.stages["export"] = {"status": "skipped", "detail": why}
        print(f"\n== export: skipped ({why})")
    else:
        s.run("export", stage_export, s, godot, needs=("python", "project"))

    blender = args.blender_bin or shutil.which("blender")
    if args.software or args.skip_gate2 or not blender:
        why = "software mode" if args.software else ("--skip-gate2" if args.skip_gate2 else "Blender not found (pass --blender-bin)")
        s.stages["gate2"] = {"status": "skipped", "detail": why}
        print(f"\n== gate2: skipped ({why})")
    else:
        s.run("gate2", stage_gate2, s, godot, tier, blender, needs=("python", "project"))

    if args.software or args.skip_hdr:
        s.stages["hdr"] = {"status": "skipped", "detail": "software mode" if args.software else "--skip-hdr"}
        print("\n== hdr: skipped")
    else:
        s.run("hdr", stage_hdr, s, godot, tier, not args.non_interactive and sys.stdin.isatty(), needs=("python", "project"))

    zip_path = s.run("package", stage_package, s, stamp)

    print("\n" + "=" * 72)
    for name, st in s.stages.items():
        print(f"  {name:11} {st['status']:8} {st['detail']}")
    print("=" * 72)
    if zip_path:
        print(f"Results: {zip_path}")
    s.write()
    essential = ["godot", "build", "project", "motion", "validation", "package"]
    return 0 if all(s.stages.get(n, {}).get("status") in ("ok", "issues") for n in essential) else 1


if __name__ == "__main__":
    sys.exit(main())
