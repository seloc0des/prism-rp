#!/usr/bin/env python3
"""Exported games: does Prism RP load and run in a release export, on Linux and Windows?

Everything else runs Prism RP inside the editor binary. An exported game loads the template_release
library instead, from next to the executable, and only the files the export packed. This check:

  1. copies the QA project to a scratch folder and installs the addon into it (tools/install_addon.py);
  2. writes Linux and Windows export presets and exports release builds of scene 01 (the main scene);
  3. checks each export ships its Prism RP library;
  4. runs the Linux build (under Xvfb when there is no display) and, if Wine is installed, the Windows
     build through Wine, and checks: the extension registered, the capture frame was saved, and the GPU
     timings show Chronos and Lumina running.

Needs Godot 4.7.2's export templates (--download-templates fetches them, about 1 GB, into Godot's
usual folder) and the release libraries (built if missing; Windows needs MinGW-w64).

    python3 tests/check_export.py --godot-bin godot --download-templates
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import install_addon  # noqa: E402

GODOT_VERSION = "4.7.2"
TEMPLATES_URL = (f"https://github.com/godotengine/godot/releases/download/{GODOT_VERSION}-stable/"
                 f"Godot_v{GODOT_VERSION}-stable_export_templates.tpz")
EXPORT_NAME = "prism_export"
TEMPLATE_FILES = {"linux": "linux_release.x86_64", "windows": "windows_release_x86_64.exe"}
PRESETS = {
    "linux": ("Linux", "Linux", f"linux/{EXPORT_NAME}.x86_64",
              {"binary_format/architecture": '"x86_64"', "binary_format/embed_pck": "false"}),
    "windows": ("Windows", "Windows Desktop", f"windows/{EXPORT_NAME}.exe",
                {"binary_format/architecture": '"x86_64"', "binary_format/embed_pck": "false",
                 "application/modify_resources": "false", "codesign/enable": "false"}),
}
REGISTERED = "[Prism RP] Registered"
FAILURE_PATTERNS = re.compile(r"(Can't open dynamic library|SCRIPT ERROR|Parse Error|GDExtension dynamic library not found)")


def templates_dir() -> Path:
    """Godot's export-template folder for this version on this OS."""
    version = f"{GODOT_VERSION}.stable"
    if sys.platform == "win32":
        return Path(os.environ["APPDATA"]) / "Godot" / "export_templates" / version
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Godot" / "export_templates" / version
    data = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return data / "godot" / "export_templates" / version


def ensure_templates(download: bool) -> Path:
    tdir = templates_dir()
    if all((tdir / f).exists() for f in TEMPLATE_FILES.values()):
        return tdir
    if not download:
        raise SystemExit(f"Godot {GODOT_VERSION} export templates aren't installed in {tdir}; "
                         "install them from the editor (Editor > Manage Export Templates) or pass --download-templates.")
    tdir.mkdir(parents=True, exist_ok=True)
    tpz = tdir.parent / f"Godot_v{GODOT_VERSION}-stable_export_templates.tpz"
    print(f"Downloading export templates (about 1 GB) to {tdir} ...", flush=True)
    urllib.request.urlretrieve(TEMPLATES_URL, tpz)
    with zipfile.ZipFile(tpz) as z:
        for info in z.infolist():
            if info.is_dir() or not info.filename.startswith("templates/"):
                continue
            target = tdir / info.filename[len("templates/"):]
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            if target.name.endswith((".x86_64", ".arm64")):
                target.chmod(0o755)
    tpz.unlink()
    return tdir


def presets_cfg(platforms: list[str]) -> str:
    out = []
    for i, plat in enumerate(platforms):
        name, platform_name, path, options = PRESETS[plat]
        out += [f"[preset.{i}]", "", f'name="{name}"', f'platform="{platform_name}"', "runnable=true",
                "dedicated_server=false", 'custom_features=""', 'export_filter="all_resources"',
                'include_filter="*.glsl"', 'exclude_filter=""', f'export_path="{path}"', "encrypt_pck=false",
                "encrypt_directory=false", "", f"[preset.{i}.options]", ""]
        out += [f"{k}={v}" for k, v in options.items()] + [""]
    return "\n".join(out)


def prepare_project(dest: Path, platforms: list[str]) -> None:
    def ignore(directory: str, names: list[str]) -> set[str]:
        here = Path(directory)
        skipped = {n for n in names if n in (".godot", "__pycache__")}
        if here == REPO / "project" / "addons":
            skipped.add("prism_rp")   # a link to the repo's addon; installed as a copy below
        return skipped

    shutil.copytree(REPO / "project", dest, ignore=ignore)
    install_addon.copy_addon(dest, platforms)
    (dest / "export_presets.cfg").write_text(presets_cfg(platforms), encoding="utf-8")


def run(cmd: list[str], timeout: float, env: dict | None = None) -> subprocess.CompletedProcess:
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, check=False)


def game_args(out: Path, winpath=lambda p: str(p)) -> list[str]:
    return ["--rendering-driver", "vulkan", "--resolution", "1280x720", "--fixed-fps", "60", "--disable-vsync",
            "--", "--prism-tier=tier1", "--prism-telemetry=on", "--qa-capture-frame=48", f"--qa-screenshot-out={winpath(out / 'captured.png')}",
            f"--qa-telemetry-out={winpath(out / 'telemetry.json')}", "--quit-after=51"]


def verify_run(log: str, screenshot: Path, telemetry: Path) -> tuple[bool, str]:
    """Did the exported game load Prism RP and run Chronos and Lumina?"""
    problems = []
    if REGISTERED not in log:
        problems.append("the extension didn't register (library not loaded?)")
    bad = FAILURE_PATTERNS.findall(log)
    if bad:
        problems.append("log: " + ", ".join(sorted(set(bad))))
    if not screenshot.exists():
        problems.append("no capture frame saved")
    timings = {}
    if telemetry.exists():
        timings = json.loads(telemetry.read_text(encoding="utf-8")).get("pass_timings_ms", {})
    ran = {k: v for k, v in timings.items() if v and float(v) > 0}
    for effect in ("chronos", "lumina"):
        if not any(effect in k for k in ran):
            problems.append(f"no {effect.capitalize()} timings")
    summary = ", ".join(f"{k} {float(v):.2f} ms" for k, v in sorted(ran.items()))
    return (not problems), ("; ".join(problems) if problems else f"ran: {summary}")


def run_exported(plat: str, exe: Path, out: Path) -> tuple[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    if plat == "linux":
        cmd = [str(exe), *game_args(out)]
        if not os.environ.get("DISPLAY") and shutil.which("xvfb-run"):
            cmd = ["xvfb-run", "-a", "-s", "-screen 0 1280x720x24", *cmd]
    else:
        wine = shutil.which("wine") or shutil.which("wine64")
        if wine is None:
            return "skipped", "Wine isn't installed: install wine to run the Windows build here"
        cmd = [wine, str(exe), *game_args(out, lambda p: "Z:" + str(p).replace("/", "\\"))]
    try:
        proc = run(cmd, timeout=600)
    except subprocess.TimeoutExpired:
        return "failed", "timed out after 10 minutes"
    log = proc.stdout + "\n" + proc.stderr
    (out / "game.log").write_text(log, encoding="utf-8")
    ok, detail = verify_run(log, out / "captured.png", out / "telemetry.json")
    return ("ok" if ok else "failed"), detail


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--godot-bin", default="godot")
    p.add_argument("--platforms", default="linux,windows")
    p.add_argument("--download-templates", action="store_true")
    p.add_argument("--skip-build", action="store_true", help="Use the release libraries already built.")
    p.add_argument("--no-run", default="", help="Comma-separated platforms to export but not run (e.g. windows in CI).")
    p.add_argument("--out-dir", default="tests/artifacts/export_check")
    args = p.parse_args()

    out_dir = (REPO / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    platforms = [x.strip() for x in args.platforms.split(",") if x.strip()]
    no_run = {x.strip() for x in args.no_run.split(",") if x.strip()}
    report: dict = {"godot": GODOT_VERSION, "platforms": {}}

    if not args.skip_build:
        for plat in platforms:
            if plat == "windows" and sys.platform != "win32" and shutil.which(install_addon.MINGW) is None:
                continue   # reported below as a missing library
            cmd = ["scons", f"platform={plat}", "target=template_release", "arch=x86_64", f"-j{os.cpu_count() or 4}"]
            if plat == "windows" and sys.platform != "win32":
                cmd.append("use_mingw=yes")
            print("$ " + " ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=REPO, check=True)
    for plat in list(platforms):
        lib = install_addon.ADDON / "bin" / install_addon.library_name(plat, "template_release")
        if not lib.exists():
            report["platforms"][plat] = {"status": "failed", "detail": f"{lib.name} isn't built"
                                         + (" (needs MinGW-w64)" if plat == "windows" else "")}
            platforms.remove(plat)

    ensure_templates(args.download_templates)
    with tempfile.TemporaryDirectory(prefix="prism_export_") as tmp:
        project = Path(tmp) / "project"
        prepare_project(project, platforms)
        imp = run([args.godot_bin, "--headless", "--path", str(project), "--import"], timeout=900)
        (out_dir / "import.log").write_text(imp.stdout + "\n" + imp.stderr, encoding="utf-8")
        for plat in platforms:
            name, _, rel, _ = PRESETS[plat]
            exe = out_dir / rel
            exe.parent.mkdir(parents=True, exist_ok=True)
            proc = run([args.godot_bin, "--headless", "--path", str(project), "--export-release", name, str(exe)], timeout=900)
            (out_dir / f"export_{plat}.log").write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
            lib = exe.parent / install_addon.library_name(plat, "template_release")
            if not exe.exists():
                report["platforms"][plat] = {"status": "failed", "detail": f"export failed (exit {proc.returncode}); see export_{plat}.log"}
                continue
            if not lib.exists():
                report["platforms"][plat] = {"status": "failed", "detail": f"export has no {lib.name} next to the executable"}
                continue
            if plat in no_run:
                report["platforms"][plat] = {"status": "ok", "detail": "exported with the Prism RP library (not run here)"}
                continue
            status, detail = run_exported(plat, exe, out_dir / f"run_{plat}")
            report["platforms"][plat] = {"status": status, "detail": detail}

    (out_dir / "export_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for plat, r in report["platforms"].items():
        print(f"{plat:8s} {r['status']:8s} {r['detail']}")
    failed = [p for p, r in report["platforms"].items() if r["status"] == "failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
