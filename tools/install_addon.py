#!/usr/bin/env python3
"""Put Prism RP into one of your own Godot 4.7 projects.

Builds the extension and copies addons/prism_rp, with the libraries, into <project>/addons/prism_rp
(replacing an older copy). Then enable "Prism RP" under Project Settings > Plugins; docs/USING.md
covers the rest.

Libraries built (skip with --skip-build to copy what addons/prism_rp/bin already holds):
  linux    template_debug (editor and debug exports) and template_release (release exports)
  windows  the same two, cross-compiled with MinGW-w64 (Debian / Ubuntu / Pop!_OS package:
           g++-mingw-w64-x86-64-posix). The DLLs need only KERNEL32 and msvcrt, so they run on
           any Windows PC.

    python3 tools/install_addon.py ~/games/my_game
    python3 tools/install_addon.py ~/games/my_game --platforms linux
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ADDON = REPO / "addons" / "prism_rp"
TARGETS = ("template_debug", "template_release")
MINGW = "x86_64-w64-mingw32-g++"


def library_name(platform: str, target: str) -> str:
    suffix = ".dll" if platform == "windows" else ".so"
    return f"libprism_rp.{platform}.{target}.x86_64{suffix}"


def build(platform: str, jobs: int) -> None:
    for target in TARGETS:
        cmd = ["scons", f"platform={platform}", f"target={target}", "arch=x86_64", f"-j{jobs}"]
        if platform == "windows" and sys.platform != "win32":
            cmd.append("use_mingw=yes")
        print("$ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=REPO, check=True)


def copy_addon(project: Path, platforms: list[str]) -> list[str]:
    """Copies the addon (sources excluded) into project/addons/prism_rp; returns the libraries copied."""
    dest = project / "addons" / "prism_rp"
    if dest.is_symlink():
        dest.unlink()
    elif dest.exists():
        shutil.rmtree(dest)
    wanted = {library_name(p, t) for p in platforms for t in TARGETS}

    def ignore(directory: str, names: list[str]) -> set[str]:
        here = Path(directory)
        skipped = {n for n in names if n in ("src", "include", "__pycache__")} if here == ADDON else set()
        if here == ADDON / "bin":
            skipped |= {n for n in names if n.startswith("libprism_rp.") and n not in wanted}
        return skipped

    shutil.copytree(ADDON, dest, ignore=ignore)
    return sorted(n for n in os.listdir(dest / "bin") if n.startswith("libprism_rp."))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("project", help="Your Godot project's folder (the one with project.godot).")
    p.add_argument("--platforms", default="linux,windows", help="Comma-separated: linux, windows.")
    p.add_argument("--skip-build", action="store_true", help="Copy the libraries already built.")
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    args = p.parse_args()

    project = Path(args.project).expanduser().resolve()
    if not (project / "project.godot").exists():
        print(f"{project} has no project.godot: point this at your Godot project's folder.")
        return 2
    platforms = [x.strip() for x in args.platforms.split(",") if x.strip()]
    unknown = set(platforms) - {"linux", "windows"}
    if unknown:
        print(f"unknown platform(s): {', '.join(sorted(unknown))}")
        return 2
    if "windows" in platforms and sys.platform != "win32" and not args.skip_build and shutil.which(MINGW) is None:
        print(f"Windows libraries need MinGW-w64 ({MINGW}); install g++-mingw-w64-x86-64-posix, "
              "or use --platforms linux.")
        return 2

    if not args.skip_build:
        for platform in platforms:
            build(platform, args.jobs)
    copied = copy_addon(project, platforms)
    missing = sorted({library_name(pl, t) for pl in platforms for t in TARGETS} - set(copied))
    print(f"\nInstalled Prism RP into {project / 'addons' / 'prism_rp'}")
    print("Libraries: " + (", ".join(copied) or "none"))
    if missing:
        print("Missing (that platform won't load Prism RP): " + ", ".join(missing))
    print("Next: Project Settings > Plugins > enable Prism RP; see docs/USING.md in the Prism RP repository.")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
