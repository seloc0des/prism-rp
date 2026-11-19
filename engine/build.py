#!/usr/bin/env python3
"""Apply the engine patches, then build the custom Godot with the Prism RP module.

Usage: engine/build.py [scons args...]   (defaults to platform=linuxbsd target=editor)

Other variants use the same entry point:
  engine/build.py platform=linuxbsd target=template_release
  engine/build.py platform=linuxbsd target=template_debug
  engine/build.py platform=windows use_mingw=yes d3d12=no target=template_release
"""
import os
import subprocess
import sys
from pathlib import Path

ENGINE = Path(__file__).resolve().parent
# Set SCONS to use a different checkout's venv (e.g. from a git worktree).
SCONS = os.environ.get("SCONS") or ENGINE.parent / ".venv" / "bin" / "scons"


def main():
    subprocess.run([sys.executable, str(ENGINE / "apply_patches.py")], check=True)
    args = sys.argv[1:] or ["platform=linuxbsd", "target=editor"]
    cmd = [str(SCONS), f"custom_modules={ENGINE / 'modules'}", "-j", str(os.cpu_count()), *args]
    # BUILD_NAME makes the version string read 4.7.2.stable.prism.
    env = {**os.environ, "BUILD_NAME": "prism"}
    sys.exit(subprocess.run(cmd, cwd=ENGINE / "godot", env=env).returncode)


if __name__ == "__main__":
    main()
