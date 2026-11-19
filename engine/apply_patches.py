#!/usr/bin/env python3
"""Reset engine/godot to its pinned commit, then apply engine/patches/*.patch in order."""
import subprocess
import sys
from pathlib import Path

ENGINE = Path(__file__).resolve().parent
GODOT = ENGINE / "godot"
PATCHES = ENGINE / "patches"


def git(*args, cwd=GODOT):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def main():
    # The superproject records the pinned commit; `update --force` checks it out
    # again, then reset and clean drop anything a previous run left behind.
    steps = (
        (["submodule", "update", "--init", "--depth", "1", "--force", "engine/godot"], ENGINE.parent),
        (["reset", "--hard", "HEAD"], GODOT),
        (["clean", "-fd"], GODOT),
    )
    for args, cwd in steps:
        result = git(*args, cwd=cwd)
        if result.returncode:
            sys.exit(f"git {args[0]} failed:\n{result.stderr}")

    patches = sorted(PATCHES.glob("[0-9][0-9][0-9][0-9]-*.patch"))
    for patch in patches:
        # git apply makes no commit, so no author identity is involved.
        result = git("apply", "--3way", "--index", str(patch))
        if result.returncode:
            sys.exit(f"patch {patch.name} does not apply:\n{result.stderr}")
        print(f"applied {patch.name}")
    print(f"{len(patches)} patch(es) applied to {git('describe', '--tags', '--always').stdout.strip()}")


if __name__ == "__main__":
    main()
