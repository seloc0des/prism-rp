#!/usr/bin/env python3
"""Package the custom Godot build for a GitHub release.

Usage: engine/package_release.py <bin-dir> <out-dir> <release-name>

<bin-dir> holds the files engine/build.py produced (engine/godot/bin, or the downloaded workflow artifacts merged
into one folder). Writes two files to <out-dir>:

  prism-godot-<release-name>-linux-editor.tar.gz   the Linux editor
  prism-godot-<release-name>-templates.tpz         the Linux and Windows export templates, for
                                                   Editor > Manage Export Templates > Install From File

The .tpz holds templates/version.txt and the templates under the names Godot looks for. The editor looks for its
templates in a folder named after its version without the build name and commit (4.7.2.stable.prism), the same
folder Install From File picks from version.txt, so the templates sit next to stock 4.7.2 ones.
"""
import re
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

EDITOR = "godot.linuxbsd.editor.x86_64.prism"
RELEASE_NAME = re.compile(r"[A-Za-z0-9._-]+")

# Built file -> name inside the .tpz. The console wrappers only exist for Windows and are optional.
TEMPLATES = {
    "godot.linuxbsd.template_release.x86_64.prism": "linux_release.x86_64",
    "godot.linuxbsd.template_debug.x86_64.prism": "linux_debug.x86_64",
    "godot.windows.template_release.x86_64.prism.exe": "windows_release_x86_64.exe",
    "godot.windows.template_debug.x86_64.prism.exe": "windows_debug_x86_64.exe",
}
OPTIONAL_TEMPLATES = {
    "godot.windows.template_release.x86_64.prism.console.exe": "windows_release_x86_64_console.exe",
    "godot.windows.template_debug.x86_64.prism.console.exe": "windows_debug_x86_64_console.exe",
}


def template_version(editor: Path) -> str:
    """The editor's templates folder name. `--version` prints <number>.<status>.prism.<build name>[.<commit>];
    the folder name is everything up to and including prism, for example 4.7.2.stable.prism or 4.8.stable.prism."""
    output = subprocess.run([str(editor), "--headless", "--version"], check=True, capture_output=True, text=True).stdout.strip()
    match = re.match(r"(\d+(?:\.\d+){1,2}\.\w+\.prism)(?:\.|$)", output)
    if match is None:
        sys.exit(f"{editor} reports '{output}', expected <number>.<status>.prism.<build>")
    return match.group(1)


def main(bin_dir: Path, out_dir: Path, release: str) -> None:
    if not RELEASE_NAME.fullmatch(release):
        sys.exit(f"release name '{release}' may only contain letters, digits, '.', '_' and '-'")
    missing = [name for name in (EDITOR, *TEMPLATES) if not (bin_dir / name).is_file()]
    if missing:
        sys.exit(f"missing from {bin_dir}: {', '.join(missing)}")
    version = template_version(bin_dir / EDITOR)
    out_dir.mkdir(parents=True, exist_ok=True)

    editor_archive = out_dir / f"prism-godot-{release}-linux-editor.tar.gz"
    with tarfile.open(editor_archive, "w:gz", dereference=True) as tar:
        tar.add(bin_dir / EDITOR, arcname=EDITOR)  # keeps the executable bit

    templates_archive = out_dir / f"prism-godot-{release}-templates.tpz"
    with zipfile.ZipFile(templates_archive, "w", zipfile.ZIP_DEFLATED) as tpz:
        tpz.writestr("templates/version.txt", version + "\n")
        for built, name in {**TEMPLATES, **OPTIONAL_TEMPLATES}.items():
            if (bin_dir / built).is_file():
                tpz.write(bin_dir / built, f"templates/{name}")
    print(editor_archive)
    print(templates_archive)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
