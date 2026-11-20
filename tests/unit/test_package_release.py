import stat
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "engine" / "package_release.py"

BUILT = [
    "godot.linuxbsd.template_release.x86_64.prism",
    "godot.linuxbsd.template_debug.x86_64.prism",
    "godot.windows.template_release.x86_64.prism.exe",
    "godot.windows.template_debug.x86_64.prism.exe",
]


def fake_bin(tmp_path, version="4.7.2.stable.prism.custom_build.ed1daf0bf", console=True):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    editor = bin_dir / "godot.linuxbsd.editor.x86_64.prism"
    editor.write_text(f"#!/bin/sh\necho {version}\n")
    editor.chmod(editor.stat().st_mode | stat.S_IXUSR)
    for name in BUILT:
        (bin_dir / name).write_bytes(name.encode())
    if console:
        (bin_dir / "godot.windows.template_release.x86_64.prism.console.exe").write_bytes(b"console")
    return bin_dir


def package(bin_dir, out_dir, release="engine-test"):
    return subprocess.run([sys.executable, str(SCRIPT), str(bin_dir), str(out_dir), release], capture_output=True, text=True)


def templates_version(tmp_path):
    with zipfile.ZipFile(tmp_path / "out" / "prism-godot-engine-test-templates.tpz") as tpz:
        return tpz.read("templates/version.txt")


def test_packages_the_editor_and_the_templates_under_godots_names(tmp_path):
    result = package(fake_bin(tmp_path), tmp_path / "out")
    assert result.returncode == 0, result.stderr

    with tarfile.open(tmp_path / "out" / "prism-godot-engine-test-linux-editor.tar.gz") as tar:
        member = tar.getmember("godot.linuxbsd.editor.x86_64.prism")
        assert member.mode & stat.S_IXUSR

    with zipfile.ZipFile(tmp_path / "out" / "prism-godot-engine-test-templates.tpz") as tpz:
        assert sorted(tpz.namelist()) == sorted([
            "templates/version.txt",
            "templates/linux_release.x86_64",
            "templates/linux_debug.x86_64",
            "templates/windows_release_x86_64.exe",
            "templates/windows_debug_x86_64.exe",
            "templates/windows_release_x86_64_console.exe",
        ])
        assert tpz.read("templates/windows_debug_x86_64.exe") == b"godot.windows.template_debug.x86_64.prism.exe"


@pytest.mark.parametrize("reported, folder", [
    ("4.7.2.stable.prism.custom_build.ed1daf0bf", "4.7.2.stable.prism"),
    ("4.7.2.stable.prism.custom_build", "4.7.2.stable.prism"),  # built without git, so no commit
    ("4.8.stable.prism.custom_build.0123abcde", "4.8.stable.prism"),  # a patch-0 release has fewer parts
    ("4.9.1.rc2.prism.custom_build.0123abcde", "4.9.1.rc2.prism"),
])
def test_version_txt_is_the_folder_the_editor_reads_templates_from(tmp_path, reported, folder):
    assert package(fake_bin(tmp_path, version=reported), tmp_path / "out").returncode == 0
    assert templates_version(tmp_path) == f"{folder}\n".encode()


def test_fails_when_a_build_output_is_missing(tmp_path):
    bin_dir = fake_bin(tmp_path)
    (bin_dir / "godot.windows.template_debug.x86_64.prism.exe").unlink()
    result = package(bin_dir, tmp_path / "out")
    assert result.returncode != 0
    assert "godot.windows.template_debug.x86_64.prism.exe" in result.stderr


def test_fails_on_a_stock_godot_editor(tmp_path):
    result = package(fake_bin(tmp_path, version="4.7.2.stable.official.abc123"), tmp_path / "out")
    assert result.returncode != 0
    assert "prism" in result.stderr
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("release", ["../escape", "a/b", "with space", "", "a;b", "x\ny"])
def test_rejects_a_release_name_that_could_leave_the_output_folder(tmp_path, release):
    result = package(fake_bin(tmp_path), tmp_path / "out", release)
    assert result.returncode != 0
    assert not (tmp_path / "out").exists()
