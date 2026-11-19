"""tools/install_addon.py: what lands in the user's project."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import install_addon as ia  # noqa: E402


def make_addon(root: Path) -> Path:
    addon = root / "addons" / "prism_rp"
    for rel in ("plugin.cfg", "prism_rp.gdextension", "scripts/dock.gd", "shaders/lumina/x.comp.glsl",
                "src/effect.cpp", "include/prism_rp/effect.h"):
        (addon / rel).parent.mkdir(parents=True, exist_ok=True)
        (addon / rel).write_text("x", encoding="utf-8")
    (addon / "bin").mkdir()
    for plat in ("linux", "windows", "macos"):
        for target in ia.TARGETS:
            (addon / "bin" / ia.library_name(plat, target)).write_bytes(b"lib")
    return addon


def test_copies_the_addon_without_sources_and_only_the_chosen_platforms(tmp_path, monkeypatch):
    monkeypatch.setattr(ia, "ADDON", make_addon(tmp_path / "repo"))
    project = tmp_path / "game"
    (project / "addons" / "prism_rp" / "bin").mkdir(parents=True)
    (project / "addons" / "prism_rp" / "bin" / "stale.so").write_bytes(b"old")   # an older install: replaced
    copied = ia.copy_addon(project, ["linux", "windows"])
    dest = project / "addons" / "prism_rp"
    assert copied == sorted(ia.library_name(p, t) for p in ("linux", "windows") for t in ia.TARGETS)
    assert (dest / "plugin.cfg").exists() and (dest / "shaders" / "lumina" / "x.comp.glsl").exists()
    assert (dest / "scripts" / "dock.gd").exists()
    assert not (dest / "src").exists() and not (dest / "include").exists()
    assert not (dest / "bin" / "stale.so").exists()


def test_library_names_match_the_extension_manifest():
    manifest = (Path(__file__).resolve().parents[2] / "addons" / "prism_rp" / "prism_rp.gdextension").read_text()
    for plat in ("linux", "windows"):
        for target in ia.TARGETS:
            assert f"res://addons/prism_rp/bin/{ia.library_name(plat, target)}" in manifest
