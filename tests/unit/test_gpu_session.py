"""tests/gpu_session.py pieces that don't need Godot: packaging, stage bookkeeping, Godot download names."""

import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gpu_session as gs  # noqa: E402


def test_package_converts_screenshots_and_rewrites_report_links(tmp_path):
    from PIL import Image

    run = tmp_path / "session" / "run"
    shot_dir = run / "validation" / "01_cornell_box" / "chronos_default"
    shot_dir.mkdir(parents=True)
    Image.new("RGB", (8, 4), (200, 30, 10)).save(shot_dir / "captured.png")
    (shot_dir / "velocity.exr").write_bytes(b"not needed in the zip")
    (run / "validation" / "report.md").write_text("![x](01_cornell_box/chronos_default/captured.png)\n", encoding="utf-8")
    (shot_dir / "godot.log").write_text("log", encoding="utf-8")

    session = gs.Session(run)
    status, _, zip_path = gs.stage_package(session, "stamp")
    assert status == "ok" and zip_path == run.parent / "gpu_session_stamp.zip"
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        assert "validation/01_cornell_box/chronos_default/captured.jpg" in names
        assert not any(n.endswith((".png", ".exr")) for n in names)
        assert "validation/01_cornell_box/chronos_default/godot.log" in names and "session.json" in names
        assert z.read("validation/report.md").decode() == "![x](01_cornell_box/chronos_default/captured.jpg)\n"
    assert not list(run.rglob("*.zip.jpg"))  # temporary JPEGs cleaned up


def test_dependent_stages_skip_when_a_prerequisite_fails(tmp_path):
    session = gs.Session(tmp_path)

    def fails():
        raise gs.StageFailed("nope")

    session.run("godot", fails)
    ran = []
    session.run("project", lambda: ran.append(1) or ("ok", "", None), needs=("godot",))
    assert session.stages["godot"]["status"] == "failed" and session.stages["godot"]["detail"] == "nope"
    assert session.stages["project"]["status"] == "skipped" and not ran


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux asset names")
def test_godot_download_names(monkeypatch):
    monkeypatch.setattr(gs.platform, "machine", lambda: "x86_64")
    assert gs.godot_asset() == (f"Godot_v{gs.GODOT_VERSION}-stable_linux.x86_64.zip", f"Godot_v{gs.GODOT_VERSION}-stable_linux.x86_64")
    monkeypatch.setattr(gs.platform, "machine", lambda: "aarch64")
    assert gs.godot_asset()[1].endswith("linux.arm64")


def test_streamed_steps_echo_and_log_their_output(tmp_path, capsys):
    log = tmp_path / "step.log"
    proc = gs.run_streamed([sys.executable, "-c", "import sys; print('one'); print('two', file=sys.stderr); sys.exit(3)"], log)
    assert proc.returncode == 3
    assert "one" in proc.stdout and "two" in proc.stdout          # stderr merged
    assert log.read_text(encoding="utf-8").split() == ["one", "two"]
    out = capsys.readouterr().out
    assert "   | one" in out and "   | two" in out                 # echoed live
