"""The engine release workflow builds exactly what engine/package_release.py packages, and what engine.yml builds."""
from pathlib import Path
import re
import sys

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "engine"))

import package_release  # noqa: E402

RELEASE = yaml.safe_load((REPO / ".github/workflows/engine-release.yml").read_text())
ENGINE = yaml.safe_load((REPO / ".github/workflows/engine.yml").read_text())


def release_matrix():
    return RELEASE["jobs"]["build"]["strategy"]["matrix"]["include"]


def test_release_builds_every_file_the_packager_requires():
    built = {entry["binary"] for entry in release_matrix()}
    required = {package_release.EDITOR, *package_release.TEMPLATES}
    assert built == required


def test_release_uses_the_same_build_commands_as_the_ci_builds():
    ci = {entry["binary"]: entry["scons"] for entry in ENGINE["jobs"]["templates"]["strategy"]["matrix"]["include"]}
    ci["godot.linuxbsd.editor.x86_64.prism"] = "platform=linuxbsd target=editor"
    for entry in release_matrix():
        scons = entry["scons"].split()
        # Only the release workflow builds the release templates with production=yes, as Godot's own templates are.
        if "target=template_release" in scons:
            assert "production=yes" in scons, entry["binary"]
            scons.remove("production=yes")
        else:
            assert "production=yes" not in scons, entry["binary"]
        assert scons == ci[entry["binary"]].split(), entry["binary"]


def test_build_artifacts_are_the_ones_the_package_job_downloads():
    steps = RELEASE["jobs"]["build"]["steps"]
    upload = next(s for s in steps if s.get("uses", "").startswith("actions/upload-artifact"))
    download = next(s for s in RELEASE["jobs"]["package"]["steps"] if s.get("uses", "").startswith("actions/download-artifact"))
    assert re.fullmatch(download["with"]["pattern"].replace("*", ".*"), upload["with"]["name"].replace("${{ matrix.name }}", "linux-editor"))
    assert download["with"]["merge-multiple"] is True


def test_package_job_calls_a_script_that_exists():
    package = next(s for s in RELEASE["jobs"]["package"]["steps"] if s.get("id") == "package")
    assert "engine/package_release.py" in package["run"]
    assert (REPO / "engine" / "package_release.py").is_file()


def test_only_the_package_job_can_write_and_only_an_engine_tag_publishes():
    assert RELEASE["permissions"] == {"contents": "read"}
    assert RELEASE["jobs"]["build"].get("permissions") is None
    assert RELEASE["jobs"]["package"]["permissions"] == {"contents": "write"}

    steps = RELEASE["jobs"]["package"]["steps"]
    publish = next(s for s in steps if s["name"] == "Publish GitHub release")
    assert publish["if"] == "startsWith(github.ref, 'refs/tags/engine-v')"
    assert "dist/*" in publish["run"]
    # The release name comes from the tag only for an engine-v tag, so a manual run on the addon's v* tag is a dev build.
    package = next(s for s in steps if s.get("id") == "package")
    assert "refs/tags/engine-v*)" in package["run"] and "dev-" in package["run"]
    assert "ref_type" not in package["run"] and "ref_type" not in publish["if"]


def test_release_tags_do_not_trigger_the_addon_release():
    addon = yaml.safe_load((REPO / ".github/workflows/release.yml").read_text())
    # PyYAML reads the key "on" as True.
    assert RELEASE[True]["push"]["tags"] == ["engine-v*"]
    assert addon[True]["push"]["tags"] == ["v*"]
