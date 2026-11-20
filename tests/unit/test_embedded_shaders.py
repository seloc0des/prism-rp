"""The engine module's generated shader header: every Track B shader, includes inlined, still valid GLSL."""
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "engine" / "modules" / "prism_rp"))

import embed_shaders  # noqa: E402

SHADERS = REPO / "addons" / "prism_rp" / "shaders"
STAGES = {".comp.glsl": "comp", ".vert.glsl": "vert", ".frag.glsl": "frag", ".mesh.glsl": "mesh"}


def stage_of(path):
    return next(stage for suffix, stage in STAGES.items() if path.name.endswith(suffix))


def test_header_embeds_every_track_b_shader_without_includes():
    header = embed_shaders.header_text(SHADERS)
    entries = embed_shaders.entry_points(SHADERS)
    assert len(entries) == 17
    for path in entries:
        assert f'{{ "{path.name}", R"PRISM_GLSL(#version' in header
    assert "#include" not in header
    assert "#extension GL_GOOGLE_include_directive" not in header


def glslang(args):
    return subprocess.run(["glslangValidator", "-V", "--target-env", "vulkan1.3", *args], capture_output=True, text=True)


@pytest.mark.skipif(shutil.which("glslangValidator") is None, reason="glslangValidator not installed")
@pytest.mark.parametrize("path", embed_shaders.entry_points(SHADERS), ids=lambda p: p.name)
def test_inlined_source_compiles(path, tmp_path):
    stage = stage_of(path)
    # An older glslang may not know an extension the shader needs (GL_EXT_mesh_shader); then the original fails too.
    if glslang(["-S", stage, f"-I{SHADERS / 'include'}", str(path), "-o", str(tmp_path / "ref.spv")]).returncode != 0:
        pytest.skip("this glslangValidator cannot compile the original shader either")
    source = tmp_path / "inlined.glsl"
    source.write_text(embed_shaders.resolve_includes(path, SHADERS / "include"))
    result = glslang(["-S", stage, str(source), "-o", str(tmp_path / "out.spv")])
    assert result.returncode == 0, result.stdout + result.stderr
