"""The engine module's generated shader header: every Track B shader, includes inlined, still valid GLSL."""
from pathlib import Path
import re
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
        assert f'{{ "{path.name}",\n\t\tR"PRISM_GLSL(#version' in header
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


def test_literals_stay_under_the_msvc_piece_limit_and_rejoin_to_the_source():
    include = SHADERS / "include"
    for path in embed_shaders.entry_points(SHADERS):
        source = embed_shaders.resolve_includes(path, include)
        pieces = re.findall(r'R"PRISM_GLSL\((.*?)\)PRISM_GLSL"', embed_shaders.raw_literals(source), re.DOTALL)
        assert all(len(p.encode("utf-8")) < 16380 for p in pieces), path.name
        assert "".join(pieces) == source, path.name
    # At least one shader needs more than one piece, or the split is untested.
    big = SHADERS / "shadows" / "vsp_raster_pages.comp.glsl"
    assert len(embed_shaders.resolve_includes(big, include).encode("utf-8")) > 16380


def test_non_ascii_shaders_survive_the_header():
    non_ascii = [p for p in embed_shaders.entry_points(SHADERS) if any(ord(c) > 127 for c in embed_shaders.resolve_includes(p, SHADERS / "include"))]
    assert non_ascii
    header = embed_shaders.header_text(SHADERS)
    for path in non_ascii:
        for char in {c for c in embed_shaders.resolve_includes(path, SHADERS / "include") if ord(c) > 127}:
            assert char in header
