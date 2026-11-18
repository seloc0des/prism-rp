"""The Track B visibility-buffer raster pipelines: each geometry stage must feed its fragment shader.

Vulkan matches stage interfaces by location (and, for mesh shaders, by the perprimitiveEXT decoration). The
lavapipe harness only runs compute shaders, so these pairs are checked here: vertex -> fragment by linking
them with glslang, mesh -> fragment by comparing the preprocessed declarations (glslang doesn't check those).
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
VIS = REPO / "addons/prism_rp/shaders/visibility"
INCLUDE = REPO / "addons/prism_rp/shaders/include"

pytestmark = pytest.mark.skipif(shutil.which("glslangValidator") is None, reason="glslangValidator not installed")

# (geometry stage, fragment stage, masked)
PIPELINES = [
    ("vis_raster.vert.glsl", "vis_raster.frag.glsl", False),
    ("vis_raster_masked.vert.glsl", "vis_raster_masked.frag.glsl", True),
    ("vis_raster.mesh.glsl", "vis_raster_ms.frag.glsl", False),
    ("vis_raster_masked.mesh.glsl", "vis_raster_ms_masked.frag.glsl", True),
]

DECL = re.compile(r"layout\(location = (\d+)\)\s+(perprimitiveEXT\s+)?(?:flat\s+)?(in|out)\s+(\w+)\s+(\w+)(\[\])?;")


def preprocess(path: Path, stage: str) -> str:
    return subprocess.run(["glslangValidator", "-E", "-S", stage, f"-I{INCLUDE}", str(path)],
                          capture_output=True, text=True, check=True).stdout


def interface(source: str, direction: str) -> dict[int, tuple[bool, str]]:
    return {int(loc): (bool(per_prim), type_) for loc, per_prim, d, type_, _name, _arr in DECL.findall(source) if d == direction}


def test_every_raster_shader_is_in_a_pipeline():
    listed = {name for pair in PIPELINES for name in pair[:2]}
    on_disk = {p.name for p in VIS.glob("vis_raster*.glsl")}
    assert on_disk == listed


@pytest.mark.parametrize("geometry, fragment, masked", PIPELINES, ids=lambda v: str(v))
def test_stage_interfaces_match(tmp_path, geometry, fragment, masked):
    stage = "mesh" if geometry.endswith(".mesh.glsl") else "vert"
    outputs = interface(preprocess(VIS / geometry, stage), "out")
    frag_src = preprocess(VIS / fragment, "frag")
    inputs = interface(frag_src, "in")
    assert inputs, f"{fragment} declares no inputs"
    for loc, decl in inputs.items():
        assert outputs.get(loc) == decl, f"location {loc}: {fragment} reads {decl}, {geometry} writes {outputs.get(loc)}"
    if stage == "mesh":
        assert outputs[0][0], "the IDs must be per-primitive in the mesh path"

    # Opaque pipelines keep early depth testing; masked ones can't (discard must stop the depth write).
    assert ("early_fragment_tests" in frag_src) != masked
    assert ("discard" in frag_src) == masked

    if stage == "vert":  # glslang links vertex + fragment and checks the interface by name and type
        (tmp_path / "x.vert").write_text((VIS / geometry).read_text())
        (tmp_path / "x.frag").write_text((VIS / fragment).read_text())
        subprocess.run(["glslangValidator", "-V", "--target-env", "vulkan1.3", "-l", f"-I{INCLUDE}",
                        str(tmp_path / "x.vert"), str(tmp_path / "x.frag"), "-o", str(tmp_path / "x.spv")],
                       capture_output=True, text=True, check=True)
