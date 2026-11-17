"""Every compute shader's push-constant block must match the C++ struct the effect uploads.

Godot rejects a push-constant upload whose size differs from the shader's block, and a mismatch in
layout silently scrambles parameters. The shader side comes from glslang's reflection; the C++ side
from the static_asserts next to each struct.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from conftest import REPO

SHADER_ROOT = REPO / "addons" / "prism_rp" / "shaders"
INCLUDE_DIR = SHADER_ROOT / "include"
CHRONOS_H = REPO / "addons/prism_rp/include/prism_rp/chronos_vpu_effect.h"
LUMINA_H = REPO / "addons/prism_rp/include/prism_rp/lumina_cascades_effect.h"

SHADER_TO_STRUCT = {
    "chronos_dilate_velocity": (CHRONOS_H, "DilateVelocityPushConstants"),
    "chronos_reactive_mask": (CHRONOS_H, "ReactiveMaskPushConstants"),
    "chronos_taau": (CHRONOS_H, "TAAUPushConstants"),
    "chronos_fft_bloom_horizontal": (CHRONOS_H, "FFTForwardHorizPushConstants"),
    "chronos_fft_bloom_vertical": (CHRONOS_H, "FFTVerticalKernelPushConstants"),
    "chronos_fft_bloom_inv_horizontal": (CHRONOS_H, "FFTInvHorizPushConstants"),
    "chronos_aces2_odt": (CHRONOS_H, "CompositePushConstants"),
    "lumina_rc_trace": (LUMINA_H, "TracePushConstants"),
    "lumina_rc_merge": (LUMINA_H, "MergePushConstants"),
    "lumina_rc_integrate": (LUMINA_H, "IntegratePushConstants"),
    "lumina_svgf_temporal": (LUMINA_H, "TemporalPushConstants"),
    "lumina_svgf_atrous": (LUMINA_H, "AtrousPushConstants"),
    "lumina_composite": (LUMINA_H, "CompositePushConstants"),
}

SHADERS = sorted(p for p in SHADER_ROOT.rglob("*.comp.glsl"))


def cpp_struct_sizes(header: Path) -> dict[str, int]:
    return {name: int(size) for name, size in re.findall(r"static_assert\(sizeof\((\w+)\)\s*==\s*(\d+)\)", header.read_text())}


def shader_push_constant_size(shader: Path) -> int:
    out = subprocess.run(
        ["glslangValidator", "-V", "-q", "--target-env", "vulkan1.3", "-S", "comp", f"-I{INCLUDE_DIR}", str(shader), "-o", "/dev/null"],
        capture_output=True, text=True, check=True,
    ).stdout
    m = re.search(r"^\s*PushConstants: .*?size (\d+)", out, re.M)
    assert m, f"no PushConstants block reflected for {shader.name}"
    return int(m.group(1))


def test_every_shader_is_mapped():
    unmapped = [s.name for s in SHADERS if s.name.removesuffix(".comp.glsl") not in SHADER_TO_STRUCT]
    assert not unmapped, f"add these shaders to SHADER_TO_STRUCT: {unmapped}"


@pytest.mark.skipif(shutil.which("glslangValidator") is None, reason="glslangValidator not installed")
@pytest.mark.parametrize("shader", SHADERS, ids=lambda p: p.name)
def test_push_constant_size_matches_cpp(shader):
    header, struct = SHADER_TO_STRUCT[shader.name.removesuffix(".comp.glsl")]
    sizes = cpp_struct_sizes(header)
    assert struct in sizes, f"{struct} has no static_assert(sizeof(...)) in {header.name}"
    assert shader_push_constant_size(shader) == sizes[struct]
