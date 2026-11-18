#!/usr/bin/env python
import os
import subprocess
import sys
from pathlib import Path

# Godot API the extension is built against (godot-cpp v10 supports several; see prism_rp.gdextension).
GODOT_API_VERSION = "4.7"
env = SConscript("thirdparty/godot-cpp/SConstruct", {"api_version": GODOT_API_VERSION})
# Clone so our C++20 and warning flags do not leak into the godot-cpp library build.
env = env.Clone()

if env["platform"] == "windows" and not env.get("use_mingw", False):
    env.Append(CXXFLAGS=["/std:c++20", "/Zc:__cplusplus", "/W4"])
else:
    env.Append(CXXFLAGS=["-std=c++20", "-Wall", "-Wextra", "-Wno-unused-parameter"])

env.Append(CPPPATH=["addons/prism_rp/include", "thirdparty/renderdoc", "thirdparty/meshoptimizer/src"])
if env["platform"] == "linux":
    env.Append(LIBS=["dl"])  # dlopen/dlsym for the RenderDoc in-app API

sources = Glob("addons/prism_rp/src/*.cpp") + Glob("addons/prism_rp/src/meshlet/*.cpp") + Glob("thirdparty/meshoptimizer/src/*.cpp")

arch_suffix = env["arch"]
lib_filename = "libprism_rp.{}.{}.{}{}".format(
    env["platform"],
    env["target"],
    arch_suffix,
    env["SHLIBSUFFIX"],
)
output_path = os.path.join("addons", "prism_rp", "bin", lib_filename)

library = env.SharedLibrary(target=output_path, source=sources)
Default(library)

STAGE_MAP = {
    ".comp.glsl": "comp",
    ".vert.glsl": "vert",
    ".frag.glsl": "frag",
    ".mesh.glsl": "mesh",
    ".task.glsl": "task",
}

def validate_spirv_action(target, source, env):
    shader_root = Path("addons/prism_rp/shaders")
    include_dir = shader_root / "include"
    shader_files = [
        p for p in shader_root.rglob("*.glsl")
        if "include" not in p.parts
    ]

    if not shader_files:
        print("[Prism RP] No shader entrypoints found to validate.")
        return 0

    for shader in sorted(shader_files):
        stage = next((s for suffix, s in STAGE_MAP.items() if shader.name.endswith(suffix)), None)
        if stage is None:
            print(f"[Prism RP] ERROR: Unrecognized shader stage suffix on {shader}", file=sys.stderr)
            return 1

        cmd = [
            "glslangValidator",
            "-V",
            "--target-env", "vulkan1.3",
            "-S", stage,
            f"-I{include_dir}",
            str(shader),
            "-o", os.devnull,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[Prism RP] SPIR-V validation FAILED for {shader}:\n{result.stdout}\n{result.stderr}", file=sys.stderr)
            return result.returncode
        print(f"[Prism RP] Validated SPIR-V ({stage}): {shader}")

    return 0

shader_check = env.Alias("shaders", [], validate_spirv_action)
AlwaysBuild(shader_check)
