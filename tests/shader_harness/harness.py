"""Python side of the lavapipe compute-shader harness (see prism_shader_runner.cpp).

    h = Harness()
    out = h.run(
        images={"src": Img("rgba16f", data=np.ones((8, 8, 4))), "dst": Img("rgba16f", 8, 8)},
        dispatches=[Dispatch("util/prism_copy.comp.glsl", {0: ("src", "sampled_nearest"), 1: ("dst", "storage")},
                             pc=struct.pack("<ii8x", 8, 8), groups=(1, 1, 1))],
        outputs=["dst"],
    )
    out["dst"]  # float32 numpy array, shape (8, 8, 4)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SHADER_ROOT = REPO / "addons" / "prism_rp" / "shaders"
INCLUDE_DIR = SHADER_ROOT / "include"
BUILD_DIR = REPO / "build" / "shader_harness"
RUNNER = BUILD_DIR / "prism_shader_runner"

# format -> (channels, numpy storage dtype, normalized unorm)
FORMATS = {
    "r32f": (1, np.float32, False),
    "rg16f": (2, np.float16, False),
    "rgba16f": (4, np.float16, False),
    "rgba32f": (4, np.float32, False),
    "r8": (1, np.uint8, True),
    "rgba8": (4, np.uint8, True),
}


def toolchain_available() -> bool:
    return shutil.which("glslangValidator") is not None and shutil.which("g++") is not None and (
        Path("/usr/include/vulkan/vulkan.h").exists() and Path("/usr/include/nlohmann/json.hpp").exists()
    )


def build_runner() -> Path:
    src = HERE / "prism_shader_runner.cpp"
    if RUNNER.exists() and RUNNER.stat().st_mtime >= src.stat().st_mtime:
        return RUNNER
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(["g++", "-std=c++20", "-O1", "-Wall", "-Wextra", "-Wno-missing-field-initializers", str(src), "-o", str(RUNNER), "-lvulkan"], check=True)
    return RUNNER


def compile_shader(rel_path: str) -> Path:
    src = SHADER_ROOT / rel_path
    out = BUILD_DIR / "spv" / (rel_path.replace("/", "__") + ".spv")
    out.parent.mkdir(parents=True, exist_ok=True)
    newest_input = max([src.stat().st_mtime] + [p.stat().st_mtime for p in INCLUDE_DIR.glob("*.glsl")])
    if not out.exists() or out.stat().st_mtime < newest_input:
        subprocess.run(["glslangValidator", "-V", "--target-env", "vulkan1.3", "-S", "comp", f"-I{INCLUDE_DIR}",
                        str(src), "-o", str(out)], check=True, capture_output=True, text=True)
    return out


@dataclass
class Img:
    format: str
    width: int = 0
    height: int = 0
    data: np.ndarray | None = None   # (height, width, channels) float values (unorm: 0..1)

    def __post_init__(self):
        if self.data is not None:
            channels = FORMATS[self.format][0]
            arr = np.asarray(self.data, dtype=np.float64)
            if arr.ndim == 2:
                arr = arr[..., None]
            assert arr.shape[2] == channels, f"{self.format} needs {channels} channels, got {arr.shape}"
            self.height, self.width = arr.shape[:2]
            self.data = arr

    def to_bytes(self) -> bytes:
        channels, dtype, unorm = FORMATS[self.format]
        arr = self.data
        if unorm:
            arr = np.clip(np.round(arr * 255.0), 0, 255)
        return np.ascontiguousarray(arr.astype(dtype)).tobytes()


@dataclass
class Dispatch:
    shader: str                                   # path relative to addons/prism_rp/shaders
    bindings: dict[int, tuple[str, str]]          # binding -> (image name, kind)
    pc: bytes = b""
    groups: tuple[int, int, int] = (1, 1, 1)


def groups_for(w: int, h: int, local: int = 8) -> tuple[int, int, int]:
    return ((w + local - 1) // local, (h + local - 1) // local, 1)


@dataclass
class Harness:
    validation: bool = True
    keep_dir: Path | None = None
    _runner: Path = field(init=False)

    def __post_init__(self):
        self._runner = build_runner()

    def run(self, images: dict[str, Img], dispatches: list[Dispatch], outputs: list[str]) -> dict[str, np.ndarray]:
        work = Path(tempfile.mkdtemp(prefix="prism_harness_", dir=self.keep_dir))
        job = {"images": {}, "dispatches": [], "outputs": {}}
        for name, img in images.items():
            spec = {"format": img.format, "width": img.width, "height": img.height, "data": None}
            if img.data is not None:
                path = work / f"{name}.in"
                path.write_bytes(img.to_bytes())
                spec["data"] = str(path)
            job["images"][name] = spec
        for d in dispatches:
            job["dispatches"].append({
                "spv": str(compile_shader(d.shader)),
                "push_constants": d.pc.hex(),
                "groups": list(d.groups),
                "bindings": [{"binding": b, "image": img, "kind": kind} for b, (img, kind) in sorted(d.bindings.items())],
            })
        for name in outputs:
            job["outputs"][name] = str(work / f"{name}.out")
        (work / "job.json").write_text(json.dumps(job, indent=1))

        env = dict(os.environ)
        if self.validation:
            env["PRISM_VALIDATION"] = "1"
        proc = subprocess.run([str(self._runner), str(work / "job.json")], capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            raise RuntimeError(f"shader runner failed ({proc.returncode}):\n{proc.stderr}")

        result = {}
        for name in outputs:
            img = images[name]
            channels, dtype, unorm = FORMATS[img.format]
            arr = np.frombuffer((work / f"{name}.out").read_bytes(), dtype=dtype).reshape(img.height, img.width, channels)
            arr = arr.astype(np.float32)
            result[name] = arr / 255.0 if unorm else arr
        if self.keep_dir is None:
            shutil.rmtree(work, ignore_errors=True)
        return result
