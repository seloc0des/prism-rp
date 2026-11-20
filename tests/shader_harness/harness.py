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
import re
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

# Size of the shaders' u_textures[] array, read from the shared include so the two cannot drift apart.
TEXTURE_ARRAY_SIZE = int(re.search(r"PRISM_MAX_MATERIAL_TEXTURES\s*=\s*(\d+)u", (INCLUDE_DIR / "vis_shading_common.glsl").read_text()).group(1))

# format -> (channels, numpy storage dtype, normalized unorm). Integer formats (uint32) come back as uint32.
FORMATS = {
    "r32f": (1, np.float32, False),
    "rg16f": (2, np.float16, False),
    "rgba16f": (4, np.float16, False),
    "rgba32f": (4, np.float32, False),
    "r8": (1, np.uint8, True),
    "rgba8": (4, np.uint8, True),
    "r32ui": (1, np.uint32, False),
    "rg32ui": (2, np.uint32, False),
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
    """`rel_path` is relative to addons/prism_rp/shaders, or absolute (test-only shaders)."""
    src = Path(rel_path) if Path(rel_path).is_absolute() else SHADER_ROOT / rel_path
    tag = str(src.relative_to(REPO)) if src.is_relative_to(REPO) else str(src)
    out = BUILD_DIR / "spv" / (tag.replace("/", "__") + ".spv")
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
    data: np.ndarray | None = None   # mip 0: (height, width, channels), or (layers, height, width, channels)
    mips: int = 1
    layers: int = 1
    array: bool | None = None        # 2D-array view; defaults to layers > 1

    def __post_init__(self):
        if self.data is not None:
            channels, dtype, _ = FORMATS[self.format]
            arr = np.asarray(self.data, dtype=np.uint64 if dtype == np.uint32 else np.float64)
            if arr.ndim == 2 or (arr.ndim == 3 and self.layers > 1):
                arr = arr[..., None]
            if arr.ndim == 3:
                arr = arr[None]
            assert arr.shape[0] == self.layers, f"expected {self.layers} layers, got {arr.shape}"
            assert arr.shape[3] == channels, f"{self.format} needs {channels} channels, got {arr.shape}"
            self.height, self.width = arr.shape[1:3]
            self.data = arr

    def to_bytes(self) -> bytes:
        channels, dtype, unorm = FORMATS[self.format]
        arr = self.data
        if unorm:
            arr = np.clip(np.round(arr * 255.0), 0, 255)
        return np.ascontiguousarray(arr.astype(dtype)).tobytes()

    def mip_shape(self, m: int) -> tuple[int, int, int, int]:
        return (self.layers, max(1, self.height >> m), max(1, self.width >> m), FORMATS[self.format][0])


@dataclass
class Buf:
    """A storage / uniform / indirect buffer. Comes back from run() as bytes."""
    size: int = 0
    data: bytes | None = None

    def __post_init__(self):
        if self.data is not None:
            self.size = max(self.size, len(self.data))
            self.data = bytes(self.data) + bytes(self.size - len(self.data))
        assert self.size > 0, "buffer needs a size or data"


@dataclass
class Dispatch:
    shader: str                                   # path relative to addons/prism_rp/shaders, or absolute
    # binding -> (image name, kind) | (image name, "storage", mip) | ([image names], sampled kind) |
    #            (buffer name, "storage_buffer" | "uniform_buffer")
    bindings: dict[int, tuple]
    pc: bytes = b""
    groups: tuple[int, int, int] = (1, 1, 1)
    indirect: tuple[str, int] | None = None       # (buffer name, byte offset): vkCmdDispatchIndirect


def groups_for(w: int, h: int, local: int = 8) -> tuple[int, int, int]:
    return ((w + local - 1) // local, (h + local - 1) // local, 1)


@dataclass
class Harness:
    validation: bool = True
    keep_dir: Path | None = None
    _runner: Path = field(init=False)

    def __post_init__(self):
        self._runner = build_runner()

    def run(self, images: dict[str, Img], dispatches: list[Dispatch], outputs: list[str],
            buffers: dict[str, Buf] | None = None) -> dict:
        """Returns images as float32 arrays (uint32 for integer formats) shaped (h, w, c), with a leading layer
        axis when layers > 1 and as a list per mip when mips > 1; buffers come back as bytes."""
        buffers = buffers or {}
        work = Path(tempfile.mkdtemp(prefix="prism_harness_", dir=self.keep_dir))
        job = {"images": {}, "buffers": {}, "dispatches": [], "outputs": {}}
        for name, img in images.items():
            spec = {"format": img.format, "width": img.width, "height": img.height, "mips": img.mips,
                    "layers": img.layers, "array": img.layers > 1 if img.array is None else img.array, "data": None}
            if img.data is not None:
                path = work / f"{name}.in"
                path.write_bytes(img.to_bytes())
                spec["data"] = str(path)
            job["images"][name] = spec
        for name, buf in buffers.items():
            spec = {"size": buf.size, "data": None}
            if buf.data is not None:
                path = work / f"{name}.buf"
                path.write_bytes(buf.data)
                spec["data"] = str(path)
            job["buffers"][name] = spec
        for d in dispatches:
            binds = []
            has_texture_array = "u_textures[" in (SHADER_ROOT / d.shader).read_text()
            for b, entry in sorted(d.bindings.items()):
                target, kind = entry[0], entry[1]
                spec = {"binding": b, "kind": kind}
                if kind in ("storage_buffer", "uniform_buffer"):
                    spec["buffer"] = target
                elif isinstance(target, (list, tuple)):
                    spec["images"] = list(target)
                    if has_texture_array:
                        spec["array_size"] = TEXTURE_ARRAY_SIZE
                else:
                    spec["image"] = target
                if len(entry) > 2:
                    spec["mip"] = entry[2]
                binds.append(spec)
            job["dispatches"].append({
                "spv": str(compile_shader(d.shader)),
                "push_constants": d.pc.hex(),
                "groups": list(d.groups),
                "indirect": None if d.indirect is None else {"buffer": d.indirect[0], "offset": d.indirect[1]},
                "bindings": binds,
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
            raw = (work / f"{name}.out").read_bytes()
            if name in buffers:
                result[name] = raw
                continue
            img = images[name]
            channels, dtype, unorm = FORMATS[img.format]
            flat = np.frombuffer(raw, dtype=dtype)
            mips, offset = [], 0
            for m in range(img.mips):
                shape = img.mip_shape(m)
                count = int(np.prod(shape))
                arr = flat[offset:offset + count].reshape(shape)
                offset += count
                if dtype != np.uint32:
                    arr = arr.astype(np.float32)
                    arr = arr / 255.0 if unorm else arr
                mips.append(arr if img.layers > 1 else arr[0])
            result[name] = mips if img.mips > 1 else mips[0]
        if self.keep_dir is None:
            shutil.rmtree(work, ignore_errors=True)
        return result
