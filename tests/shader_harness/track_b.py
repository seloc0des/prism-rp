"""Struct packing, Godot-convention matrices and tiny scenes for the Track B shader tests.

Layouts mirror addons/prism_rp/shaders/include/{visibility_common,vis_shading_common,vsp_common}.glsl.
Matrices are numpy row-major math matrices; mat() packs them column-major the way GLSL reads them.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

HERE_SHADERS = __import__("pathlib").Path(__file__).resolve().parent / "shaders"

EMPTY = 0xFFFFFFFF
FLAG_VISIBLE, FLAG_TWO_SIDED, FLAG_ALPHA_TEST, FLAG_CAST_SHADOW = 1, 2, 4, 8
INVALID_TEX = 0xFFFFFFFF


def mat(m: np.ndarray) -> bytes:
    return np.asarray(m, dtype=np.float32).T.tobytes()


def vec4(*v) -> bytes:
    return struct.pack("<4f", *v)


# --------------------------------------------------------------------------- matrices (Godot conventions)

def godot_projection(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Camera projection as RenderSceneData hands it out: GL perspective + Projection::set_depth_correction
    (flip Y for Vulkan, reverse Z, remap Z to [0, 1])."""
    f = 1.0 / np.tan(np.radians(fovy_deg) / 2)
    gl = np.array([[f / aspect, 0, 0, 0], [0, f, 0, 0],
                   [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)], [0, 0, -1, 0]])
    correction = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -0.5, 0.5], [0, 0, 0, 1]], dtype=float)
    return correction @ gl


def look_at_view(eye, target, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """World -> view (camera looks down -Z, +Y up), the inverse of a Godot camera transform."""
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    back = eye - target
    back /= np.linalg.norm(back)
    right = np.cross(up, back)
    right /= np.linalg.norm(right)
    true_up = np.cross(back, right)
    cam = np.eye(4)
    cam[:3, 0], cam[:3, 1], cam[:3, 2], cam[:3, 3] = right, true_up, back, eye
    return np.linalg.inv(cam)


def frustum_planes(view_proj: np.ndarray) -> list[np.ndarray]:
    """Inward planes (xyz normal, w distance) of a Vulkan reverse-Z clip volume: -w<=x,y<=w, 0<=z<=w."""
    r = view_proj
    planes = [r[3] + r[0], r[3] - r[0], r[3] + r[1], r[3] - r[1], r[3] - r[2], r[2]]
    return [p / np.linalg.norm(p[:3]) for p in planes]


def project(view_proj: np.ndarray, p) -> np.ndarray:
    clip = view_proj @ np.append(np.asarray(p, float), 1.0)
    return clip


def translate(x, y, z) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def scale(x, y, z) -> np.ndarray:
    return np.diag([x, y, z, 1.0])


# --------------------------------------------------------------------------- GPU structs

def gpu_meshlet(vertex_offset, triangle_offset, vertex_count, triangle_count, center, radius,
                cone_apex=(0, 0, 0), cone_axis=(0, 0, 1), cone_cutoff=1.0) -> bytes:
    return struct.pack("<4I4f4f4f", vertex_offset, triangle_offset, vertex_count, triangle_count,
                       *center, radius, *cone_apex, 0.0, *cone_axis, cone_cutoff)


def gpu_instance(world, meshlet_offset, meshlet_count, vertex_base, material_id, flags, prev_world=None) -> bytes:
    prev_world = world if prev_world is None else prev_world
    return mat(world) + mat(prev_world) + struct.pack("<8I", meshlet_offset, meshlet_count, vertex_base, material_id, flags, 0, 0, 0)


def gpu_material(base_color=(1, 1, 1, 1), emissive=(0, 0, 0), roughness=1.0, metallic=0.0, ao_strength=1.0,
                 alpha_cutoff=0.5, normal_scale=1.0, tex=(INVALID_TEX,) * 4, flags=0) -> bytes:
    return (vec4(*base_color) + vec4(*emissive, roughness) + vec4(metallic, ao_strength, alpha_cutoff, normal_scale)
            + struct.pack("<4I4I", *tex, flags, 0, 0, 0))


def gpu_vertex_attributes(normal, tangent=(1, 0, 0, 1), uv=(0, 0)) -> bytes:
    return vec4(*normal, 0.0) + vec4(*tangent) + vec4(*uv, 0.0, 0.0)


def pack_triangles(tris) -> bytes:
    return b"".join(struct.pack("<I", a | (b << 8) | (c << 16)) for a, b, c in tris)


def u32s(*v) -> bytes:
    return struct.pack(f"<{len(v)}I", *v)


def payload(instance_id, meshlet_id, prim_id, alpha=False) -> tuple[int, int]:
    return instance_id, (meshlet_id << 8) | (prim_id << 1) | int(alpha)


def cull_ubo(view_proj, prev_view_proj, camera_pos, near, planes, hiz_mips, candidates, max_visible,
             max_false_negatives, phase, frustum=1, cone=1, hiz=1) -> bytes:
    data = mat(view_proj) + mat(prev_view_proj) + vec4(*camera_pos, near) + b"".join(vec4(*p) for p in planes)
    data += u32s(hiz_mips, candidates, max_visible, max_false_negatives, phase, frustum, cone, hiz)
    assert len(data) == 272
    return data


def shade_ubo(view_proj, prev_view_proj, camera_pos, sun_dir, sun_intensity, sun_color, ambient, width, height,
              max_work_items) -> bytes:
    data = mat(view_proj) + mat(prev_view_proj) + mat(np.linalg.inv(view_proj))
    data += vec4(*camera_pos, 1.0) + vec4(*sun_dir, sun_intensity) + vec4(*sun_color, ambient)
    data += u32s(width, height, (width + 7) // 8, max_work_items)
    assert len(data) == 256
    return data


# --------------------------------------------------------------------------- virtual shadow pages

VSP_LEVELS, VSP_PAGES, VSP_PAGE_SIZE, VSP_VIRTUAL_RES = 8, 128, 128, 16384
VSP_ATLAS_PAGES, VSP_ATLAS_RES, VSP_MAX_PAGES = 32, 4096, 1024
PTE_VALID, PTE_DIRTY, PTE_ALLOCATED, PTE_REQUESTED = 1 << 31, 1 << 30, 1 << 29, 1 << 28
PTE_AGE_SHIFT, PTE_PAGE_MASK = 20, 0xFFF
SUN_TOP, SUN_RANGE = 50.0, 100.0   # light depth z = (SUN_TOP - y) / SUN_RANGE for the straight-down sun


def sun_level_matrix(extent: float, center=(0.0, 0.0)) -> np.ndarray:
    """Orthographic light for a sun straight overhead: ndc.x = 2(x - cx)/extent, ndc.y = 2(z - cz)/extent,
    z = (SUN_TOP - y) / SUN_RANGE (smaller = nearer the sun)."""
    cx, cz = center
    return np.array([[2 / extent, 0, 0, -2 * cx / extent],
                     [0, 0, 2 / extent, -2 * cz / extent],
                     [0, -1 / SUN_RANGE, 0, SUN_TOP / SUN_RANGE],
                     [0, 0, 0, 1]])


@dataclass
class VspLevel:
    extent: float
    center: tuple[float, float] = (0.0, 0.0)
    offset: tuple[int, int] = (0, 0)
    scroll: tuple[int, int] = (0, 0)

    def pack(self) -> bytes:
        return (mat(sun_level_matrix(self.extent, self.center)) + vec4(self.center[0], 0.0, self.center[1], self.extent)
                + struct.pack("<4i", *self.offset, *self.scroll))

    def uv(self, p) -> np.ndarray:
        ndc = sun_level_matrix(self.extent, self.center) @ np.append(np.asarray(p, float), 1.0)
        return ndc[:2] * 0.5 + 0.5

    def linear_page(self, p) -> tuple[int, int]:
        x, y = np.clip(np.floor(self.uv(p) * VSP_PAGES), 0, VSP_PAGES - 1).astype(int)
        return int(x), int(y)

    def toroidal(self, linear) -> tuple[int, int]:
        return ((linear[0] + self.offset[0]) & (VSP_PAGES - 1), (linear[1] + self.offset[1]) & (VSP_PAGES - 1))


@dataclass
class VspSetup:
    view: np.ndarray
    proj: np.ndarray
    camera_pos: tuple
    width: int
    height: int
    levels: list[VspLevel] = field(default_factory=lambda: [VspLevel(16.0 * 2 ** k) for k in range(VSP_LEVELS)])
    near: float = 0.05
    sun_dir: tuple = (0.0, 1.0, 0.0)
    depth_bias_m: float = 0.02
    tan_sun: float = 0.01
    min_filter_texels: float = 1.0
    max_filter_texels: float = 2.0
    lod_bias: float = 0.0
    max_dirty_pages: int = 4096
    dirty_aabbs: int = 0
    pass_mode: int = 0
    force_invalidate: int = 0
    candidates: int = 0
    max_raster_items: int = 4096
    evict_after: int = 8
    frame_index: int = 0

    def ubo(self, **overrides) -> bytes:
        s = VspSetup(**{**self.__dict__, **overrides})
        view_proj = s.proj @ s.view
        data = mat(np.linalg.inv(view_proj)) + mat(s.view) + vec4(*s.camera_pos, s.near)
        data += vec4(*s.sun_dir, s.depth_bias_m) + vec4(s.tan_sun, s.min_filter_texels, s.max_filter_texels, s.lod_bias)
        data += u32s(s.width, s.height, s.dirty_aabbs, s.max_dirty_pages)
        data += u32s(s.pass_mode, s.force_invalidate, s.candidates, s.max_raster_items, s.evict_after, s.frame_index, 0, 0)
        data += b"".join(lvl.pack() for lvl in s.levels)
        assert len(data) == 224 + 96 * VSP_LEVELS
        return data

    def ground_depth(self, ground_y: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Reverse-Z depth of the plane y = ground_y seen by the camera, and the world position per pixel."""
        view_proj = self.proj @ self.view
        inv = np.linalg.inv(view_proj)
        depth = np.zeros((self.height, self.width))
        world = np.zeros((self.height, self.width, 3))
        for y in range(self.height):
            for x in range(self.width):
                ndc = np.array([(x + 0.5) / self.width * 2 - 1, (y + 0.5) / self.height * 2 - 1])
                a = inv @ np.array([*ndc, 1.0, 1.0])
                b = inv @ np.array([*ndc, 0.0, 1.0])
                a, b = a[:3] / a[3], b[:3] / b[3]
                t = (ground_y - a[1]) / (b[1] - a[1])
                if not 0 <= t <= 1:
                    continue
                p = a + t * (b - a)
                clip = view_proj @ np.append(p, 1.0)
                depth[y, x] = clip[2] / clip[3]
                world[y, x] = p
        return depth, world


def vsp_state(free_count=VSP_MAX_PAGES, dirty=0, raster=0, clear_dispatch=(0, 1, 1), raster_dispatch=(0, 1, 1),
              rects=None) -> bytes:
    rects = rects or [(EMPTY, EMPTY, 0, 0)] * VSP_LEVELS
    data = u32s(free_count, dirty, raster, 0) + u32s(*clear_dispatch, 0) + u32s(*raster_dispatch, 0)
    data += b"".join(u32s(*r) for r in rects)
    assert len(data) == 176
    return data


def unpack_vsp_state(raw: bytes) -> dict:
    v = struct.unpack("<44I", raw[:176])
    return {"free": v[0], "dirty": v[1], "raster": v[2], "clear_dispatch": v[4:7], "raster_dispatch": v[8:11],
            "rects": [v[12 + 4 * i: 16 + 4 * i] for i in range(VSP_LEVELS)]}
