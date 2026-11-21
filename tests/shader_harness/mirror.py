"""Lumina's scene mirror (LuminaSceneMirror) for harness tests: meshes, materials and lights packed into the buffers
lumina_scene_common.glsl declares, and the TLAS the runner builds from the same triangles. One instance and one
surface per mesh, one vertex per triangle corner."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

from harness import AsInstance, Buf, Img

MASK_CAMERA = 0x01
MASK_SHADOW = 0x02
# The instance flags LuminaSceneMirror's instance_flags_for sets (VkGeometryInstanceFlagBitsKHR values).
CULL_DISABLE = 0x1
FLIP_FACING = 0x2
FORCE_OPAQUE = 0x4

MATERIAL_EMISSION_MULTIPLY = 16
MATERIAL_DOUBLE_SIDED = 32

LIGHT_DIRECTIONAL, LIGHT_OMNI, LIGHT_SPOT = 0, 1, 2


@dataclass
class Material:
    albedo: tuple = (1.0, 1.0, 1.0)     # linear
    emission: tuple = (0.0, 0.0, 0.0)   # linear
    emission_energy: float = 0.0
    emission_texture: int = 0           # slot in Scene.textures + 1; 0: none
    emission_multiply: bool = False     # Godot's Multiply emission operator
    double_sided: bool = False          # Godot's cull mode Disabled

    def pack(self) -> bytes:
        flags = (MATERIAL_DOUBLE_SIDED if self.double_sided else 0) | (MATERIAL_EMISSION_MULTIPLY if self.emission_multiply else 0)
        # LuminaMaterial: albedo, uv transform, emission, albedo texture, flags, alpha scissor, emission texture.
        return struct.pack("<4f4f4fIIfI", *self.albedo, 1.0, 1.0, 1.0, 0.0, 0.0, *self.emission, self.emission_energy,
                           0, flags, 0.5, self.emission_texture)


@dataclass
class Mesh:
    corners: np.ndarray                 # (3 * triangles, 3) world space, Godot's winding (clockwise seen from the front)
    material: Material = field(default_factory=Material)
    casts_shadows: bool = True


@dataclass
class Light:
    kind: int
    color: tuple = (1.0, 1.0, 1.0)      # linear, times energy and indirect energy
    position: tuple = (0.0, 0.0, 0.0)
    direction: tuple = (0.0, -1.0, 0.0)  # the way it shines
    range: float = 5.0
    attenuation: float = 1.0
    spot_angle_deg: float = 45.0
    spot_attenuation: float = 1.0
    shadows: bool = True

    def pack(self) -> bytes:
        d = np.asarray(self.direction, float)
        d = d / np.linalg.norm(d)
        return struct.pack("<4f4f4f4f4I", *self.position, 1.0 / self.range, *d, self.attenuation, *self.color, 0.0,
                           np.cos(np.radians(self.spot_angle_deg)), 1.0 / self.spot_attenuation, 0.0, 0.0,
                           self.kind, int(self.shadows), 0, 0)


def quad(center, normal, size, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """A square of side `size` whose front (for Godot) faces `normal`, as two triangles."""
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    u = np.asarray(up, float)
    if abs(np.dot(u, n)) > 0.9:
        u = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    r = np.cross(u, n)
    r /= np.linalg.norm(r)
    u = np.cross(n, r)
    c = np.asarray(center, float)
    h = size / 2.0
    q = [c - h * r - h * u, c + h * r - h * u, c + h * r + h * u, c - h * r + h * u]   # counter-clockwise from the front
    return np.array([q[0], q[2], q[1], q[0], q[3], q[2]])


def sphere(center, radius, segments=64, rings=32, inward=False) -> np.ndarray:
    """A closed UV sphere whose fronts face out, or in toward the centre."""
    c = np.asarray(center, float)

    def point(ring, segment):
        theta, phi = np.pi * ring / rings, 2.0 * np.pi * segment / segments
        return c + radius * np.array([np.sin(theta) * np.cos(phi), np.cos(theta), np.sin(theta) * np.sin(phi)])

    triangles = []
    for r in range(rings):
        for g in range(segments):
            a, b, d, e = point(r, g), point(r, g + 1), point(r + 1, g), point(r + 1, g + 1)
            triangles += [(a, b, e), (a, e, d)]
    corners = []
    for p0, p1, p2 in triangles:
        n = np.cross(p2 - p0, p1 - p0)   # the front's side, as lumina_scene_point computes it
        if np.linalg.norm(n) < 1e-12:
            continue   # the poles' degenerate triangles
        outward = np.dot(n, (p0 + p1 + p2) / 3.0 - c) > 0.0
        corners += [p0, p1, p2] if outward != inward else [p0, p2, p1]
    return np.array(corners)


def cube(center, half) -> np.ndarray:
    """A closed cube, faces outward."""
    c = np.asarray(center, float)
    faces = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            n = np.zeros(3)
            n[axis] = sign
            faces.append(quad(c + n * half, n, 2.0 * half))
    return np.concatenate(faces)


@dataclass
class Scene:
    meshes: list[Mesh]
    lights: list[Light] = field(default_factory=list)
    textures: list[tuple] = field(default_factory=list)   # 1 x 1 rgba each, in texture slots 1, 2, ...

    def buffers(self) -> dict[str, Buf]:
        positions, attributes, surfaces = [], [], []
        corner_base = 0
        for mesh in self.meshes:
            corners = np.asarray(mesh.corners, np.float32)
            positions.append(corners)
            count = len(corners)
            attributes.append(np.tile(np.array([0, 0, 0xFFFFFFFF, 0], np.uint32), (count, 1)))   # uv 0, white, no normal
            surfaces.append((corner_base, corner_base, count // 3, 0))
            corner_base += count
        indices = np.concatenate([np.arange(len(p), dtype=np.uint32) for p in positions])
        count = len(self.meshes)
        materials = Material().pack() + b"".join(m.material.pack() for m in self.meshes)   # 0: Godot's default
        lights = struct.pack("<4I", len(self.lights), 0, 0, 0) + b"".join(light.pack() for light in self.lights)
        return {
            "scene_positions": Buf(data=np.concatenate(positions).tobytes()),
            "scene_indices": Buf(data=indices.tobytes()),
            "scene_attributes": Buf(data=np.concatenate(attributes).tobytes()),
            "scene_surfaces": Buf(data=struct.pack(f"<{4 * count}I", *(v for s in surfaces for v in s))),
            "scene_instances": Buf(data=struct.pack(f"<{4 * count}I", *(v for i in range(count) for v in (i, i, 0, 0)))),
            "scene_instance_materials": Buf(data=struct.pack(f"<{count}I", *range(1, count + 1))),
            "scene_materials": Buf(data=materials),
            "scene_lights": Buf(size=16 + 80, data=lights),
        }

    def acceleration_structures(self) -> dict[str, list[AsInstance]]:
        instances = []
        for i, mesh in enumerate(self.meshes):
            flags = FORCE_OPAQUE | FLIP_FACING | (CULL_DISABLE if mesh.material.double_sided else 0)
            mask = MASK_CAMERA | (MASK_SHADOW if mesh.casts_shadows else 0)
            instances.append(AsInstance([np.asarray(mesh.corners, np.float32)], custom_index=i, mask=mask, flags=flags))
        return {"scene": instances}

    def images(self) -> dict[str, Img]:
        images = {"scene_white": Img("rgba8", data=np.ones((1, 1, 4)))}
        for i, rgba in enumerate(self.textures):
            images[f"scene_texture_{i + 1}"] = Img("rgba8", data=np.array(rgba, float).reshape(1, 1, 4))
        return images

    def bindings(self, set_index: int = 1) -> dict:
        return {
            (set_index, 0): ("scene", "acceleration_structure"),
            (set_index, 1): ("scene_positions", "storage_buffer"),
            (set_index, 2): ("scene_indices", "storage_buffer"),
            (set_index, 3): ("scene_attributes", "storage_buffer"),
            (set_index, 4): ("scene_surfaces", "storage_buffer"),
            (set_index, 5): ("scene_instances", "storage_buffer"),
            (set_index, 6): ("scene_instance_materials", "storage_buffer"),
            (set_index, 7): ("scene_materials", "storage_buffer"),
            (set_index, 8): (None, "sampler"),
            (set_index, 9): (["scene_white", *(f"scene_texture_{i + 1}" for i in range(len(self.textures)))], "texture"),
            (set_index, 10): ("scene_lights", "storage_buffer"),
        }
