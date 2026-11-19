"""A Cornell-box scene in the prism_scene_export.gd JSON format, built in Python, so the Cycles
pipeline can be tested without Godot. Coordinates and winding follow Godot conventions."""

import json
from pathlib import Path


def _quad(name, corners, color, emission=(0, 0, 0)):
    # corners in Godot coords, ordered so (0, 1, 2) / (0, 2, 3) are clockwise seen from the front.
    pos = [c for p in corners for c in p]
    return {"name": name, "positions": pos, "normals": [], "uvs": [], "indices": [0, 1, 2, 0, 2, 3],
            "material": {"albedo": [*color, 1.0], "roughness": 1.0, "metallic": 0.0, "emission": list(emission),
                         "emission_energy": 1.0, "albedo_texture": None, "transparency": "opaque", "alpha_scissor": 0.5, "unshaded": False}}


def cornell(red=(0.7, 0.05, 0.05)):
    s = 1.5
    white = (0.75, 0.75, 0.75)
    meshes = [
        # floor y=0, front face +Y: clockwise seen from above
        _quad("floor", [(-s, 0, -s), (s, 0, -s), (s, 0, s), (-s, 0, s)], white),
        # ceiling y=3, front face -Y
        _quad("ceiling", [(-s, 3, -s), (-s, 3, s), (s, 3, s), (s, 3, -s)], white),
        # back wall z=-s, front face +Z
        _quad("back", [(-s, 0, -s), (-s, 3, -s), (s, 3, -s), (s, 0, -s)], white),
        # left wall x=-s, red, front face +X
        _quad("left", [(-s, 0, s), (-s, 3, s), (-s, 3, -s), (-s, 0, -s)], red),
        # right wall x=+s, white, front face -X
        _quad("right", [(s, 0, -s), (s, 3, -s), (s, 3, s), (s, 0, s)], white),
    ]
    return {
        "version": 2,
        "resolution": [96, 72],
        "camera": {"basis_columns": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "origin": [0, 1.5, 4.2],
                   "fov_deg": 50.0, "keep_aspect": "height", "near": 0.05, "far": 100.0},
        "environment": {"background": [0, 0, 0], "ambient": [0, 0, 0], "ambient_energy": 0.0},
        "meshes": meshes,
        "lights": [{"type": "omni", "position": [0, 2.6, 0], "color": [1, 1, 1], "energy": 3.0, "range": 8.0,
                    "attenuation": 1.0, "shadow": True}],
    }


def write(path: Path, scene=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scene or cornell()))
    return path
