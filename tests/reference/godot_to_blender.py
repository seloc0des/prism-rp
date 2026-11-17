"""Godot -> Blender conversions for the Gate 2 reference (pure Python, no bpy; unit-tested).

Coordinates: Godot is Y-up, right-handed, cameras look down local -Z. Blender is Z-up,
right-handed, cameras also look down local -Z with +Y up. Mapping world vectors with
(x, y, z) -> (x, -z, y) preserves handedness, and because both cameras use the same local axes,
converting each basis column the same way gives the Blender camera matrix directly.

Light units are approximate, which is why Gate 2 also reports a scale-fitted PSNR:
  - Godot omni (non-physical units): irradiance at distance d ~ energy / d^2 (ignoring Godot's
    range window and attenuation curve). Blender point light of power P gives P / (4 pi^2 d^2).
    So P = 4 pi^2 * energy.
  - Godot directional energy is irradiance; Blender sun strength is irradiance (W/m^2): equal.
  - Emission: both are radiance (color * strength), equal.
"""

from __future__ import annotations

import math


def vec(v):
    """Godot world/direction vector -> Blender."""
    x, y, z = v
    return (x, -z, y)


def flat_positions(flat):
    """Flat [x0, y0, z0, x1, ...] Godot positions -> list of Blender (x, y, z) tuples."""
    return [vec(flat[i:i + 3]) for i in range(0, len(flat), 3)]


def uv(u, v):
    """Godot UVs start at the top-left (v down); Blender's at the bottom-left (v up)."""
    return (u, 1.0 - v)


def triangles_ccw(indices):
    """Godot clockwise front faces -> counter-clockwise (Blender: face normal points out)."""
    return [(indices[i], indices[i + 2], indices[i + 1]) for i in range(0, len(indices), 3)]


def camera_matrix(basis_columns, origin):
    """4x4 row-major Blender world matrix for a Godot camera (basis given as its 3 columns)."""
    cx, cy, cz = (vec(c) for c in basis_columns)
    o = vec(origin)
    return [
        [cx[0], cy[0], cz[0], o[0]],
        [cx[1], cy[1], cz[1], o[1]],
        [cx[2], cy[2], cz[2], o[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def camera_fov(fov_deg, keep_aspect, width, height):
    """(blender sensor_fit, angle in radians). Godot's fov is vertical for KEEP_HEIGHT."""
    return ("VERTICAL" if keep_aspect == "height" else "HORIZONTAL"), math.radians(fov_deg)


def omni_power_watts(energy):
    return 4.0 * math.pi * math.pi * energy


def sun_strength(energy):
    return energy


def look_rotation_matrix(direction):
    """3x3 row-major rotation whose local -Z points along `direction` (Blender coords)."""
    d = direction
    n = math.sqrt(sum(c * c for c in d)) or 1.0
    f = [c / n for c in d]                  # forward = -Z
    z = [-c for c in f]
    up = [0.0, 0.0, 1.0] if abs(z[2]) < 0.999 else [0.0, 1.0, 0.0]
    x = [up[1] * z[2] - up[2] * z[1], up[2] * z[0] - up[0] * z[2], up[0] * z[1] - up[1] * z[0]]
    xn = math.sqrt(sum(c * c for c in x))
    x = [c / xn for c in x]
    y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]]
    return [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]]
