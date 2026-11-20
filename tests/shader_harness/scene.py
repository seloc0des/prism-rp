"""Analytic G-buffers for Lumina tests: Godot-style camera + a few infinite planes, ray traced in numpy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Camera:
    width: int = 64
    height: int = 48
    fov_y_deg: float = 60.0
    near: float = 0.1
    far: float = 100.0

    @property
    def proj(self) -> tuple[float, float, float, float]:
        """(P00, P11, P20, P21) of the depth-corrected (flip-Y) projection, as Lumina expects."""
        p11 = 1.0 / np.tan(np.radians(self.fov_y_deg) / 2.0)
        p00 = p11 / (self.width / self.height)
        return (p00, -p11, 0.0, 0.0)

    @property
    def depth_params(self) -> tuple[float, float]:
        n, f = self.near, self.far
        return (f * n / (f - n), n / (f - n))

    def depth_from_linear(self, z: np.ndarray) -> np.ndarray:
        a, b = self.depth_params
        return a / z - b

    def ray_dirs(self) -> np.ndarray:
        """View-space direction per pixel, scaled so dir.z == -1 (so t == linear depth)."""
        p00, p11, _, _ = self.proj
        y, x = np.mgrid[0:self.height, 0:self.width].astype(np.float64)
        u = (x + 0.5) / self.width
        v = (y + 0.5) / self.height
        ndc_x, ndc_y = u * 2 - 1, v * 2 - 1
        return np.stack([ndc_x / p00, ndc_y / p11, -np.ones_like(u)], axis=-1)


@dataclass
class Plane:
    point: tuple[float, float, float]
    normal: tuple[float, float, float]
    color: tuple[float, float, float]
    bounds: tuple | None = None   # ((xmin, xmax), (ymin, ymax), (zmin, zmax)) in view space; None = infinite


def render(cam: Camera, planes: list[Plane]) -> dict[str, np.ndarray]:
    """Returns depth (reverse-Z, 0 = sky), normal_roughness (rgba8 values 0..1) and color (rgba)."""
    dirs = cam.ray_dirs()
    h, w = cam.height, cam.width
    best_t = np.full((h, w), np.inf)
    normal = np.zeros((h, w, 3))
    color = np.zeros((h, w, 3))
    for pl in planes:
        p = np.array(pl.point); n = np.array(pl.normal) / np.linalg.norm(pl.normal)
        denom = dirs @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (p @ n) / denom
        hit = (denom < -1e-6) & (t > cam.near) & (t < cam.far) & (t < best_t)
        if pl.bounds is not None:
            pos = dirs * np.where(np.isfinite(t), t, 0.0)[..., None]
            for axis, (lo, hi) in enumerate(pl.bounds):
                hit &= (pos[..., axis] >= lo) & (pos[..., axis] <= hi)
        best_t[hit] = t[hit]
        normal[hit] = n
        color[hit] = pl.color
    sky = ~np.isfinite(best_t)
    depth = np.where(sky, 0.0, cam.depth_from_linear(np.where(sky, 1.0, best_t)))
    nr = np.concatenate([normal * 0.5 + 0.5, np.full((h, w, 1), 0.5)], axis=-1)
    nr[sky] = (0.5, 0.5, 1.0, 0.5)
    rgba = np.concatenate([color, np.ones((h, w, 1))], axis=-1)
    return {"depth": depth, "normal": nr, "color": rgba, "linear_z": np.where(sky, 0.0, best_t)}


def cornell(cam: Camera, red=(4.0, 0.0, 0.0), floor=(0.05, 0.05, 0.05)) -> dict[str, np.ndarray]:
    """Floor at y = -1, back wall at z = -6, bright red wall at x = +1.5 facing -X."""
    return render(cam, [
        Plane((0, -1, 0), (0, 1, 0), floor),
        Plane((0, 0, -6), (0, 0, 1), floor),
        Plane((1.5, 0, 0), (-1, 0, 0), red),
    ])


def irradiance_truth(cam: Camera, planes: list[Plane], n_dirs: int = 256) -> np.ndarray:
    """One-bounce E / pi (what Lumina's irradiance estimates) per pixel: the cosine-weighted mean of the
    radiance (plane colour) seen from each visible point, by casting n_dirs fixed cosine-distributed rays.
    Room means converge by 256 rays for large lights; a light a few pixels wide reads about 5% high there
    (most pixels' rays miss it, a few catch it). 0 on sky pixels. Planes are one-sided, as in render()."""
    g = render(cam, planes)
    z = g["linear_z"]
    surf = z > 0
    pos = (cam.ray_dirs() * z[..., None])[surf]
    nrm = g["normal"][..., :3][surf] * 2.0 - 1.0
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    k = np.arange(n_dirs) + 0.5
    r = np.sqrt(k / n_dirs)
    phi = k * np.pi * (3.0 - np.sqrt(5.0))
    local = np.stack([r * np.cos(phi), r * np.sin(phi), np.sqrt(1.0 - r * r)], axis=-1)     # (D, 3), cosine-weighted
    helper = np.where(np.abs(nrm[:, 1:2]) < 0.9, [[0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0]])
    t = np.cross(nrm, helper)
    t /= np.linalg.norm(t, axis=-1, keepdims=True)
    b = np.cross(nrm, t)
    d = local[None, :, 0:1] * t[:, None] + local[None, :, 1:2] * b[:, None] + local[None, :, 2:3] * nrm[:, None]  # (P, D, 3)
    o = (pos + nrm * 1e-4)[:, None, :]
    best = np.full(d.shape[:2], np.inf)
    col = np.zeros(d.shape)
    for pl in planes:
        pn = np.array(pl.normal, float)
        pn /= np.linalg.norm(pn)
        den = d @ pn
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = ((np.array(pl.point, float) - o) @ pn) / den
        hit = (den < -1e-6) & (tt > 1e-4) & (tt < best)
        if pl.bounds is not None:
            hp = o + d * np.where(np.isfinite(tt), tt, 0.0)[..., None]
            for axis, (lo, hi) in enumerate(pl.bounds):
                hit &= (hp[..., axis] >= lo) & (hp[..., axis] <= hi)
        best[hit] = tt[hit]
        col[hit] = pl.color
    out = np.zeros(z.shape + (3,))
    out[surf] = col.mean(axis=1)
    return out
