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
    albedo: tuple[float, float, float] = (0.0, 0.0, 0.0)   # what irradiance_truth's later bounces reflect


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


def irradiance_truth(cam: Camera, planes: list[Plane], n_dirs: int = 256, pixels: np.ndarray | None = None,
                     bounces: int = 1) -> np.ndarray:
    """One-bounce E / pi (what Lumina's irradiance estimates) per pixel: the cosine-weighted mean of the
    radiance (plane colour) seen from each visible point, by casting n_dirs fixed cosine-distributed rays.
    Room means converge by 256 rays for large lights; a light a few pixels wide reads about 5% high there
    (most pixels' rays miss it, a few catch it). 0 on sky pixels. Planes are one-sided, as in render().
    `pixels` (a boolean mask) limits the work to those pixels; the rest come back 0.

    bounces > 1 adds light that bounced more often: a ray's hit sends out its plane's colour plus its albedo
    times the E / pi at the hit, estimated by one random cosine-distributed ray from there (path tracing:
    noisy per pixel, exact on average over a region). bounces = n counts n - 1 such reflections."""
    g = render(cam, planes)
    z = g["linear_z"]
    surf = z > 0
    if pixels is not None:
        surf = surf & pixels
    pos = (cam.ray_dirs() * z[..., None])[surf]
    nrm = g["normal"][..., :3][surf] * 2.0 - 1.0
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    local = _cosine_dirs(n_dirs)   # (D, 3), cosine-weighted, the same for every pixel
    rng = np.random.default_rng(7)
    radiance = np.zeros((len(pos), 3))
    chunk = max(1, (1 << 22) // n_dirs)   # pixels per pass: keeps the (P, D, 3) arrays near 100 MB
    for i in range(0, len(pos), chunk):
        radiance[i:i + chunk] = _mean_radiance(pos[i:i + chunk], nrm[i:i + chunk], local, planes, bounces - 1, rng)
    out = np.zeros(z.shape + (3,))
    out[surf] = radiance
    return out


def _cosine_dirs(n):
    k = np.arange(n) + 0.5
    r = np.sqrt(k / n)
    phi = k * np.pi * (3.0 - np.sqrt(5.0))
    return np.stack([r * np.cos(phi), r * np.sin(phi), np.sqrt(1.0 - r * r)], axis=-1)


def _mean_radiance(pos, nrm, local, planes, deeper=0, rng=None):
    """irradiance_truth for points `pos` with normals `nrm` (P, 3): mean radiance over the directions `local`
    ((D, 3), the same for every point, or (P, D, 3)), each hit adding its albedo times the light one random ray
    from there finds, `deeper` times over."""
    helper = np.where(np.abs(nrm[:, 1:2]) < 0.9, [[0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0]])
    t = np.cross(nrm, helper)
    t /= np.linalg.norm(t, axis=-1, keepdims=True)
    b = np.cross(nrm, t)
    if local.ndim == 2:
        local = local[None]
    d = local[..., 0:1] * t[:, None] + local[..., 1:2] * b[:, None] + local[..., 2:3] * nrm[:, None]  # (P, D, 3)
    o = (pos + nrm * 1e-4)[:, None, :]
    best = np.full(d.shape[:2], np.inf)
    col = np.zeros(d.shape)
    which = np.full(d.shape[:2], -1)
    for i, pl in enumerate(planes):
        pn = np.array(pl.normal, float)
        pn /= np.linalg.norm(pn)
        den = d @ pn
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = ((np.array(pl.point, float) - o) @ pn) / den
        hit = (den < -1e-6) & (tt > 1e-4) & (tt < best)
        if pl.bounds is not None:
            idx = np.nonzero(hit)   # check the bounds only where the ray can hit the plane at all
            hp = o[idx[0], 0] + d[idx] * tt[idx][:, None]
            inside = np.ones(len(idx[0]), bool)
            for axis, (lo, hi) in enumerate(pl.bounds):
                inside &= (hp[:, axis] >= lo) & (hp[:, axis] <= hi)
            hit[idx[0][~inside], idx[1][~inside]] = False
        best[hit] = tt[hit]
        col[hit] = pl.color
        which[hit] = i
    if deeper:
        albedo = np.array([pl.albedo for pl in planes] + [(0.0, 0.0, 0.0)], float)[which]   # index -1: missed
        idx = np.nonzero(albedo.max(axis=-1) > 0.0)
        if len(idx[0]):
            hit_pos = o[idx[0], 0] + d[idx] * best[idx][:, None]
            hit_n = np.array([np.array(pl.normal, float) / np.linalg.norm(pl.normal) for pl in planes])[which[idx]]
            u = rng.random((len(hit_pos), 1, 2))
            r, phi = np.sqrt(u[..., 0]), 2.0 * np.pi * u[..., 1]
            one_ray = np.stack([r * np.cos(phi), r * np.sin(phi), np.sqrt(1.0 - r * r)], axis=-1)   # (H, 1, 3)
            col[idx] += albedo[idx] * _mean_radiance(hit_pos, hit_n, one_ray, planes, deeper - 1, rng)
    return col.mean(axis=1)
