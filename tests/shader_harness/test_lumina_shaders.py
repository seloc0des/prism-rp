"""Lumina-Cascades compute shaders on lavapipe with analytic scenes."""

import numpy as np
import pytest

import pc
from harness import Dispatch, Img, groups_for
from scene import Camera, Plane, cornell, render

BASE_BLOCK = 4


def cascade_layout(cam, spacing, count):
    levels = []
    for i in range(count):
        s = spacing << i
        block = BASE_BLOCK << i
        probes = ((cam.width + s - 1) // s, (cam.height + s - 1) // s)
        levels.append({"spacing": s, "block": block, "probes": probes, "size": (probes[0] * block, probes[1] * block)})
    return levels


def run_cascades(gpu, cam, g, spacing=8, count=3, steps=16, thickness=0.2):
    """trace every level -> merge top-down -> integrate. Returns irradiance (h, w, 4)."""
    levels = cascade_layout(cam, spacing, count)
    images = {
        "depth": Img("r32f", data=g["depth"]), "normal": Img("rgba8", data=g["normal"]),
        "color": Img("rgba16f", data=g["color"]), "irr": Img("rgba16f", cam.width, cam.height),
    }
    for i, lv in enumerate(levels):
        images[f"c{i}"] = Img("rgba16f", *lv["size"])
    dispatches = []
    for i, lv in enumerate(levels):
        scale = 4.0 ** i
        dispatches.append(Dispatch(
            "lumina/lumina_rc_trace.comp.glsl",
            {0: ("depth", "sampled_nearest"), 1: ("normal", "sampled_nearest"), 2: ("color", "sampled_linear"), 3: (f"c{i}", "storage")},
            pc.trace(cam.width, cam.height, lv["probes"], lv["block"], float(lv["spacing"]),
                     spacing * (scale - 1) / 3.0, spacing * scale, steps, thickness, 0, 0.01, cam.proj, cam.depth_params),
            groups_for(*lv["size"])))
    for i in range(count - 2, -1, -1):
        lv, parent = levels[i], levels[i + 1]
        dispatches.append(Dispatch(
            "lumina/lumina_rc_merge.comp.glsl", {0: (f"c{i + 1}", "sampled_nearest"), 1: (f"c{i}", "storage")},
            pc.merge(lv["probes"], parent["probes"], lv["block"], parent["block"], float(lv["spacing"]), float(parent["spacing"])),
            groups_for(*lv["size"])))
    lv0 = levels[0]
    dispatches.append(Dispatch(
        "lumina/lumina_rc_integrate.comp.glsl",
        {0: ("c0", "sampled_nearest"), 1: ("depth", "sampled_nearest"), 2: ("normal", "sampled_nearest"), 3: ("irr", "storage")},
        pc.integrate(cam.width, cam.height, lv0["probes"], lv0["block"], float(lv0["spacing"]), cam.depth_params),
        groups_for(cam.width, cam.height)))
    return gpu.run(images, dispatches, ["irr"])["irr"]


def test_integrate_uniform_radiance_returns_it(gpu):
    """Uniform incoming radiance L over the sphere: E / pi over the hemisphere must be L.

    Exercises the octahedral direction decode and the normalised cosine weighting (exact for
    uniform light; the old 4pi/N solid-angle assumption came out 18% dark here).
    """
    cam = Camera(width=32, height=32)
    g = render(cam, [Plane((0, 0, -5), (0, 0, 1), (0, 0, 0))])   # wall facing the camera
    lv = cascade_layout(cam, 8, 1)[0]
    L = np.array([0.3, 0.6, 0.9])
    c0 = np.zeros((lv["size"][1], lv["size"][0], 4)); c0[..., :3] = L
    out = gpu.run(
        {"c0": Img("rgba16f", data=c0), "depth": Img("r32f", data=g["depth"]), "normal": Img("rgba8", data=g["normal"]),
         "irr": Img("rgba16f", 32, 32)},
        [Dispatch("lumina/lumina_rc_integrate.comp.glsl",
                  {0: ("c0", "sampled_nearest"), 1: ("depth", "sampled_nearest"), 2: ("normal", "sampled_nearest"), 3: ("irr", "storage")},
                  pc.integrate(32, 32, lv["probes"], lv["block"], 8.0, cam.depth_params), groups_for(32, 32))], ["irr"])["irr"]
    np.testing.assert_allclose(out[8:24, 8:24, :3].mean(axis=(0, 1)), L, rtol=0.01)


def test_merge_combines_interval_transmittance_and_parent(gpu):
    probes, parent_probes = (4, 4), (2, 2)
    block, parent_block = 4, 8
    size = (probes[0] * block, probes[1] * block)
    psize = (parent_probes[0] * parent_block, parent_probes[1] * parent_block)
    parent = np.zeros((psize[1], psize[0], 4)); parent[..., :3] = (0.2, 0.4, 0.6); parent[..., 3] = 0.5
    level = np.zeros((size[1], size[0], 4))
    level[:, : size[0] // 2] = (0.0, 0.0, 0.0, 1.0)      # clear interval: sees the parent
    level[:, size[0] // 2:] = (1.0, 0.5, 0.25, 0.0)      # blocked interval: parent hidden
    out = gpu.run(
        {"parent": Img("rgba16f", data=parent), "lvl": Img("rgba16f", data=level)},
        [Dispatch("lumina/lumina_rc_merge.comp.glsl", {0: ("parent", "sampled_nearest"), 1: ("lvl", "storage")},
                  pc.merge(probes, parent_probes, block, parent_block, 8.0, 16.0), groups_for(*size))], ["lvl"])["lvl"]
    np.testing.assert_allclose(out[:, : size[0] // 2], np.broadcast_to((0.2, 0.4, 0.6, 0.5), out[:, : size[0] // 2].shape), atol=2e-3)
    np.testing.assert_allclose(out[:, size[0] // 2:], np.broadcast_to((1.0, 0.5, 0.25, 0.0), out[:, size[0] // 2:].shape), atol=2e-3)


def test_red_wall_bleeds_onto_nearby_floor(gpu):
    cam = Camera()
    g = cornell(cam)
    irr = run_cascades(gpu, cam, g)
    floor = (g["normal"][..., 1] > 0.9) & (g["depth"] > 0)
    h, w = floor.shape
    xs = np.arange(w)[None, :].repeat(h, 0)
    near_wall = floor & (xs > w * 0.6)
    far_wall = floor & (xs < w * 0.3)
    assert near_wall.sum() > 20 and far_wall.sum() > 20
    red_near = irr[..., 0][near_wall].mean()
    red_far = irr[..., 0][far_wall].mean()
    assert red_near > 0.02, red_near
    assert red_near > 2.0 * red_far
    # Light from a pure red emitter stays red.
    assert irr[..., 0][near_wall].mean() > 5.0 * irr[..., 1][near_wall].mean()


def test_black_scene_has_no_gi(gpu):
    cam = Camera()
    irr = run_cascades(gpu, cam, cornell(cam, red=(0, 0, 0), floor=(0, 0, 0)))
    assert np.abs(irr[..., :3]).max() < 1e-4


def test_sky_pixels_get_no_irradiance(gpu):
    cam = Camera()
    g = render(cam, [Plane((0, -1, 0), (0, 1, 0), (1, 1, 1))])   # floor only; upper half is sky
    irr = run_cascades(gpu, cam, g)
    sky = g["depth"] <= 0
    assert sky.sum() > 100
    assert np.abs(irr[sky]).max() < 1e-6


def _temporal(gpu, cam, g, current, prev, velocity=None, flags=0):
    h, w = current.shape[:2]
    velocity = np.zeros((h, w, 2)) if velocity is None else velocity
    imgs = {
        "irr": Img("rgba16f", data=current), "phist": Img("rgba16f", data=prev["hist"]),
        "pmom": Img("rg16f", data=prev["mom"]), "pgeo": Img("rgba16f", data=prev["geo"]),
        "vel": Img("rg16f", data=velocity), "depth": Img("r32f", data=g["depth"]), "normal": Img("rgba8", data=g["normal"]),
        "hist": Img("rgba16f", w, h), "mom": Img("rg16f", w, h), "geo": Img("rgba16f", w, h), "filt": Img("rgba16f", w, h),
    }
    return gpu.run(imgs, [Dispatch(
        "lumina/lumina_svgf_temporal.comp.glsl",
        {0: ("irr", "sampled_nearest"), 1: ("phist", "sampled_linear"), 2: ("pmom", "sampled_linear"), 3: ("pgeo", "sampled_nearest"),
         4: ("vel", "sampled_nearest"), 5: ("depth", "sampled_nearest"), 6: ("normal", "sampled_nearest"),
         7: ("hist", "storage"), 8: ("mom", "storage"), 9: ("geo", "storage"), 10: ("filt", "storage")},
        pc.temporal(w, h, cam.depth_params, flags=flags), groups_for(w, h))], ["hist", "mom", "geo", "filt"])


def test_svgf_temporal_accumulates_and_resets(gpu):
    cam = Camera(width=32, height=24)
    g = render(cam, [Plane((0, 0, -4), (0, 0, 1), (1, 1, 1))])
    h, w = 24, 32
    empty = {"hist": np.zeros((h, w, 4)), "mom": np.zeros((h, w, 2)), "geo": np.zeros((h, w, 4))}
    a = np.zeros((h, w, 4)); a[..., :3] = 0.2
    b = np.zeros((h, w, 4)); b[..., :3] = 0.6

    f1 = _temporal(gpu, cam, g, a, empty, flags=1)
    np.testing.assert_allclose(f1["hist"][..., 3], 1.0, atol=1e-3)          # history length 1
    np.testing.assert_allclose(f1["hist"][..., 0], 0.2, atol=1e-3)
    np.testing.assert_allclose(f1["geo"][..., 3], g["linear_z"], rtol=2e-3)  # linear depth stored

    f2 = _temporal(gpu, cam, g, b, f1)
    np.testing.assert_allclose(f2["hist"][..., 3], 2.0, atol=1e-3)
    np.testing.assert_allclose(f2["hist"][..., 0], 0.4, atol=2e-3)          # alpha = 1/2

    f2_reset = _temporal(gpu, cam, g, b, f1, flags=1)
    np.testing.assert_allclose(f2_reset["hist"][..., 0], 0.6, atol=1e-3)


def test_svgf_temporal_rejects_history_across_depth_change(gpu):
    cam = Camera(width=32, height=24)
    near = render(cam, [Plane((0, 0, -2), (0, 0, 1), (1, 1, 1))])
    far = render(cam, [Plane((0, 0, -8), (0, 0, 1), (1, 1, 1))])
    h, w = 24, 32
    a = np.zeros((h, w, 4)); a[..., :3] = 0.2
    b = np.zeros((h, w, 4)); b[..., :3] = 0.6
    empty = {"hist": np.zeros((h, w, 4)), "mom": np.zeros((h, w, 2)), "geo": np.zeros((h, w, 4))}
    f1 = _temporal(gpu, cam, near, a, empty, flags=1)
    f2 = _temporal(gpu, cam, far, b, f1)        # surface moved: history must be dropped
    np.testing.assert_allclose(f2["hist"][..., 3], 1.0, atol=1e-3)
    np.testing.assert_allclose(f2["hist"][..., 0], 0.6, atol=1e-3)


def _atrous(gpu, cam, g, data, iterations=3):
    h, w = data.shape[:2]
    imgs = {"a": Img("rgba16f", data=data), "b": Img("rgba16f", w, h),
            "depth": Img("r32f", data=g["depth"]), "normal": Img("rgba8", data=g["normal"])}
    dispatches, src = [], "a"
    for it in range(iterations):
        dst = "b" if src == "a" else "a"
        dispatches.append(Dispatch(
            "lumina/lumina_svgf_atrous.comp.glsl",
            {0: (src, "sampled_nearest"), 1: ("depth", "sampled_nearest"), 2: ("normal", "sampled_nearest"), 3: (dst, "storage")},
            pc.atrous(w, h, 1 << it, depth_params=cam.depth_params), groups_for(w, h)))
        src = dst
    return gpu.run(imgs, dispatches, [src])[src]


def test_atrous_preserves_constant_and_reduces_noise(gpu):
    cam = Camera(width=32, height=32)
    g = render(cam, [Plane((0, 0, -4), (0, 0, 1), (1, 1, 1))])
    const = np.zeros((32, 32, 4)); const[..., :3] = 0.5; const[..., 3] = 0.01
    out = _atrous(gpu, cam, g, const)
    np.testing.assert_allclose(out[..., :3], 0.5, atol=2e-3)

    rng = np.random.default_rng(7)
    noisy = np.zeros((32, 32, 4)); noisy[..., :3] = 0.5 + rng.normal(0, 0.1, (32, 32, 1)); noisy[..., 3] = 0.01
    out = _atrous(gpu, cam, g, noisy)
    assert out[8:24, 8:24, 0].std() < 0.5 * noisy[8:24, 8:24, 0].std()
    assert out[8:24, 8:24, 0].mean() == pytest.approx(noisy[8:24, 8:24, 0].mean(), abs=0.02)


def test_atrous_does_not_bleed_across_depth_edges(gpu):
    cam = Camera(width=32, height=32)
    left = render(cam, [Plane((0, 0, -2), (0, 0, 1), (1, 1, 1))])
    right = render(cam, [Plane((0, 0, -10), (0, 0, 1), (1, 1, 1))])
    g = {k: left[k].copy() for k in left}
    for k in ("depth", "normal"):
        g[k][:, 16:] = right[k][:, 16:]
    data = np.zeros((32, 32, 4)); data[:, :16, :3] = 1.0; data[..., 3] = 0.01
    out = _atrous(gpu, cam, g, data)
    assert out[:, 13:16, 0].min() > 0.9      # near side stays bright up to the edge
    assert out[:, 16:19, 0].max() < 0.1      # far side stays dark


def test_composite_adds_gi_and_leaves_sky(gpu):
    cam = Camera(width=16, height=16)
    g = render(cam, [Plane((0, -1, 0), (0, 1, 0), (0.2, 0.2, 0.2))])
    gi = np.zeros((16, 16, 4)); gi[..., :3] = (0.1, 0.2, 0.3)
    imgs = {"color": Img("rgba16f", data=g["color"]), "gi": Img("rgba16f", data=gi), "raw": Img("rgba16f", 16, 16),
            "hist": Img("rgba16f", 16, 16), "depth": Img("r32f", data=g["depth"]), "normal": Img("rgba8", data=g["normal"])}
    out = gpu.run(imgs, [Dispatch(
        "lumina/lumina_composite.comp.glsl",
        {0: ("color", "storage"), 1: ("gi", "sampled_nearest"), 2: ("raw", "sampled_nearest"), 3: ("hist", "sampled_nearest"),
         4: ("depth", "sampled_nearest"), 5: ("normal", "sampled_nearest")},
        pc.composite_lumina(16, 16, intensity=2.0, albedo_mode=1), groups_for(16, 16))], ["color"])["color"]
    surf = g["depth"] > 0
    np.testing.assert_allclose(out[surf][:, :3], g["color"][surf][:, :3] + 2.0 * np.array([0.1, 0.2, 0.3]), atol=3e-3)
    np.testing.assert_allclose(out[~surf][:, :3], g["color"][~surf][:, :3], atol=1e-6)
