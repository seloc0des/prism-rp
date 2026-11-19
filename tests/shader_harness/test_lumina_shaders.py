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


def run_cascades(gpu, cam, g, spacing=8, count=3, steps=16, thickness=0.2, prefilter=True):
    """[downsample] -> trace every level -> merge top-down -> integrate. Returns irradiance (h, w, 4).
    As in the C++, level 0 traces the colour buffer and level i the prefiltered radiance pyramid's level
    i - 1 (prefilter=False traces every level against the colour buffer, as before), and the top level
    runs on to the edge of the screen with twice the steps."""
    levels = cascade_layout(cam, spacing, count)
    images = {
        "depth": Img("r32f", data=g["depth"]), "normal": Img("rgba8", data=g["normal"]),
        "color": Img("rgba16f", data=g["color"]), "irr": Img("rgba16f", cam.width, cam.height),
    }
    for i, lv in enumerate(levels):
        images[f"c{i}"] = Img("rgba16f", *lv["size"])
    dispatches = []
    if prefilter:
        for i in range(count - 1):
            div = 2 << i
            size = ((cam.width + div - 1) // div, (cam.height + div - 1) // div)
            images[f"p{i}"] = Img("rgba16f", *size)
            dispatches.append(Dispatch(
                "lumina/lumina_radiance_downsample.comp.glsl",
                {0: ("color" if i == 0 else f"p{i - 1}", "sampled_linear"), 1: (f"p{i}", "storage")},
                pc.downsample(*size), groups_for(*size)))
    for i, lv in enumerate(levels):
        scale = 4.0 ** i
        radiance = "color" if i == 0 or not prefilter else f"p{i - 1}"
        dispatches.append(Dispatch(
            "lumina/lumina_rc_trace.comp.glsl",
            {0: ("depth", "sampled_nearest"), 1: ("normal", "sampled_nearest"), 2: (radiance, "sampled_linear"), 3: (f"c{i}", "storage")},
            pc.trace(cam.width, cam.height, lv["probes"], lv["block"], float(lv["spacing"]),
                     spacing * (scale - 1) / 3.0, spacing * scale if i + 1 < count else -1.0,
                     steps if i + 1 < count else 2 * steps, thickness, 0, 0.01, cam.proj, cam.depth_params),
            groups_for(*lv["size"])))
    for i in range(count - 2, -1, -1):
        lv, parent = levels[i], levels[i + 1]
        dispatches.append(Dispatch(
            "lumina/lumina_rc_merge.comp.glsl",
            {0: (f"c{i + 1}", "sampled_nearest"), 1: (f"c{i}", "storage"), 2: ("depth", "sampled_nearest")},
            pc.merge(lv["probes"], parent["probes"], lv["block"], parent["block"], float(lv["spacing"]), float(parent["spacing"]),
                     (cam.width, cam.height), cam.depth_params),
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


def test_integrate_clamps_a_single_hot_direction(gpu):
    # One direction of every probe lands on a tiny, very bright spot (the hotspot a light makes on a
    # nearby wall); the rest see dim surroundings. Unclamped, that one sample lights the whole probe and
    # leaves probe-sized blobs (found in the Cornell box). A broad bright area, seen by many directions,
    # must still come through: see the uniform test above.
    cam = Camera(width=32, height=32)
    g = render(cam, [Plane((0, 0, -5), (0, 0, 1), (0, 0, 0))])
    lv = cascade_layout(cam, 8, 1)[0]
    block = lv["block"]
    c0 = np.zeros((lv["size"][1], lv["size"][0], 4)); c0[..., :3] = 0.1
    c0[block // 2::block, block // 2::block, :3] = 200.0      # one direction near the normal, per probe
    out = gpu.run(
        {"c0": Img("rgba16f", data=c0), "depth": Img("r32f", data=g["depth"]), "normal": Img("rgba8", data=g["normal"]),
         "irr": Img("rgba16f", 32, 32)},
        [Dispatch("lumina/lumina_rc_integrate.comp.glsl",
                  {0: ("c0", "sampled_nearest"), 1: ("depth", "sampled_nearest"), 2: ("normal", "sampled_nearest"), 3: ("irr", "storage")},
                  pc.integrate(32, 32, lv["probes"], lv["block"], 8.0, cam.depth_params), groups_for(32, 32))], ["irr"])["irr"]
    assert out[8:24, 8:24, 0].mean() < 0.4, out[8:24, 8:24, 0].mean()   # unclamped: about 20


def test_radiance_downsample_averages_2x2_blocks(gpu):
    rng = np.random.default_rng(3)
    src = np.zeros((16, 24, 4)); src[..., :3] = rng.uniform(0, 4, (16, 24, 3)); src[..., 3] = 1
    out = gpu.run({"src": Img("rgba16f", data=src), "dst": Img("rgba16f", 12, 8)},
                  [Dispatch("lumina/lumina_radiance_downsample.comp.glsl",
                            {0: ("src", "sampled_linear"), 1: ("dst", "storage")}, pc.downsample(12, 8), groups_for(12, 8))],
                  ["dst"])["dst"]
    expected = src[..., :3].reshape(8, 2, 12, 2, 3).mean(axis=(1, 3))
    np.testing.assert_allclose(out[..., :3], expected, rtol=2e-3, atol=2e-3)


def test_a_small_bright_patch_lights_steadily_as_it_moves(gpu):
    # A small, bright patch on the back wall (like the spot a point light makes on a nearby wall),
    # moved a few centimetres at a time. Coarse levels sampling it at full resolution hit it at full
    # strength or miss it outright, so the light it throws flickers from step to step and gathers into
    # blobs on the probes that happen to hit it. Tracing a prefiltered copy spreads each sample over the
    # ray's footprint: the bounce light stays steady (and the patch is no longer mostly missed).
    cam = Camera(width=128, height=96)
    dim = (0.05, 0.05, 0.05)
    frames = {False: [], True: []}
    for dx in np.linspace(0.0, 0.12, 5):
        x0 = 0.3 + dx
        g = render(cam, [
            Plane((0, -1, 0), (0, 1, 0), dim), Plane((0, 0, -6), (0, 0, 1), dim),
            Plane((-1.5, 0, 0), (1, 0, 0), dim), Plane((1.5, 0, 0), (-1, 0, 0), dim),
            Plane((x0, 0.3, -5.99), (0, 0, 1), (40.0, 40.0, 40.0), bounds=((x0 - 0.15, x0 + 0.15), (0.15, 0.45), (-6.5, -5.5))),
        ])
        walls = (g["linear_z"] > 0) & (g["color"][..., 0] < 1.0)
        for prefilter in (False, True):
            frames[prefilter].append(np.where(walls, run_cascades(gpu, cam, g, prefilter=prefilter)[..., 0], np.nan))
    flicker = {k: float(np.nanmean(np.nanstd(np.stack(v), axis=0)) / np.nanmean(np.stack(v))) for k, v in frames.items()}
    assert flicker[True] < 0.5 * flicker[False] and flicker[True] < 0.08, flicker   # measured: 0.03 vs 0.15


def test_merge_combines_interval_transmittance_and_parent(gpu):
    probes, parent_probes = (4, 4), (2, 2)
    block, parent_block = 4, 8
    size = (probes[0] * block, probes[1] * block)
    psize = (parent_probes[0] * parent_block, parent_probes[1] * parent_block)
    parent = np.zeros((psize[1], psize[0], 4)); parent[..., :3] = (0.2, 0.4, 0.6); parent[..., 3] = 0.5
    level = np.zeros((size[1], size[0], 4))
    level[:, : size[0] // 2] = (0.0, 0.0, 0.0, 1.0)      # clear interval: sees the parent
    level[:, size[0] // 2:] = (1.0, 0.5, 0.25, 0.0)      # blocked interval: parent hidden
    cam = Camera(width=32, height=32)
    flat = render(cam, [Plane((0, 0, -4), (0, 0, 1), (1, 1, 1))])   # one depth everywhere: plain bilinear
    out = gpu.run(
        {"parent": Img("rgba16f", data=parent), "lvl": Img("rgba16f", data=level), "depth": Img("r32f", data=flat["depth"])},
        [Dispatch("lumina/lumina_rc_merge.comp.glsl",
                  {0: ("parent", "sampled_nearest"), 1: ("lvl", "storage"), 2: ("depth", "sampled_nearest")},
                  pc.merge(probes, parent_probes, block, parent_block, 8.0, 16.0, (cam.width, cam.height), cam.depth_params),
                  groups_for(*size))], ["lvl"])["lvl"]
    np.testing.assert_allclose(out[:, : size[0] // 2], np.broadcast_to((0.2, 0.4, 0.6, 0.5), out[:, : size[0] // 2].shape), atol=2e-3)
    np.testing.assert_allclose(out[:, size[0] // 2:], np.broadcast_to((1.0, 0.5, 0.25, 0.0), out[:, size[0] // 2:].shape), atol=2e-3)


def test_merge_does_not_take_parents_from_another_depth(gpu):
    # Parent probes on a near surface hold light this probe (on a far surface) can't see; with a plain
    # screen-space blend it inherited part of it. Depth weights keep only the parent at its depth.
    probes, parent_probes = (4, 4), (2, 2)
    block, parent_block = 4, 8
    size = (probes[0] * block, probes[1] * block)
    psize = (parent_probes[0] * parent_block, parent_probes[1] * parent_block)
    cam = Camera(width=32, height=32)
    # Left half of the screen far (z = 8), right half near (z = 2).
    g = render(cam, [Plane((0, 0, -8), (0, 0, 1), (1, 1, 1)),
                     Plane((0, 0, -2), (0, 0, 1), (1, 1, 1), bounds=((0.0, 10.0), (-10.0, 10.0), (-3.0, -1.0)))])
    parent = np.zeros((psize[1], psize[0], 4))
    parent[:, : psize[0] // 2] = (0.1, 0.1, 0.1, 1.0)     # parent probe on the far half
    parent[:, psize[0] // 2:] = (5.0, 0.0, 0.0, 1.0)      # parent probe on the near half: bright red
    level = np.zeros((size[1], size[0], 4)); level[..., 3] = 1.0   # every interval clear: all parent
    out = gpu.run(
        {"parent": Img("rgba16f", data=parent), "lvl": Img("rgba16f", data=level), "depth": Img("r32f", data=g["depth"])},
        [Dispatch("lumina/lumina_rc_merge.comp.glsl",
                  {0: ("parent", "sampled_nearest"), 1: ("lvl", "storage"), 2: ("depth", "sampled_nearest")},
                  pc.merge(probes, parent_probes, block, parent_block, 8.0, 16.0, (cam.width, cam.height), cam.depth_params),
                  groups_for(*size))], ["lvl"])["lvl"]
    # Child probe column 1 (centre x = 12 px) is on the far half but between the two parent probes.
    far_child = out[:, 1 * block:(1 + 1) * block, 0]
    assert far_child.max() < 0.2, far_child.max()


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


def test_light_does_not_leak_through_the_back_of_objects(gpu):
    # A bright red card faces the camera. Floor in front of it sees its lit face and should turn red;
    # floor behind it only sees its unlit back. A screen-space ray passing behind the card lands
    # inside its depth "thickness" and used to pick up the lit front face's colour: the red and blue
    # smears found on Sponza's columns.
    cam = Camera(width=96, height=64)
    floor = (0.05, 0.05, 0.05)
    g = render(cam, [
        Plane((0, -1, 0), (0, 1, 0), floor),
        Plane((0, 0, -7), (0, 0, 1), floor),
        Plane((0, 0, -3.5), (0, 0, 1), (4.0, 0.0, 0.0), bounds=((-0.6, 0.6), (-1.0, 0.6), (-4.0, -3.0))),
    ])
    irr = run_cascades(gpu, cam, g, thickness=0.2)
    pos = cam.ray_dirs() * g["linear_z"][..., None]
    on_floor = (g["normal"][..., 1] > 0.9) & (g["linear_z"] > 0)
    beside = (np.abs(pos[..., 0]) > 0.7) & (np.abs(pos[..., 0]) < 1.6)
    front = on_floor & beside & (pos[..., 2] > -3.0) & (pos[..., 2] < -2.0)
    behind = on_floor & beside & (pos[..., 2] > -5.5) & (pos[..., 2] < -4.2)
    assert front.sum() > 20 and behind.sum() > 20
    red_front = float(irr[front, 0].mean())
    red_behind = float(irr[behind, 0].mean())
    assert red_front > 0.05                      # the lit face does bounce onto the floor in front
    assert red_behind < 0.25 * red_front, (red_front, red_behind)


def rect_irradiance(point, normal, a, b, c, n=64):
    """E / pi at `point` (surface normal `normal`) from a Lambertian rectangle of radiance 1 spanned by
    a->b and a->c, all mutually visible: sum of cos * cos' / r^2 dA over the rectangle, divided by pi."""
    a, b, c = (np.asarray(v, np.float64) for v in (a, b, c))
    s = (np.arange(n) + 0.5) / n
    q = a + (b - a) * s[:, None, None] + (c - a) * s[None, :, None]
    area = np.linalg.norm(np.cross(b - a, c - a))
    nq = np.cross(b - a, c - a) / area
    d = q - np.asarray(point, np.float64)
    r2 = np.sum(d * d, axis=-1)
    d /= np.sqrt(r2)[..., None]
    if np.mean(d @ nq) > 0:
        nq = -nq                       # face the point
    cos_p = np.clip(d @ np.asarray(normal, np.float64), 0.0, None)
    cos_q = np.clip(-(d @ nq), 0.0, None)
    return float(np.sum(cos_p * cos_q / r2) * area / (n * n) / np.pi)


def corridor(cam, far_color, near_color=0.1, split=-7.0, end=-14.0):
    """Camera looking down a closed corridor x in [-1.5, 1.5], y in [-1, 1.5], back wall at z = end.
    Surfaces nearer than z = split are near_color, the rest (and the back wall) far_color."""
    big = 50.0
    planes = []
    for color, (z0, z1) in ((near_color, (split, big)), (far_color, (end - 1.0, split))):
        rgb = (color,) * 3
        planes += [
            Plane((0, -1, 0), (0, 1, 0), rgb, bounds=((-big, big), (-big, big), (z0, z1))),
            Plane((0, 1.5, 0), (0, -1, 0), rgb, bounds=((-big, big), (-big, big), (z0, z1))),
            Plane((-1.5, 0, 0), (1, 0, 0), rgb, bounds=((-big, big), (-big, big), (z0, z1))),
            Plane((1.5, 0, 0), (-1, 0, 0), rgb, bounds=((-big, big), (-big, big), (z0, z1))),
        ]
    planes.append(Plane((0, 0, end), (0, 0, 1), (far_color,) * 3))
    return render(cam, planes)


def test_walls_near_the_camera_see_the_far_end_of_the_room(gpu):
    # Gate 2 on a real GPU: walls and floor close to the camera came out nearly black. Each interval's
    # length was converted to world units at the probe's own depth, so a probe 2 m from the camera
    # could only see 2 m. Here the far half of a corridor is lit and the near half is dark: the near
    # left wall must still get the far half's light, as much as a direct integration says.
    cam = Camera(width=96, height=72)
    split, end, lit = -5.0, -14.0, 0.6
    base = run_cascades(gpu, cam, corridor(cam, far_color=0.1, split=split, end=end), spacing=8, count=2)
    g = corridor(cam, far_color=lit, split=split, end=end)
    irr = run_cascades(gpu, cam, g, spacing=8, count=2)
    pos = cam.ray_dirs() * g["linear_z"][..., None]
    left = (g["normal"][..., 0] > 0.9) & (g["linear_z"] > 0)
    near = left & (pos[..., 2] < -2.2) & (pos[..., 2] > -3.5) & (pos[..., 1] > -0.5) & (pos[..., 1] < 1.0)
    assert near.sum() > 30, near.sum()
    gained = float((irr[near, :3] - base[near, :3]).mean())

    # Expected: (lit - 0.1) times what the far floor, ceiling, right wall and back wall send to the
    # near wall's centre (the left wall's own far part is coplanar and sends nothing).
    p, n = (-1.5, 0.25, -2.85), (1.0, 0.0, 0.0)
    rects = [
        ((-1.5, -1, split), (1.5, -1, split), (-1.5, -1, end)),     # floor
        ((-1.5, 1.5, split), (1.5, 1.5, split), (-1.5, 1.5, end)),  # ceiling
        ((1.5, -1, split), (1.5, 1.5, split), (1.5, -1, end)),      # right wall
        ((-1.5, -1, end), (1.5, -1, end), (-1.5, 1.5, end)),        # back wall
    ]
    expected = (lit - 0.1) * sum(rect_irradiance(p, n, *r) for r in rects)
    assert expected > 0.02, expected
    assert 0.7 * expected < gained < 1.3 * expected, (gained, expected)


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
    # A change within the noise band (see the next test for a real lighting change).
    a = np.zeros((h, w, 4)); a[..., :3] = 0.50
    b = np.zeros((h, w, 4)); b[..., :3] = 0.55

    f1 = _temporal(gpu, cam, g, a, empty, flags=1)
    np.testing.assert_allclose(f1["hist"][..., 3], 1.0, atol=1e-3)          # history length 1
    np.testing.assert_allclose(f1["hist"][..., 0], 0.50, atol=1e-3)
    np.testing.assert_allclose(f1["geo"][..., 3], g["linear_z"], rtol=2e-3)  # linear depth stored

    f2 = _temporal(gpu, cam, g, b, f1)
    np.testing.assert_allclose(f2["hist"][..., 3], 2.0, atol=1e-3)
    np.testing.assert_allclose(f2["hist"][..., 0], 0.525, atol=2e-3)        # alpha = 1/2

    f2_reset = _temporal(gpu, cam, g, b, f1, flags=1)
    np.testing.assert_allclose(f2_reset["hist"][..., 0], 0.55, atol=1e-3)


def _settled_history(g, value, length=32.0):
    h, w = g["depth"].shape
    hist = np.zeros((h, w, 4)); hist[..., :3] = value; hist[..., 3] = length
    mom = np.zeros((h, w, 2)); mom[..., 0] = value; mom[..., 1] = value * value
    geo = np.zeros((h, w, 4)); geo[..., 2] = 1.0; geo[..., 3] = g["linear_z"]
    return {"hist": hist, "mom": mom, "geo": geo}


def test_svgf_temporal_follows_a_lighting_change_within_frames(gpu):
    # A light switches on: the raw irradiance triples. Long history must not hold the result back for
    # many frames (Gate 2: settled within 10% in 6 frames). Blending alone, at alpha_min 0.1, is still
    # 47% off after 6 frames.
    cam = Camera(width=32, height=24)
    g = render(cam, [Plane((0, 0, -4), (0, 0, 1), (1, 1, 1))])
    h, w = 24, 32
    bright = np.zeros((h, w, 4)); bright[..., :3] = 0.6
    frame = _settled_history(g, 0.2)
    errors = []
    for _ in range(4):
        frame = _temporal(gpu, cam, g, bright, frame)
        errors.append(abs(float(frame["hist"][8:16, 8:24, 0].mean()) - 0.6) / 0.6)
    assert errors[0] < 0.10 and all(e1 < e0 for e0, e1 in zip(errors, errors[1:])) and errors[-1] < 0.05, errors


def test_svgf_temporal_still_averages_out_noise(gpu):
    # Frame-to-frame noise in the raw irradiance stays inside the clamp band, so history keeps averaging it.
    cam = Camera(width=32, height=24)
    g = render(cam, [Plane((0, 0, -4), (0, 0, 1), (1, 1, 1))])
    h, w = 24, 32
    rng = np.random.default_rng(7)
    frame = _settled_history(g, 0.5)
    noisy = None
    for _ in range(24):
        noisy = np.zeros((h, w, 4)); noisy[..., :3] = 0.5 + rng.uniform(-0.05, 0.05, (h, w, 1))
        frame = _temporal(gpu, cam, g, noisy, frame)
    out = frame["hist"][2:-2, 2:-2, 0]
    assert out.std() < 0.4 * noisy[2:-2, 2:-2, 0].std(), (out.std(), noisy[..., 0].std())
    assert abs(out.mean() - 0.5) < 0.01


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
