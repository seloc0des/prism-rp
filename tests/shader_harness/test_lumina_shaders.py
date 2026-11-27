"""Lumina-Cascades compute shaders on lavapipe with analytic scenes."""

import numpy as np
import pytest

import pc
from harness import Dispatch, Img, groups_for
from scene import Camera, Plane, cornell, irradiance_truth, render

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
    """[radiance mip chain] -> trace every level -> merge top-down -> integrate. Returns irradiance (h, w, 4).
    As in the C++, mip 0 of the chain is the colour buffer plus linear depth and every level traces the
    whole chain, each hit at the mip that matches its cone's footprint (prefilter=False traces the colour
    buffer alone, so every hit is sampled at full resolution), level 0 traces one ray per direction (unless it
    is also the top) and the others 2 x 2, and the top level runs on to the edge of the screen with twice the
    steps."""
    levels = cascade_layout(cam, spacing, count)
    mips = int(np.log2(max(cam.width, cam.height))) + 1
    images = {
        "depth": Img("r32f", data=g["depth"]), "normal": Img("rgba8", data=g["normal"]),
        "color": Img("rgba16f", data=g["color"]), "irr": Img("rgba16f", cam.width, cam.height),
    }
    for i, lv in enumerate(levels):
        images[f"c{i}"] = Img("rgba16f", *lv["size"])
    dispatches = []
    if prefilter:
        images["radiance"] = Img("rgba16f", cam.width, cam.height, mips=mips)
        dispatches.append(Dispatch(
            "lumina/lumina_radiance_copy.comp.glsl",
            {0: ("color", "sampled_nearest"), 1: ("depth", "sampled_nearest"), 2: ("radiance", "storage", 0)},
            pc.radiance_copy(cam.width, cam.height, cam.depth_params), groups_for(cam.width, cam.height)))
        for m in range(1, mips):
            size = (max(1, cam.width >> m), max(1, cam.height >> m))
            dispatches.append(Dispatch(
                "lumina/lumina_radiance_downsample.comp.glsl",
                {0: ("radiance", "sampled_nearest", m - 1), 1: ("radiance", "storage", m)},
                pc.downsample(*size), groups_for(*size)))
    radiance = "radiance" if prefilter else "color"
    for i, lv in enumerate(levels):
        scale = 4.0 ** i
        dispatches.append(Dispatch(
            "lumina/lumina_rc_trace.comp.glsl",
            {0: ("depth", "sampled_nearest"), 1: ("normal", "sampled_nearest"), 2: (radiance, "sampled_linear"), 3: (f"c{i}", "storage")},
            pc.trace(cam.width, cam.height, lv["probes"], lv["block"], float(lv["spacing"]),
                     spacing * (scale - 1) / 3.0, spacing * scale if i + 1 < count else -1.0,
                     steps if i + 1 < count else 2 * steps, thickness, 0, 0.01, cam.proj, cam.depth_params,
                     sub_rays=1 if i == 0 and i + 1 < count else 2),
            groups_for(*lv["size"])))
    for i in range(count - 2, -1, -1):
        lv, parent = levels[i], levels[i + 1]
        dispatches.append(Dispatch(
            "lumina/lumina_rc_merge.comp.glsl",
            {0: (f"c{i + 1}", "sampled_linear"), 1: (f"c{i}", "storage"), 2: ("depth", "sampled_nearest"),
             3: ("normal", "sampled_nearest")},
            pc.merge(lv["probes"], parent["probes"], lv["block"], parent["block"], float(lv["spacing"]), float(parent["spacing"]),
                     (cam.width, cam.height), cam.depth_params),
            groups_for(*lv["size"])))
    lv0 = levels[0]
    dispatches.append(Dispatch(
        "lumina/lumina_rc_integrate.comp.glsl",
        {0: ("c0", "sampled_nearest"), 1: ("depth", "sampled_nearest"), 2: ("normal", "sampled_nearest"), 3: ("irr", "storage")},
        pc.integrate(cam.width, cam.height, lv0["probes"], lv0["block"], float(lv0["spacing"]), cam.depth_params, cam.proj),
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
                  pc.integrate(32, 32, lv["probes"], lv["block"], 8.0, cam.depth_params, cam.proj), groups_for(32, 32))], ["irr"])["irr"]
    np.testing.assert_allclose(out[8:24, 8:24, :3].mean(axis=(0, 1)), L, rtol=0.01)


def test_integrate_blends_each_probes_own_cosine_weighted_average(gpu):
    # Every probe holds different radiance, on one flat tilted plane (so only the bilinear weights vary).
    cam = Camera(width=64, height=48)
    nrm = np.array([0.2, 0.3, 1.0]) / np.linalg.norm([0.2, 0.3, 1.0])
    g = render(cam, [Plane((0, 0, -4), tuple(nrm), (1, 1, 1))])
    assert (g["depth"] > 0).all()
    lv = cascade_layout(cam, 8, 1)[0]
    block, spacing = lv["block"], 8.0
    px, py = lv["probes"]
    rng = np.random.default_rng(5)
    c0 = rng.uniform(0, 1, (lv["size"][1], lv["size"][0], 4)).astype(np.float16).astype(np.float64)
    out = gpu.run(
        {"c0": Img("rgba16f", data=c0), "depth": Img("r32f", data=g["depth"]), "normal": Img("rgba8", data=g["normal"]),
         "irr": Img("rgba16f", 64, 48)},
        [Dispatch("lumina/lumina_rc_integrate.comp.glsl",
                  {0: ("c0", "sampled_nearest"), 1: ("depth", "sampled_nearest"), 2: ("normal", "sampled_nearest"), 3: ("irr", "storage")},
                  pc.integrate(64, 48, lv["probes"], block, spacing, cam.depth_params, cam.proj), groups_for(64, 48))], ["irr"])["irr"]

    def direction(x, y):
        e = (np.array([x, y]) + 0.5) / block * 2.0 - 1.0
        v = np.array([e[0], e[1], 1.0 - abs(e[0]) - abs(e[1])])
        if v[2] < 0:
            v[:2] = (1.0 - np.abs(v[1::-1])) * np.where(v[:2] >= 0, 1.0, -1.0)
        return v / np.linalg.norm(v)

    dirs = np.array([[direction(x, y) for x in range(block)] for y in range(block)])
    n = g["normal"][..., :3] * 2.0 - 1.0
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    expected = np.zeros((48, 64, 3))
    for py_ in range(48):
        for px_ in range(64):
            cos = np.maximum(dirs @ n[py_, px_], 0.0)
            coord = (np.array([px_, py_]) + 0.5) / spacing - 0.5
            p0 = np.floor(coord).astype(int)
            f = coord - p0
            for j in range(4):
                o = np.array([j & 1, j >> 1])
                pr = np.clip(p0 + o, 0, np.array([px, py]) - 1)
                w = (f[0] if o[0] else 1 - f[0]) * (f[1] if o[1] else 1 - f[1])
                tile = c0[pr[1] * block:(pr[1] + 1) * block, pr[0] * block:(pr[0] + 1) * block, :3]
                expected[py_, px_] += w * (tile * cos[..., None]).sum(axis=(0, 1)) / cos.sum()
    np.testing.assert_allclose(out[..., :3], expected, atol=2e-3)


def test_integrate_lights_a_surface_seen_at_a_grazing_angle(gpu):
    # A floor seen from just above it: across 8 px its depth changes by far more than 5%, though every probe
    # sits on the same plane. Uniform light must reach every floor pixel; a depth-difference test rejected
    # the probes and left grazing surfaces (Sponza's gallery undersides, the walls beside the camera) black.
    cam = Camera(width=64, height=48)
    g = render(cam, [Plane((0, -0.15, 0), (0, 1, 0), (0, 0, 0))])
    lv = cascade_layout(cam, 8, 1)[0]
    L = np.array([0.3, 0.6, 0.9])
    c0 = np.zeros((lv["size"][1], lv["size"][0], 4)); c0[..., :3] = L
    out = gpu.run(
        {"c0": Img("rgba16f", data=c0), "depth": Img("r32f", data=g["depth"]), "normal": Img("rgba8", data=g["normal"]),
         "irr": Img("rgba16f", 64, 48)},
        [Dispatch("lumina/lumina_rc_integrate.comp.glsl",
                  {0: ("c0", "sampled_nearest"), 1: ("depth", "sampled_nearest"), 2: ("normal", "sampled_nearest"), 3: ("irr", "storage")},
                  pc.integrate(64, 48, lv["probes"], lv["block"], 8.0, cam.depth_params, cam.proj), groups_for(64, 48))], ["irr"])["irr"]
    floor = g["depth"] > 0
    lit = out[..., 1][floor]
    assert (lit > 0.5 * L[1]).mean() == 1.0, f"{(lit <= 0.5 * L[1]).mean():.0%} of floor pixels unlit"


def test_radiance_copy_stores_colour_and_linear_depth(gpu):
    cam = Camera(width=32, height=24)
    g = render(cam, [Plane((0, -1, 0), (0, 1, 0), (0.5, 0.25, 2.0))])   # floor only; upper half is sky
    floor = g["linear_z"] > 0
    color = g["color"].copy()
    color[~floor, :3] = (0.3, 0.6, 1.0)               # a background colour on the sky
    out = gpu.run({"color": Img("rgba16f", data=color), "depth": Img("r32f", data=g["depth"]), "dst": Img("rgba16f", 32, 24)},
                  [Dispatch("lumina/lumina_radiance_copy.comp.glsl",
                            {0: ("color", "sampled_nearest"), 1: ("depth", "sampled_nearest"), 2: ("dst", "storage")},
                            pc.radiance_copy(32, 24, cam.depth_params), groups_for(32, 24))],
                  ["dst"])["dst"]
    assert floor.sum() > 50 and (~floor).sum() > 50
    np.testing.assert_allclose(out[floor, :3], g["color"][floor, :3], rtol=2e-3)
    np.testing.assert_allclose(out[floor, 3], g["linear_z"][floor], rtol=2e-3)
    assert (out[~floor, :3] == 0.0).all()           # rays find no light in the sky, so none blurs off it
    assert (out[~floor, 3] > 1e4).all()             # and the sky is never nearer than a hit


@pytest.mark.parametrize("src_w, src_h", [(24, 16), (25, 17), (5, 3)])
def test_radiance_downsample_averages_colour_and_keeps_the_nearest_depth(gpu, src_w, src_h):
    # Odd sizes: the last texel takes in the row or column left over, so nothing at the edges is lost.
    rng = np.random.default_rng(3)
    src = np.zeros((src_h, src_w, 4)); src[..., :3] = rng.uniform(0, 4, (src_h, src_w, 3)); src[..., 3] = rng.uniform(1, 50, (src_h, src_w))
    w, h = src_w // 2, src_h // 2
    out = gpu.run({"src": Img("rgba16f", data=src), "dst": Img("rgba16f", w, h)},
                  [Dispatch("lumina/lumina_radiance_downsample.comp.glsl",
                            {0: ("src", "sampled_nearest"), 1: ("dst", "storage")}, pc.downsample(w, h), groups_for(w, h))],
                  ["dst"])["dst"]
    half = src.astype(np.float16).astype(np.float64)
    for y in range(h):
        for x in range(w):
            x1 = src_w if x == w - 1 else 2 * x + 2
            y1 = src_h if y == h - 1 else 2 * y + 2
            block = half[2 * y:y1, 2 * x:x1]
            np.testing.assert_allclose(out[y, x, :3], block[..., :3].mean(axis=(0, 1)), rtol=2e-3, atol=2e-3)
            np.testing.assert_allclose(out[y, x, 3], block[..., 3].min(), rtol=1e-3)


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
    assert flicker[True] < 0.5 * flicker[False] and flicker[True] < 0.08, flicker   # measured: 0.05 vs 0.13


def test_merge_averages_each_parent_probes_2x2_directions_and_blends_across_probes(gpu):
    probes, parent_probes = (7, 5), (4, 3)
    block, parent_block = 4, 8
    spacing, parent_spacing = 8.0, 16.0
    size = (probes[0] * block, probes[1] * block)
    psize = (parent_probes[0] * parent_block, parent_probes[1] * parent_block)
    rng = np.random.default_rng(3)
    parent = rng.uniform(0, 1, (psize[1], psize[0], 4)).astype(np.float16).astype(np.float64)
    level = np.zeros((size[1], size[0], 4)); level[..., 3] = 1.0
    cam = Camera(width=56, height=40)
    flat = render(cam, [Plane((0, 0, -4), (0, 0, 1), (1, 1, 1))])   # one depth everywhere: every depth weight is 1
    out = gpu.run(
        {"parent": Img("rgba16f", data=parent), "lvl": Img("rgba16f", data=level), "depth": Img("r32f", data=flat["depth"]), "normal": Img("rgba8", data=flat["normal"])},
        [Dispatch("lumina/lumina_rc_merge.comp.glsl",
                  {0: ("parent", "sampled_linear"), 1: ("lvl", "storage"), 2: ("depth", "sampled_nearest"),
                   3: ("normal", "sampled_nearest")},
                  pc.merge(probes, parent_probes, block, parent_block, spacing, parent_spacing, (cam.width, cam.height), cam.depth_params),
                  groups_for(*size))], ["lvl"])["lvl"]

    expected = np.zeros_like(out)
    for ty in range(size[1]):
        for tx in range(size[0]):
            probe = np.array([tx // block, ty // block])
            direction = np.array([tx, ty]) - probe * block
            coord = (probe + 0.5) * spacing / parent_spacing - 0.5
            p0 = np.floor(coord).astype(int)
            f = coord - p0
            for j in range(4):
                o = np.array([j & 1, j >> 1])
                pp = np.clip(p0 + o, 0, np.array(parent_probes) - 1)
                w = (f[0] if o[0] else 1 - f[0]) * (f[1] if o[1] else 1 - f[1])
                b = pp * parent_block + direction * 2
                expected[ty, tx] += w * parent[b[1]:b[1] + 2, b[0]:b[0] + 2].reshape(-1, 4).mean(axis=0)
    np.testing.assert_allclose(out, expected, atol=1e-3)


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
        {"parent": Img("rgba16f", data=parent), "lvl": Img("rgba16f", data=level), "depth": Img("r32f", data=flat["depth"]), "normal": Img("rgba8", data=flat["normal"])},
        [Dispatch("lumina/lumina_rc_merge.comp.glsl",
                  {0: ("parent", "sampled_linear"), 1: ("lvl", "storage"), 2: ("depth", "sampled_nearest"),
                   3: ("normal", "sampled_nearest")},
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
        {"parent": Img("rgba16f", data=parent), "lvl": Img("rgba16f", data=level), "depth": Img("r32f", data=g["depth"]), "normal": Img("rgba8", data=g["normal"])},
        [Dispatch("lumina/lumina_rc_merge.comp.glsl",
                  {0: ("parent", "sampled_linear"), 1: ("lvl", "storage"), 2: ("depth", "sampled_nearest"),
                   3: ("normal", "sampled_nearest")},
                  pc.merge(probes, parent_probes, block, parent_block, 8.0, 16.0, (cam.width, cam.height), cam.depth_params),
                  groups_for(*size))], ["lvl"])["lvl"]
    # Child probe column 1 (centre x = 12 px) is on the far half but between the two parent probes.
    far_child = out[:, 1 * block:(1 + 1) * block, 0]
    assert far_child.max() < 0.2, far_child.max()



def oct_decode(e):
    """lumina_oct_decode: [-1, 1]^2 -> unit direction."""
    v = np.array([e[0], e[1], 1.0 - abs(e[0]) - abs(e[1])])
    if v[2] < 0.0:
        v[:2] = (1.0 - np.abs(v[[1, 0]])) * np.where(v[:2] >= 0.0, 1.0, -1.0)
    return v / np.linalg.norm(v)


def test_merge_ignores_parents_whose_surface_faces_away_from_the_direction(gpu):
    # A parent probe on a wall traces a direction pointing into the wall from behind its own surface and
    # finds nothing, so a floor probe beside it inherited "empty and black" for every direction into the
    # wall. With one depth everywhere, the left half of the screen faces +z and the right half +x; the
    # right parent's answer only counts for directions on its own side.
    probes, parent_probes = (4, 4), (2, 2)
    block, parent_block = 4, 8
    size = (probes[0] * block, probes[1] * block)
    psize = (parent_probes[0] * parent_block, parent_probes[1] * parent_block)
    cam = Camera(width=32, height=32)
    flat = render(cam, [Plane((0, 0, -4), (0, 0, 1), (1, 1, 1))])
    normal = flat["normal"].copy()
    normal[:, 16:, :3] = (1.0, 0.5, 0.5)                   # +x, encoded
    parent = np.zeros((psize[1], psize[0], 4))
    parent[:, : psize[0] // 2] = (0.1, 0.1, 0.1, 1.0)     # parent probe facing +z
    parent[:, psize[0] // 2:] = (5.0, 0.0, 0.0, 1.0)      # parent probe facing +x: bright red
    level = np.zeros((size[1], size[0], 4)); level[..., 3] = 1.0   # every interval clear: all parent
    out = gpu.run(
        {"parent": Img("rgba16f", data=parent), "lvl": Img("rgba16f", data=level),
         "depth": Img("r32f", data=flat["depth"]), "normal": Img("rgba8", data=normal)},
        [Dispatch("lumina/lumina_rc_merge.comp.glsl",
                  {0: ("parent", "sampled_linear"), 1: ("lvl", "storage"), 2: ("depth", "sampled_nearest"),
                   3: ("normal", "sampled_nearest")},
                  pc.merge(probes, parent_probes, block, parent_block, 8.0, 16.0, (cam.width, cam.height), cam.depth_params),
                  groups_for(*size))], ["lvl"])["lvl"]
    # Child probe column 1 (centre x = 12 px) faces +z, a quarter of the way to the +x parent.
    away, toward = [], []
    for dy in range(block):
        for dx in range(block):
            d = oct_decode((np.array([dx, dy]) + 0.5) / block * 2.0 - 1.0)
            if d[2] <= 0.0:
                continue        # into this probe's own surface: both parents face away, the depth blend is used
            red = out[1 * block + dy, 1 * block + dx, 0]
            (away if d[0] < 0.0 else toward).append(red)
    assert away and toward
    np.testing.assert_allclose(away, 0.1, atol=2e-3)                    # the +x parent left out
    np.testing.assert_allclose(toward, 0.75 * 0.1 + 0.25 * 5.0, rtol=2e-3)   # plain bilinear blend


def test_a_level_finds_every_surface_its_ray_crosses_before_the_interval_ends(gpu):
    # Steps sit at (i + jitter) / steps of the interval, so the last one falls short of its end. A ray that
    # crossed a surface after it was missed by its own level, and the next level started behind the
    # surface, saw no crossing and called the direction clear and black: every level boundary lost hits.
    # Floor probes look at a back wall; each direction's crossing is placed on screen exactly, and the
    # level must hit whatever it reaches before the interval's end, and nothing after.
    cam = Camera(width=128, height=96)
    planes = [Plane((0, -1, 0), (0, 1, 0), (1.0, 1.0, 1.0)), Plane((0, 0, -6), (0, 0, 1), (1.0, 1.0, 1.0))]
    g = render(cam, planes)
    spacing, block, start, length = 8, 8, 8.0, 40.0
    probes = (cam.width // spacing, cam.height // spacing)
    size = (probes[0] * block, probes[1] * block)
    out = gpu.run(
        {"depth": Img("r32f", data=g["depth"]), "normal": Img("rgba8", data=g["normal"]),
         "color": Img("rgba16f", data=g["color"]), "c": Img("rgba16f", *size)},
        [Dispatch("lumina/lumina_rc_trace.comp.glsl",
                  {0: ("depth", "sampled_nearest"), 1: ("normal", "sampled_nearest"), 2: ("color", "sampled_linear"), 3: ("c", "storage")},
                  pc.trace(cam.width, cam.height, probes, block, float(spacing), start, length, 6, 0.2, 0, 0.01, cam.proj,
                           cam.depth_params, sub_rays=1),
                  groups_for(*size))], ["c"])["c"]

    p00, p11, _, _ = cam.proj
    def to_px(q):
        return np.array([(p00 * q[0] / -q[2] * 0.5 + 0.5) * cam.width, (p11 * q[1] / -q[2] * 0.5 + 0.5) * cam.height])

    reached, beyond = [], []
    for py in range(probes[1]):
        for px in range(probes[0]):
            cx, cy = int((px + 0.5) * spacing), int((py + 0.5) * spacing)
            z = g["linear_z"][cy, cx]
            if z <= 0 or g["normal"][cy, cx, 1] < 0.9:
                continue                                   # floor probes only
            origin = cam.ray_dirs()[cy, cx] * z
            origin = origin + np.array([0.0, 1.0, 0.0]) * (0.01 * z)
            s0 = to_px(origin)
            for dy in range(block):
                for dx in range(block):
                    d = oct_decode((np.array([dx, dy]) + 0.5) / block * 2.0 - 1.0)
                    if d[1] <= 0.05 or d[2] >= -0.05:
                        continue                           # up off the floor and away from the camera
                    t = (-6.0 - origin[2]) / d[2]
                    hit = origin + d * t
                    sh = to_px(hit)
                    if not (0 <= sh[0] < cam.width and 0 <= sh[1] < cam.height):
                        continue
                    dist = float(np.linalg.norm(sh - s0))
                    a = out[py * block + dy, px * block + dx, 3]
                    if start + 0.75 * length < dist < start + 0.97 * length:
                        reached.append(a)
                    elif start + 1.05 * length < dist < start + 2.0 * length:
                        beyond.append(a)
    assert len(reached) > 40 and len(beyond) > 40, (len(reached), len(beyond))
    assert max(reached) < 0.5, f"{sum(a > 0.5 for a in reached)} of {len(reached)} crossings before the end missed"
    assert min(beyond) > 0.5, f"{sum(a < 0.5 for a in beyond)} of {len(beyond)} crossings after the end counted"

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


@pytest.mark.parametrize("size", [(128, 96), (256, 192)])
def test_floor_behind_a_card_gets_no_light_from_its_front_at_any_resolution(gpu, size):
    # The card scene above passes at 96 x 64 by luck of where the probes fall. At 128 x 96 the level-1
    # probes on the floor sit 0.1 m in front of the card and see its lit face, the level-0 probes behind it
    # match none of their parents' depths, and the merge normalised the least bad parent to full weight:
    # 7.3x the exact light behind the card (3.5x at 256 x 192). Measured against the exact answer, not
    # zero: the grey floor behind the card has light of its own from the back wall. Shipping settings.
    cam = Camera(width=size[0], height=size[1])
    floor = (0.05, 0.05, 0.05)
    planes = [
        Plane((0, -1, 0), (0, 1, 0), floor),
        Plane((0, 0, -7), (0, 0, 1), floor),
        Plane((0, 0, -3.5), (0, 0, 1), (4.0, 0.0, 0.0), bounds=((-0.6, 0.6), (-1.0, 0.6), (-4.0, -3.0))),
    ]
    g = render(cam, planes)
    irr = run_cascades(gpu, cam, g, count=4, steps=6, thickness=0.1)[..., 0]
    truth = irradiance_truth(cam, planes)[..., 0]
    pos = cam.ray_dirs() * g["linear_z"][..., None]
    on_floor = (g["normal"][..., 1] > 0.9) & (g["linear_z"] > 0)
    beside = (np.abs(pos[..., 0]) > 0.7) & (np.abs(pos[..., 0]) < 1.6)
    front = on_floor & beside & (pos[..., 2] > -3.0) & (pos[..., 2] < -2.0)
    behind = on_floor & beside & (pos[..., 2] > -5.5) & (pos[..., 2] < -4.2)
    assert front.sum() > 20 and behind.sum() > 20
    ratio_behind = float(irr[behind].mean() / truth[behind].mean())
    ratio_front = float(irr[front].mean() / truth[front].mean())
    assert 0.5 < ratio_behind < 1.5, ratio_behind   # no card light, and not darker than the floor's own
    assert 0.5 < ratio_front < 1.5, ratio_front     # the lit face still bounces onto the floor in front


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


def black_room(extra):
    """Closed black room around the camera (floor y = -1, ceiling y = 1.5, left wall x = -1.5, back wall
    z = -6) plus `extra` planes, which should include the right wall at x = 1.5."""
    black = (0.0, 0.0, 0.0)
    return [Plane((0, -1, 0), (0, 1, 0), black), Plane((0, 1.5, 0), (0, -1, 0), black),
            Plane((-1.5, 0, 0), (1, 0, 0), black), Plane((0, 0, -6), (0, 0, 1), black)] + extra


def light_in_black_room(gpu, cam, planes, count=4):
    """Lumina's irradiance and the exact one-bounce truth (red channel), and the mask of dark surface
    pixels, which receive all their light from the bright planes. Shipping defaults: 4 levels, 6 steps."""
    g = render(cam, planes)
    irr = run_cascades(gpu, cam, g, count=count, steps=6, thickness=0.1)[..., 0]
    dark = (g["linear_z"] > 0) & (g["color"][..., 0] == 0.0)
    return irr, irradiance_truth(cam, planes)[..., 0], dark, g


# Shipping layout (8 px spacing, 4 levels): the top level's interval starts 8 * (4^3 - 1) / 3 = 168 px from
# its probe, past the edge of a 128 x 96 frame, so small scenes never run it. At 512 x 384 it traces a
# good part of the frame, the way it does at 1080p.
TOP_LEVEL_SIZE = (512, 384)
SIZES = pytest.mark.parametrize("size", [(128, 96), TOP_LEVEL_SIZE], ids=lambda s: f"{s[0]}x{s[1]}")


def bright_wall_room():
    """The far part of the right wall glows; everything else is black."""
    big = 50.0
    return black_room([
        Plane((1.5, 0, 0), (-1, 0, 0), (1.0, 1.0, 1.0), bounds=((-big, big), (-big, big), (-6.5, -3.0))),
        Plane((1.5, 0, 0), (-1, 0, 0), (0.0, 0.0, 0.0), bounds=((-big, big), (-big, big), (-3.0, big))),
    ])


@SIZES
def test_a_bright_wall_lights_a_black_room_as_much_as_it_should(gpu, size):
    # The far part of the right wall glows; everything else is black. A firefly clamp in integrate (any
    # direction much brighter than the probe's mean cut down) cuts this wall's real light too: 0.50x of
    # the truth with one at 8x the mean. What Lumina misses without one (about 0.7x) is mostly near the
    # crease and at the wall's edges, where screen-space mips mix the wall with its black neighbours.
    cam = Camera(*size)
    irr, truth, dark, _ = light_in_black_room(gpu, cam, bright_wall_room())
    ratio = float(irr[dark].mean() / truth[dark].mean())
    assert 0.6 < ratio < 1.2, ratio


@SIZES
def test_a_small_bright_spot_lights_the_whole_black_room(gpu, size):
    # A 0.3 m patch, about 4 px on screen, is the only light. Sampled at full resolution, the coarse
    # levels' fixed directions hit it at full strength or miss it, so a few surfaces get it all and the
    # floor and ceiling almost none; a firefly clamp cuts most of the rest. Tracing each hit at the mip
    # that matches its cone's footprint spreads the patch over every direction it covers. The back wall is
    # only about 32 px wide here, so coarse texels around the patch take in the nearer side walls, floor
    # and ceiling, and the trace steps down to sharper mips to keep from leaking light past silhouettes
    # (see the card test above); the coarse levels' 2 x 2 sub-rays per direction need less blur and keep it lit.
    # (Pixel sizes are at 128 x 96; at 512 x 384 everything is four times wider and the top level runs.)
    cam = Camera(*size)
    x0 = 0.3
    irr, truth, dark, g = light_in_black_room(gpu, cam, black_room([
        Plane((1.5, 0, 0), (-1, 0, 0), (0.0, 0.0, 0.0)),
        Plane((x0, 0.3, -5.99), (0, 0, 1), (40.0, 40.0, 40.0), bounds=((x0 - 0.15, x0 + 0.15), (0.15, 0.45), (-6.5, -5.5))),
    ]))
    ratio = float(irr[dark].mean() / truth[dark].mean())
    assert 0.6 < ratio < 1.2, ratio
    n = g["normal"][..., :3] * 2.0 - 1.0
    surfaces = {"floor": n[..., 1] > 0.9, "ceiling": n[..., 1] < -0.9, "left wall": n[..., 0] > 0.9, "right wall": n[..., 0] < -0.9}
    per_surface = {k: float(irr[m & dark].mean() / truth[m & dark].mean()) for k, m in surfaces.items()}
    assert all(0.5 * ratio < r < 2.0 * ratio for r in per_surface.values()), per_surface


@pytest.mark.xfail(strict=True, reason="the top level loses light: 0.83x of the truth with 4 levels, 0.89x with 3 (docs/NEXT_STEPS.md)")
def test_the_top_level_adds_light_instead_of_losing_it(gpu):
    # A fourth level only reaches further, so the room can't get darker than with three, where level 2
    # is the top and runs to the screen's edge itself. At 512 x 384 it does, on every surface.
    cam = Camera(*TOP_LEVEL_SIZE)
    planes = bright_wall_room()
    three, truth, dark, _ = light_in_black_room(gpu, cam, planes, count=3)
    four = light_in_black_room(gpu, cam, planes, count=4)[0]
    ratio3 = float(three[dark].mean() / truth[dark].mean())
    ratio4 = float(four[dark].mean() / truth[dark].mean())
    assert ratio4 > 0.97 * ratio3, (ratio3, ratio4)


def test_light_does_not_leak_through_the_back_of_objects_with_the_top_level(gpu):
    # The card test above at shipping settings, in a frame large enough for the top level to trace. The
    # floor behind the card still gets the grey floor and back wall's own light, so the leak is measured
    # against the exact answer rather than against zero.
    cam = Camera(*TOP_LEVEL_SIZE)
    floor = (0.05, 0.05, 0.05)
    planes = [
        Plane((0, -1, 0), (0, 1, 0), floor),
        Plane((0, 0, -7), (0, 0, 1), floor),
        Plane((0, 0, -3.5), (0, 0, 1), (4.0, 0.0, 0.0), bounds=((-0.6, 0.6), (-1.0, 0.6), (-4.0, -3.0))),
    ]
    g = render(cam, planes)
    irr = run_cascades(gpu, cam, g, count=4, steps=6, thickness=0.1)[..., 0]
    truth = irradiance_truth(cam, planes)[..., 0]
    pos = cam.ray_dirs() * g["linear_z"][..., None]
    on_floor = (g["normal"][..., 1] > 0.9) & (g["linear_z"] > 0)
    beside = (np.abs(pos[..., 0]) > 0.7) & (np.abs(pos[..., 0]) < 1.6)
    front = on_floor & beside & (pos[..., 2] > -3.0) & (pos[..., 2] < -2.0)
    behind = on_floor & beside & (pos[..., 2] > -5.5) & (pos[..., 2] < -4.2)
    assert front.sum() > 300 and behind.sum() > 300
    lit = float(truth[front].mean())
    assert float(irr[front].mean()) > 0.5 * lit                        # the lit face does bounce onto the floor in front
    leak = float(irr[behind].mean() - truth[behind].mean())
    assert leak < 0.25 * lit, (leak, lit)


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


def test_svgf_temporal_keeps_stored_depth_inside_fp16_range(gpu):
    from types import SimpleNamespace
    h, w = 8, 8
    far_cam = SimpleNamespace(depth_params=(10.0, 0.0))     # a depth of 1e-5 is 1e6 m away: past fp16's 65504
    g = {"depth": np.full((h, w), 1e-5), "normal": np.tile(np.array([0.5, 0.5, 1.0, 0.0]), (h, w, 1))}
    empty = {"hist": np.zeros((h, w, 4)), "mom": np.zeros((h, w, 2)), "geo": np.zeros((h, w, 4))}
    out = _temporal(gpu, far_cam, g, np.full((h, w, 4), 0.5), empty, flags=1)
    assert np.isfinite(out["geo"]).all()
    assert out["geo"][..., 3].max() <= 60000.0


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
    normal = g["normal"][..., :3] * 2.0 - 1.0
    geo = np.concatenate([normal / np.linalg.norm(normal, axis=-1, keepdims=True), g["linear_z"][..., None]], axis=-1)
    imgs = {"a": Img("rgba16f", data=data), "b": Img("rgba16f", w, h), "geo": Img("rgba16f", data=geo)}
    dispatches, src = [], "a"
    for it in range(iterations):
        dst = "b" if src == "a" else "a"
        dispatches.append(Dispatch(
            "lumina/lumina_svgf_atrous.comp.glsl",
            {0: (src, "sampled_nearest"), 1: ("geo", "sampled_nearest"), 2: (dst, "storage")},
            pc.atrous(w, h, 1 << it), groups_for(w, h)))
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
    for k in ("depth", "normal", "linear_z"):
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
