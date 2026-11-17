"""Track B visibility-buffer shaders on lavapipe: shared helpers, Hi-Z, meshlet culling, tile classification
and material shading. Expected values come from numpy, independently of the shaders' formulas."""

import struct

import numpy as np
import pytest

import track_b as tb
from harness import Buf, Dispatch, Img, groups_for

HELPERS = str(tb.HERE_SHADERS / "test_vis_helpers.comp.glsl")
MODE_PAYLOAD, MODE_CULL, MODE_BARYCENTRIC, MODE_NORMAL, MODE_HIZ = range(5)

PROJ = tb.godot_projection(70.0, 16 / 9, 0.05, 100.0)
VIEW = tb.look_at_view((0.0, 1.0, 3.0), (0.0, 0.0, -2.0))
VIEW_PROJ = PROJ @ VIEW
EYE = np.array([0.0, 1.0, 3.0])


def f4(*v) -> bytes:
    return struct.pack("<4f", *(list(v) + [0.0] * (4 - len(v))))


def run_helpers(gpu, mode, cases, images=None, pre=(), hiz="hiz", hiz_mips=1):
    images = dict(images or {})
    images.setdefault(hiz, Img("r32f", data=np.zeros((1, 1))))
    out = gpu.run(
        images=images,
        buffers={"in": Buf(data=b"".join(cases)), "out": Buf(size=64 * len(cases))},
        dispatches=list(pre) + [Dispatch(HELPERS, {0: ("in", "storage_buffer"), 1: ("out", "storage_buffer"), 2: (hiz, "sampled_nearest")},
                                         pc=struct.pack("<4I", mode, len(cases), hiz_mips, 0), groups=((len(cases) + 63) // 64, 1, 1))],
        outputs=["out"],
    )
    return out["out"]


def hiz_dispatches(depth, hiz, w, h):
    sizes = [(w, h)]
    while max(sizes[-1]) > 1:
        sizes.append((max(1, sizes[-1][0] // 2), max(1, sizes[-1][1] // 2)))
    shader = "visibility/vis_hi_z_build.comp.glsl"
    ds = [Dispatch(shader, {0: (depth, "sampled_nearest"), 1: (hiz, "storage", 0)},
                   pc=struct.pack("<4i2I", w, h, w, h, 0, 0), groups=groups_for(w, h))]
    for level in range(1, len(sizes)):
        (sw, sh), (dw, dh) = sizes[level - 1], sizes[level]
        ds.append(Dispatch(shader, {0: (hiz, "sampled_nearest"), 1: (hiz, "storage", level)},
                           pc=struct.pack("<4i2I", sw, sh, dw, dh, 1, level - 1), groups=groups_for(dw, dh)))
    return ds, sizes


# --------------------------------------------------------------------------- shared helpers

def test_payload_round_trip_and_empty_sentinel(gpu):
    cases = [(0, 0, 0, 0),                       # packed to (0, 0) before: indistinguishable from "empty"
             (1, 2, 3, 1),
             (0xFFFFFFFE, 0xFFFFFF, 123, 1),     # largest instance / meshlet / primitive
             (7, 4096, 5, 0),                    # meshlet IDs past the old 12-bit limit
             (123456, 777777, 64, 0)]
    raw = run_helpers(gpu, MODE_PAYLOAD, [struct.pack("<4I", *c) + bytes(48) for c in cases])
    for i, (inst, meshlet, prim, alpha) in enumerate(cases):
        out = struct.unpack_from("<8I", raw, 64 * i)
        assert out[:2] == tb.payload(inst, meshlet, prim, bool(alpha))
        assert out[2] == 0, f"{cases[i]} reads as empty"
        assert out[3] == 1, "PRISM_VIS_PAYLOAD_EMPTY must read as empty"
        assert out[4:8] == (inst, meshlet, prim, alpha)


def random_rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def test_triangle_cull_keeps_godot_front_faces(gpu):
    """Front-facing (clockwise on screen) triangles must survive under Godot's Y-flipped projection, back faces
    must not, mirrored instances swap the two, two-sided keeps both."""
    rng = np.random.default_rng(1)
    cases, expected = [], []
    for i in range(240):
        m3 = random_rotation(rng) * rng.uniform(0.5, 2.0)
        if i % 3 == 1:
            m3 = m3 @ np.diag([-1.0, 1.0, 1.0])            # mirrored
        elif i % 3 == 2:
            m3 = m3 @ np.diag([1.0, rng.uniform(0.3, 3.0), 1.0])
        center = np.array([rng.uniform(-1.5, 1.5), rng.uniform(-0.5, 1.5), rng.uniform(-6.0, -2.0)])
        local = rng.normal(size=(3, 3)) * 0.4
        world = [center + m3 @ p for p in local]
        clips = [tb.project(VIEW_PROJ, p) for p in world]
        if min(c[3] for c in clips) < 0.1:
            continue
        ndc = [c[:2] / c[3] for c in clips]
        # Godot's front face: clockwise winding, outward normal cross(p2 - p0, p1 - p0) in local space
        outward = np.linalg.inv(m3).T @ np.cross(local[2] - local[0], local[1] - local[0])
        front = np.dot(outward, EYE - world[0]) > 0
        mirrored = np.linalg.det(m3) < 0
        for two_sided in (False, True):
            cases.append(f4(*ndc[0], *ndc[1]) + f4(*ndc[2], float(two_sided), float(mirrored)) + bytes(32))
            expected.append((not two_sided) and (not front))
    degenerate = [np.array([0.1, 0.1]), np.array([0.2, 0.2]), np.array([0.3, 0.3])]
    cases.append(f4(*degenerate[0], *degenerate[1]) + f4(*degenerate[2], 1.0, 0.0) + bytes(32))
    expected.append(True)

    raw = run_helpers(gpu, MODE_CULL, cases)
    culled = [struct.unpack_from("<f", raw, 64 * i)[0] == 1.0 for i in range(len(cases))]
    assert culled == expected
    assert sum(expected) > 50 and len(expected) - sum(expected) > 50  # both outcomes exercised


def ray_bary(tri, ndc, inv_view_proj):
    """Barycentrics of the point on the triangle's plane seen through `ndc` (ray-plane intersection)."""
    a = inv_view_proj @ np.array([*ndc, 1.0, 1.0])
    b = inv_view_proj @ np.array([*ndc, 0.5, 1.0])
    a, b = a[:3] / a[3], b[:3] / b[3]
    n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    t = np.dot(tri[0] - a, n) / np.dot(b - a, n)
    q = a + t * (b - a)
    m = np.column_stack([tri[1] - tri[0], tri[2] - tri[0]])
    l12 = np.linalg.lstsq(m, q - tri[0], rcond=None)[0]
    return np.array([1 - l12.sum(), l12[0], l12[1]])


def test_barycentrics_exact_with_corners_behind_camera(gpu):
    rng = np.random.default_rng(2)
    width, height = 1920, 1080
    inv_vp = np.linalg.inv(VIEW_PROJ)
    cases, expect = [], []
    behind = 0
    while len(cases) < 120:
        # A visible point Q with known barycentrics, on a big triangle that may reach behind the camera
        q = EYE + (np.linalg.inv(VIEW[:3, :3]) @ np.array([rng.uniform(-1, 1), rng.uniform(-0.5, 0.5), -1.0])) * rng.uniform(1.0, 6.0)
        lam = rng.dirichlet([1.0, 1.0, 1.0])
        v0, v1 = q + rng.normal(size=3) * 6.0, q + rng.normal(size=3) * 6.0
        v2 = (q - lam[0] * v0 - lam[1] * v1) / lam[2]
        tri = [v0, v1, v2]
        clips = [tb.project(VIEW_PROJ, p) for p in tri]
        if min(lam) < 0.05 or np.linalg.norm(v2 - q) > 60:
            continue
        behind += any(c[3] < 0 for c in clips)
        clip_q = tb.project(VIEW_PROJ, q)
        ndc = clip_q[:2] / clip_q[3]
        h = 0.01
        step = np.array([2.0 / width, 2.0 / height]) * h
        ddx = (ray_bary(tri, ndc + [step[0], 0], inv_vp) - ray_bary(tri, ndc - [step[0], 0], inv_vp)) / (2 * h)
        ddy = (ray_bary(tri, ndc + [0, step[1]], inv_vp) - ray_bary(tri, ndc - [0, step[1]], inv_vp)) / (2 * h)
        cases.append(f4(*clips[0]) + f4(*clips[1]) + f4(*clips[2]) + f4(*ndc, 1.0 / width, 1.0 / height))
        expect.append((lam, ddx, ddy))
    assert behind >= 20, "need triangles with corners behind the camera"

    raw = run_helpers(gpu, MODE_BARYCENTRIC, cases)
    out = np.frombuffer(raw, dtype=np.float32).reshape(-1, 4, 4)
    for (lam, ddx, ddy), o in zip(expect, out):
        assert o[0, 3] == 1.0
        np.testing.assert_allclose(o[0, :3], lam, atol=2e-4)
        scale = max(np.abs(ddx).max(), np.abs(ddy).max(), 1e-6)
        np.testing.assert_allclose(o[1, :3], ddx, atol=2e-3 * scale + 1e-7)
        np.testing.assert_allclose(o[2, :3], ddy, atol=2e-3 * scale + 1e-7)


def test_barycentrics_report_edge_on_triangles(gpu):
    # The triangle's plane contains the eye: every pixel sees it edge-on.
    tri = [EYE + np.array([0.0, 0.0, -1.0]), EYE + np.array([0.0, 0.0, -3.0]), EYE + np.array([0.0, 1.0, -2.0])]
    clips = [tb.project(VIEW_PROJ, p) for p in tri]
    raw = run_helpers(gpu, MODE_BARYCENTRIC, [f4(*clips[0]) + f4(*clips[1]) + f4(*clips[2]) + f4(0.1, 0.2, 1 / 1920, 1 / 1080)])
    assert struct.unpack_from("<4f", raw, 0)[3] == 0.0


def test_normal_matrix_matches_inverse_transpose(gpu):
    rng = np.random.default_rng(3)
    cases, expected = [], []
    for i in range(64):
        m3 = rng.normal(size=(3, 3))
        if i % 2:
            m3 = m3 @ np.diag([-1.0, 1.0, 1.0]) if np.linalg.det(m3) > 0 else m3
        n = rng.normal(size=3)
        cases.append(f4(*m3[:, 0]) + f4(*m3[:, 1]) + f4(*m3[:, 2]) + f4(*n))
        e = np.linalg.inv(m3).T @ n
        expected.append(e / np.linalg.norm(e))
    out = np.frombuffer(run_helpers(gpu, MODE_NORMAL, cases), dtype=np.float32).reshape(-1, 4, 4)
    np.testing.assert_allclose(out[:, 0, :3], np.array(expected), atol=1e-4)


# --------------------------------------------------------------------------- Hi-Z

def hiz_test_depth(rng, w, h):
    depth = np.full((h, w), 0.8) + rng.uniform(-0.05, 0.05, size=(h, w))   # a near wall...
    holes = rng.random((h, w)) < 0.01                                     # ...with far pixels poking through
    holes[:, -3:] |= rng.random((h, 3)) < 0.15                            # including the right and bottom edges
    holes[-3:, :] |= rng.random((3, w)) < 0.15
    depth[holes] = 0.1
    return depth.astype(np.float32)


@pytest.mark.parametrize("size", [(45, 27), (100, 61), (64, 64)])
def test_hiz_pyramid_covers_every_pixel(gpu, size):
    w, h = size
    depth = hiz_test_depth(np.random.default_rng(w * h), w, h)
    dispatches, sizes = hiz_dispatches("depth", "hiz", w, h)
    out = gpu.run(images={"depth": Img("r32f", data=depth), "hiz": Img("r32f", w, h, mips=len(sizes))},
                  dispatches=dispatches, outputs=["hiz"])["hiz"]
    for level, (lw, lh) in enumerate(sizes):
        got = out[level][..., 0]
        assert got.shape == (lh, lw)
        for ty in range(lh):
            y0, y1 = ty << level, (h if ty == lh - 1 else (ty + 1) << level)
            for tx in range(lw):
                x0, x1 = tx << level, (w if tx == lw - 1 else (tx + 1) << level)
                assert got[ty, tx] == depth[y0:y1, x0:x1].min(), (level, tx, ty)


def test_hiz_occlusion_is_conservative(gpu):
    w, h = 100, 61
    rng = np.random.default_rng(4)
    depth = hiz_test_depth(rng, w, h)
    dispatches, sizes = hiz_dispatches("depth", "hiz", w, h)

    rects, nearest = [], []
    for _ in range(3000):
        size_uv = rng.uniform(0.0, 0.4, size=2) ** 2 * 2
        lo = rng.uniform(0.0, 1.0, size=2)
        rects.append(np.concatenate([lo, np.minimum(lo + size_uv, 1.0)]).astype(np.float32))
        nearest.append(np.float32(rng.uniform(0.3, 0.75)))
    cases = [f4(*r) + f4(z) + bytes(32) for r, z in zip(rects, nearest)]
    raw = run_helpers(gpu, MODE_HIZ, cases, images={"depth": Img("r32f", data=depth), "hiz": Img("r32f", w, h, mips=len(sizes))},
                      pre=dispatches, hiz_mips=len(sizes))
    occluded = [struct.unpack_from("<f", raw, 64 * i)[0] == 1.0 for i in range(len(cases))]

    size0 = np.array([w, h], dtype=np.float32)
    truly_hidden = 0
    for r, z, occ in zip(rects, nearest, occluded):
        p_min = np.clip(np.floor(r[:2] * size0), 0, size0 - 1).astype(int)
        p_max = np.clip(np.floor(r[2:] * size0), 0, size0 - 1).astype(int)
        true_far = depth[p_min[1]:p_max[1] + 1, p_min[0]:p_max[0] + 1].min()
        if occ:
            assert z < true_far, f"rect {r} reported occluded but a pixel at depth {true_far} is behind {z}"
        truly_hidden += z < true_far
    # ...and it still culls. Far "holes" make big footprints unprovable (a level-4 footprint spans up to
    # 1024 pixels), so only part of the truly hidden rects can be culled; ~27% with this depth map.
    assert sum(occluded) > 200 and truly_hidden > sum(occluded)


# --------------------------------------------------------------------------- meshlet culling

CULL_VIEW = tb.look_at_view((0.0, 0.0, 5.0), (0.0, 0.0, 0.0))
CULL_VIEW_PROJ = tb.godot_projection(70.0, 1.0, 0.1, 100.0) @ CULL_VIEW


def cull_scene(instances):
    """instances: list of (world, flags, meshlet kwargs). One meshlet per instance."""
    meshlets = b"".join(tb.gpu_meshlet(0, 0, 3, 10 + i, (0, 0, 0), 0.5, **kw) for i, (_, _, kw) in enumerate(instances))
    inst = b"".join(tb.gpu_instance(world, i, 1, 0, 0, flags) for i, (world, flags, _) in enumerate(instances))
    candidates = b"".join(tb.u32s(i, 0) for i in range(len(instances)))
    return meshlets, inst, candidates


def run_cull(gpu, instances, phase, hiz_depth, max_visible=16, max_fn=16, hiz=1, counters=(0, 0, 0), false_negatives=b""):
    meshlets, inst, candidates = cull_scene(instances)
    ubo = tb.cull_ubo(CULL_VIEW_PROJ, CULL_VIEW_PROJ, (0, 0, 5), 0.1, tb.frustum_planes(CULL_VIEW_PROJ), 0, len(instances),
                      max_visible, max_fn, phase, hiz=hiz)
    dispatches, sizes = hiz_dispatches("depth", "hiz", 64, 64)
    ubo = ubo[:240] + struct.pack("<I", len(sizes)) + ubo[244:]
    groups = ((len(instances) + 63) // 64, 1, 1)
    out = gpu.run(
        images={"depth": Img("r32f", data=np.full((64, 64), hiz_depth)), "hiz": Img("r32f", 64, 64, mips=len(sizes))},
        buffers={"ubo": Buf(data=ubo), "meshlets": Buf(data=meshlets), "instances": Buf(data=inst), "candidates": Buf(data=candidates),
                 "fn": Buf(data=false_negatives, size=8 * max_fn), "visible": Buf(size=16 * max_visible),
                 "draws": Buf(size=32 * max_visible), "counters": Buf(data=tb.u32s(*counters, 0)),
                 "tasks": Buf(data=tb.u32s(0, 1, 1, 0, 1, 1))},
        dispatches=dispatches + [Dispatch("visibility/vis_cull_meshlets.comp.glsl", {
            0: ("ubo", "uniform_buffer"), 1: ("meshlets", "storage_buffer"), 2: ("instances", "storage_buffer"),
            3: ("candidates", "storage_buffer"), 4: ("fn", "storage_buffer"), 5: ("visible", "storage_buffer"),
            6: ("draws", "storage_buffer"), 7: ("counters", "storage_buffer"), 8: ("tasks", "storage_buffer"),
            9: ("hiz", "sampled_nearest")}, groups=groups)],
        outputs=["fn", "visible", "draws", "counters", "tasks"])
    counters = struct.unpack("<3I", out["counters"][:12])
    lists = {}
    for name, list_index, count in (("opaque", 0, counters[0]), ("masked", 1, counters[2])):
        entries = []
        for slot in range(min(count, max_visible)):
            index = list_index * max_visible + slot
            vis = struct.unpack_from("<2I", out["visible"], 8 * index)
            draw = struct.unpack_from("<4I", out["draws"], 16 * index)
            assert draw == (3 * (10 + vis[0]), 1, 0, index)
            entries.append(vis[0])
        lists[name] = sorted(entries)
    tasks = struct.unpack("<6I", out["tasks"])
    fn = [struct.unpack_from("<2I", out["fn"], 8 * i)[0] for i in range(min(counters[1], max_fn))]
    return counters, lists, tasks, sorted(fn), out["fn"]


def test_cull_frustum_cone_and_lists(gpu):
    away_cone = {"cone_apex": (0, 0, 0), "cone_axis": (0, 0, -1), "cone_cutoff": 0.5}   # every face points away from +Z
    instances = [
        (tb.translate(0, 0, 0), tb.FLAG_VISIBLE, {}),                                          # 0 visible
        (tb.translate(1, 0, 0), tb.FLAG_VISIBLE | tb.FLAG_ALPHA_TEST, {}),                     # 1 masked list
        (tb.translate(100, 0, 0), tb.FLAG_VISIBLE, {}),                                        # 2 outside the frustum
        (tb.translate(0, 0, 0), 0, {}),                                                        # 3 hidden instance
        (tb.translate(-1, 0, 0), tb.FLAG_VISIBLE, away_cone),                                  # 4 back-facing cluster
        (tb.translate(-1, 1, 0) @ tb.scale(2, 1, 1), tb.FLAG_VISIBLE, away_cone),              # 5 non-uniform scale: no cone test
        (tb.translate(-1, -1, 0), tb.FLAG_VISIBLE | tb.FLAG_TWO_SIDED, away_cone),             # 6 two-sided: no cone test
        (tb.translate(1, 1, 0) @ tb.scale(-1, 1, 1), tb.FLAG_VISIBLE, away_cone),              # 7 mirrored, still back-facing
    ]
    counters, lists, tasks, _, _ = run_cull(gpu, instances, 0, 0.0, hiz=0)
    assert lists == {"opaque": [0, 5, 6], "masked": [1]}
    assert counters == (3, 0, 1)
    assert tasks == (3, 1, 1, 1, 1, 1)


def test_cull_false_negative_overflow_draws_instead_of_dropping(gpu):
    instances = [(tb.translate(x, 0, 0), tb.FLAG_VISIBLE, {}) for x in (-1.5, -0.75, 0.0, 0.75, 1.5)]
    # Last frame's depth is the nearest possible everywhere: every meshlet tests as occluded in Phase 1.
    counters, lists, _, fn, fn_raw = run_cull(gpu, instances, 0, 1.0, max_fn=2)
    assert counters[1] == 5                       # all five rejected by Hi-Z
    assert len(fn) == 2                           # two fit the queue...
    assert sorted(fn + lists["opaque"]) == [0, 1, 2, 3, 4]   # ...and the other three are drawn, not lost

    # Phase 2 against an empty (far) depth buffer: the queued two come back.
    counters2, lists2, tasks2, _, _ = run_cull(gpu, instances, 1, 0.0, max_fn=2, counters=(0, 2, 0), false_negatives=fn_raw)
    assert lists2["opaque"] == fn
    assert tasks2[0] == 2


def test_cull_phase1_queues_everything_hidden(gpu):
    instances = [(tb.translate(x, 0, 0), tb.FLAG_VISIBLE, {}) for x in (-1.0, 0.0, 1.0)]
    counters, lists, tasks, fn, _ = run_cull(gpu, instances, 0, 1.0)
    assert fn == [0, 1, 2] and lists["opaque"] == [] and tasks[0] == 0
    counters, lists, _, fn, _ = run_cull(gpu, instances, 0, 0.0)   # nothing in front: all visible
    assert lists["opaque"] == [0, 1, 2] and fn == []


# --------------------------------------------------------------------------- tile classification

def classify(gpu, vis, instances_materials, max_items):
    h, w = vis.shape[:2]
    inst = b"".join(tb.gpu_instance(np.eye(4), 0, 1, 0, m, tb.FLAG_VISIBLE) for m in instances_materials)
    ubo = tb.shade_ubo(VIEW_PROJ, VIEW_PROJ, EYE, (0, 1, 0), 1.0, (1, 1, 1), 0.0, w, h, max_items)
    out = gpu.run(
        images={"vis": Img("rg32ui", data=vis)},
        buffers={"ubo": Buf(data=ubo), "instances": Buf(data=inst), "items": Buf(size=8 * max_items),
                 "dispatch": Buf(data=tb.u32s(256, 0, 1, 0))},
        dispatches=[Dispatch("visibility/vis_tile_classify.comp.glsl", {
            0: ("ubo", "uniform_buffer"), 1: ("vis", "sampled_nearest"), 2: ("instances", "storage_buffer"),
            3: ("items", "storage_buffer"), 4: ("dispatch", "storage_buffer")}, groups=groups_for(w, h))],
        outputs=["items", "dispatch"])
    dispatch = struct.unpack("<4I", out["dispatch"])
    items = [struct.unpack_from("<2I", out["items"], 8 * i) for i in range(min(dispatch[3], max_items))]
    return dispatch, items


def test_classify_bins_materials_and_sizes_the_2d_dispatch(gpu):
    vis = np.full((12, 20, 2), tb.EMPTY, dtype=np.uint64)
    vis[1, 1] = tb.payload(0, 0, 0)     # (0, 0): the old "background" value, a real pixel now
    vis[2, 2] = tb.payload(1, 5, 7)
    vis[1, 9] = tb.payload(2, 0, 1)     # material 300 -> last bin
    vis[1, 10] = tb.payload(3, 0, 1)    # material 255 -> same bin
    vis[9, 17] = tb.payload(1, 0, 2)
    materials = [0, 3, 300, 255]
    dispatch, items = classify(gpu, vis, materials, max_items=64)
    tiles = {}
    for packed, bin_ in items:
        tiles.setdefault((packed & 0xFFFF, packed >> 16), set()).add(bin_)
    assert tiles == {(0, 0): {0, 3}, (1, 0): {255}, (2, 1): {3}}
    assert dispatch == (256, 1, 1, 4)

    dispatch, items = classify(gpu, vis, materials, max_items=2)   # more work than room
    assert dispatch == (256, 1, 1, 4)                               # count is honest, rows cover what's stored
    assert len(items) == 2


# --------------------------------------------------------------------------- material shading

def quad(x0, x1, y0, y1, z, clockwise=True, normal=(0, 0, 1)):
    """Quad facing +Z; clockwise = Godot front face as seen from +Z. UV (0,0) at top-left."""
    pos = [(x0, y1, z), (x1, y1, z), (x1, y0, z), (x0, y0, z)]
    uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
    tris = [(0, 1, 2), (0, 2, 3)] if clockwise else [(0, 2, 1), (0, 3, 2)]
    return pos, uvs, tris, normal


def test_shade_materials_end_to_end(gpu):
    w = h = 32
    proj = tb.godot_projection(70.0, 1.0, 0.1, 50.0)
    view = tb.look_at_view((0, 0, 3), (0, 0, 0))
    view_proj = proj @ view
    quads = [quad(-1.8, -0.7, -0.5, 0.5, 0.0),                                       # 0: unshaded, checker texture
             quad(-0.5, 0.5, -0.5, 0.5, 0.0),                                        # 1: shaded, front face
             quad(0.7, 1.8, -0.5, 0.5, 0.0, clockwise=False, normal=(0, 0, -1))]     # 2: back face, double-sided
    materials = [tb.gpu_material(tex=(0, tb.INVALID_TEX, tb.INVALID_TEX, tb.INVALID_TEX), flags=4),
                 tb.gpu_material(base_color=(0.5, 0.5, 0.5, 1)),
                 tb.gpu_material(base_color=(0.5, 0.5, 0.5, 1), flags=1)]

    positions, attrs, meshlet_verts, tris_packed, meshlets, instances = b"", b"", b"", b"", b"", b""
    world_tris = []
    for qi, (pos, uvs, tris, normal) in enumerate(quads):
        positions += b"".join(struct.pack("<4f", *p, 1.0) for p in pos)
        attrs += b"".join(tb.gpu_vertex_attributes(normal, (1, 0, 0, 1), uv) for uv in uvs)
        meshlet_verts += tb.u32s(0, 1, 2, 3)
        tris_packed += tb.pack_triangles(tris)
        meshlets += tb.gpu_meshlet(4 * qi, 2 * qi, 4, 2, (0, 0, 0), 2.0)
        instances += tb.gpu_instance(np.eye(4), qi, 1, 4 * qi, qi, tb.FLAG_VISIBLE)
        world_tris += [(qi, t, [np.array(pos[k], float) for k in tris[t]], [np.array(uvs[k], float) for k in tris[t]]) for t in range(2)]

    # Rasterize the visibility buffer in numpy (pixel centers inside a triangle's projection)
    vis = np.full((h, w, 2), tb.EMPTY, dtype=np.uint64)
    expected_uv = {}
    inv_vp = np.linalg.inv(view_proj)
    for y in range(h):
        for x in range(w):
            p = np.array([(x + 0.5) / w * 2 - 1, (y + 0.5) / h * 2 - 1])
            for qi, t, corners, uvs in world_tris:
                c = [tb.project(view_proj, v) for v in corners]
                n = [ci[:2] / ci[3] for ci in c]
                e = [(n[(k + 1) % 3][0] - n[k][0]) * (p[1] - n[k][1]) - (n[(k + 1) % 3][1] - n[k][1]) * (p[0] - n[k][0])
                     for k in range(3)]
                if all(v >= 0 for v in e) or all(v <= 0 for v in e):
                    vis[y, x] = tb.payload(qi, 0, t)
                    b = ray_bary(corners, p, inv_vp)
                    expected_uv[(x, y)] = (qi, b @ np.array(uvs))
                    break

    checker = np.array([[[1, 0, 0, 1], [0, 1, 0, 1]], [[0, 0, 1, 1], [1, 1, 1, 1]]], float)
    ubo = tb.shade_ubo(view_proj, view_proj, (0, 0, 3), (0, 0, 1), 1.0, (1, 1, 1), 0.0, w, h, 64)
    bufs = {"ubo": Buf(data=ubo), "meshlets": Buf(data=meshlets), "instances": Buf(data=instances), "positions": Buf(data=positions),
            "attrs": Buf(data=attrs), "mverts": Buf(data=meshlet_verts), "mtris": Buf(data=tris_packed), "materials": Buf(data=b"".join(materials)),
            "items": Buf(size=8 * 64), "dispatch": Buf(data=tb.u32s(256, 0, 1, 0))}
    images = {"vis": Img("rg32ui", data=vis), "albedo": Img("rgba8", data=checker), "hdr": Img("rgba16f", w, h),
              "albedo_rough": Img("rgba8", w, h), "normal": Img("rgba16f", w, h), "motion": Img("rg16f", w, h)}
    out = gpu.run(images=images, buffers=bufs, outputs=["hdr", "normal", "motion"], dispatches=[
        Dispatch("visibility/vis_tile_classify.comp.glsl", {0: ("ubo", "uniform_buffer"), 1: ("vis", "sampled_nearest"),
                 2: ("instances", "storage_buffer"), 3: ("items", "storage_buffer"), 4: ("dispatch", "storage_buffer")},
                 groups=groups_for(w, h)),
        Dispatch("visibility/vis_shade_materials.comp.glsl", {
            0: ("ubo", "uniform_buffer"), 1: ("vis", "sampled_nearest"), 2: ("meshlets", "storage_buffer"),
            3: ("instances", "storage_buffer"), 4: ("positions", "storage_buffer"), 5: ("attrs", "storage_buffer"),
            6: ("mverts", "storage_buffer"), 7: ("mtris", "storage_buffer"), 8: ("materials", "storage_buffer"),
            9: ("items", "storage_buffer"), 10: (["albedo"], "sampled_nearest"), 11: ("hdr", "storage"),
            12: ("albedo_rough", "storage"), 13: ("normal", "storage"), 14: ("motion", "storage"),
            15: ("dispatch", "storage_buffer")}, indirect=("dispatch", 0))])

    def oct_decode(e):
        v = np.array([e[0], e[1], 1.0 - abs(e[0]) - abs(e[1])])
        if v[2] < 0:
            v[:2] = (1.0 - np.abs(v[[1, 0]])) * np.where(v[:2] >= 0, 1.0, -1.0)
        return v / np.linalg.norm(v)

    seen = {0: 0, 1: 0, 2: 0}
    for y in range(h):
        for x in range(w):
            if (x, y) not in expected_uv:
                assert not out["hdr"][y, x].any(), "background pixel was written"
                continue
            qi, uv = expected_uv[(x, y)]
            seen[qi] += 1
            if qi == 0:
                if np.any(np.abs(uv - 0.5) < 0.03):
                    continue  # checker edge: nearest-texel choice is a coin toss
                texel = checker[int(uv[1] * 2), int(uv[0] * 2)]
                np.testing.assert_allclose(out["hdr"][y, x, :3], texel[:3], atol=2e-3, err_msg=f"pixel {x},{y} uv {uv}")
            else:
                n = oct_decode(out["normal"][y, x, :2].astype(float) * 2 - 1)
                np.testing.assert_allclose(n, [0, 0, 1], atol=5e-3, err_msg=f"quad {qi} pixel {x},{y}")
            assert not out["motion"][y, x].any()
    assert min(seen.values()) > 20
