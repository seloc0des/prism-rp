"""Lumina world-space radiance cache shaders on lavapipe."""

import numpy as np
import pytest

import pc
from harness import Dispatch, Img, groups_for
from scene import Camera, Plane, render

WORLD_BLOCK = 8
N = 4
SPACING = 1.0
RAY_LENGTH = 8.0


def oct_decode(e):
    v = np.array([e[0], e[1], 1.0 - abs(e[0]) - abs(e[1])])
    if v[2] < 0:
        v[:2] = (1.0 - np.abs(v[[1, 0]])) * np.where(v[:2] >= 0, 1.0, -1.0)
    return v / np.linalg.norm(v)


def oct_encode(n):
    n = np.asarray(n, float) / np.abs(n).sum()
    e = n[:2].copy()
    if n[2] < 0:
        e = (1.0 - np.abs(n[[1, 0]])) * np.where(n[:2] >= 0, 1.0, -1.0)
    return e


def dir_texel(d, block=WORLD_BLOCK):
    e = oct_encode(d) * 0.5 + 0.5
    return tuple(np.clip((e * block).astype(int), 0, block - 1))


def atlas_texel(slot, dt):
    return ((slot[0] + slot[2] * N) * WORLD_BLOCK + dt[0], slot[1] * WORLD_BLOCK + dt[1])


def base_cell_for(cam_pos):
    return tuple(int(np.floor(c / SPACING)) - N // 2 for c in cam_pos)


def world_pc(cam, cam_pos, right, up, base, a1w=0, screen_block=0, misc_z=RAY_LENGTH, steps=32.0):
    return pc.world((cam.width, cam.height), N, screen_block or WORLD_BLOCK, base, a1w, cam_pos, SPACING,
                    right, 0.25, up, 0.1, cam.proj, cam.depth_params, misc_z, steps)


def classify_and_update(gpu, cam, g, cam_pos, right, up, state=None, probes=None, frame=0):
    base = base_cell_for(cam_pos)
    state = np.zeros((N, N * N, 4)) if state is None else state
    probes = np.zeros((N * WORLD_BLOCK, N * N * WORLD_BLOCK, 4)) if probes is None else probes
    push = world_pc(cam, cam_pos, right, up, base, a1w=frame)
    out = gpu.run(
        {"depth": Img("r32f", data=g["depth"]), "color": Img("rgba16f", data=g["color"]),
         "state": Img("rgba32f", data=state), "probes": Img("rgba16f", data=probes)},
        [Dispatch("lumina/lumina_ws_classify.comp.glsl", {0: ("depth", "sampled_nearest"), 1: ("state", "storage")}, push, (1, 1, 1)),
         Dispatch("lumina/lumina_ws_update.comp.glsl",
                  {0: ("depth", "sampled_nearest"), 1: ("color", "sampled_linear"), 2: ("state", "sampled_nearest"), 3: ("probes", "storage")},
                  push, groups_for(N * N * WORLD_BLOCK, N * WORLD_BLOCK))],
        ["state", "probes"])
    return out["state"], out["probes"], base


FORWARD = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))    # camera looking down world -Z
BACKWARD = ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0))  # turned 180 degrees: looking down world +Z


def red_wall_scene(cam):
    return render(cam, [Plane((0, 0, -6), (0, 0, 1), (2.0, 0.0, 0.0))])


def test_classify_marks_new_cells_then_scrolls(gpu):
    cam = Camera(width=32, height=24)
    g = red_wall_scene(cam)
    state, _, _ = classify_and_update(gpu, cam, g, (0.2, 0.3, 0.1), *FORWARD)
    assert np.all(state[..., 1] == 1.0)                      # first frame: every slot is new
    state2, _, _ = classify_and_update(gpu, cam, g, (0.2, 0.3, 0.1), *FORWARD, state=state)
    assert np.all(state2[..., 1] == 0.0)                     # same grid: nothing changed
    state3, _, _ = classify_and_update(gpu, cam, g, (1.2, 0.3, 0.1), *FORWARD, state=state2)
    assert state3[..., 1].sum() == N * N                     # moved one cell in +X: one slice of slots changed


def test_update_picks_up_visible_wall_and_skips_rays_behind_camera(gpu):
    cam = Camera(width=48, height=36)
    g = red_wall_scene(cam)
    cam_pos = (0.0, 0.0, 0.0)
    _, probes, base = classify_and_update(gpu, cam, g, cam_pos, *FORWARD)
    # A probe in front of the camera (cell z = -2 -> centre z = -1.5), looking at the wall (-Z).
    cell = (0, 0, -2)
    slot = tuple(np.mod(np.array(cell), N))
    fwd = probes[atlas_texel(slot, dir_texel((0, 0, -1)))[::-1]]
    assert fwd[0] == pytest.approx(2.0, rel=0.02) and fwd[3] == 1.0
    back = probes[atlas_texel(slot, dir_texel((0, 0, 1)))[::-1]]
    assert back[3] == 0.0                                    # toward/behind the camera: no information


def test_probe_just_behind_a_wall_records_nothing(gpu):
    # A probe whose centre sits a little behind a visible wall (0.2 m, inside the old half-cell margin) can't
    # see anything from there: every ray starts hidden behind the wall. Its first step counted as a hit on the
    # wall's lit front, so it took the wall's colour in every direction and lit its surroundings like a lamp
    # (the streaks on Sponza's columns with the cache on).
    cam = Camera(width=64, height=48)
    g = render(cam, [Plane((0, 0, -1.3), (0, 0, 1), (2.0, 0.0, 0.0))])
    _, probes, base = classify_and_update(gpu, cam, g, (0.0, 0.0, 0.0), *FORWARD)
    cell = (0, 0, -2)                                       # centre (0.5, 0.5, -1.5): 0.2 m behind the wall
    slot = tuple(c % N for c in cell)
    x0, y0 = atlas_texel(slot, (0, 0))
    block = probes[y0:y0 + WORLD_BLOCK, x0:x0 + WORLD_BLOCK]
    assert block[..., 0].max() < 0.1, f"{(block[..., 0] > 0.1).sum()} of {WORLD_BLOCK ** 2} directions took the wall's red"

def test_cache_remembers_light_after_turning_away(gpu):
    cam = Camera(width=48, height=36)
    g = red_wall_scene(cam)
    cam_pos = (0.0, 0.0, 0.0)
    state, probes, _ = classify_and_update(gpu, cam, g, cam_pos, *FORWARD)
    # Turn around: the wall is now behind the camera, the view shows only sky.
    sky = render(cam, [])
    _, probes2, _ = classify_and_update(gpu, cam, sky, cam_pos, *BACKWARD, state=state, probes=probes)
    cell = (0, 0, -2)
    slot = tuple(np.mod(np.array(cell), N))
    remembered = probes2[atlas_texel(slot, dir_texel((0, 0, -1)))[::-1]]
    assert remembered[0] == pytest.approx(2.0, rel=0.02)


def test_scrolled_in_cells_start_empty(gpu):
    cam = Camera(width=48, height=36)
    g = red_wall_scene(cam)
    state, probes, _ = classify_and_update(gpu, cam, g, (0.0, 0.0, 0.0), *FORWARD)
    # Move far enough that every slot holds a new cell, and look at sky: nothing to sample.
    sky = render(cam, [])
    _, probes2, _ = classify_and_update(gpu, cam, sky, (40.0, 0.0, 0.0), *BACKWARD, state=state, probes=probes)
    # No stale red from the old cells, and no history carried over: at most this frame's one sample
    # (an all-sky view still records "nothing there" for rays that stay on screen).
    assert np.all(probes2[..., :3] == 0.0)
    assert probes2[..., 3].max() <= 1.0


def _ws_merge(gpu, cam, g, interval, probes, state, cam_pos, right, up, screen_block, spacing_px, flags=0, normal=None):
    base = base_cell_for(cam_pos)
    push = world_pc(cam, cam_pos, right, up, base, a1w=flags, screen_block=screen_block, misc_z=float(spacing_px))
    return gpu.run(
        {"depth": Img("r32f", data=g["depth"]), "probes": Img("rgba16f", data=probes), "state": Img("rgba32f", data=state),
         "lvl": Img("rgba16f", data=interval), "normal": Img("rgba8", data=g["normal"] if normal is None else normal)},
        [Dispatch("lumina/lumina_ws_merge.comp.glsl",
                  {0: ("depth", "sampled_nearest"), 1: ("probes", "sampled_nearest"), 2: ("state", "sampled_nearest"), 3: ("lvl", "storage"),
                   4: ("normal", "sampled_nearest")},
                  push, groups_for(interval.shape[1], interval.shape[0]))], ["lvl"])["lvl"]


def test_ws_merge_adds_far_field_through_open_intervals(gpu):
    cam = Camera(width=32, height=24)
    g = render(cam, [Plane((0, 0, -1.2), (0, 0, 1), (0.1, 0.1, 0.1))])   # surface inside the grid
    spacing_px, block = 16, 8
    probes_x, probes_y = 2, 2
    interval = np.zeros((probes_y * block, probes_x * block, 4))
    interval[..., 3] = 1.0                                   # open: everything reaches the far field
    interval[:, : probes_x * block // 2] = (0.3, 0.0, 0.0, 0.0)   # left probes blocked
    W = np.array([0.1, 0.5, 0.9])
    probes = np.zeros((N * WORLD_BLOCK, N * N * WORLD_BLOCK, 4)); probes[..., :3] = W; probes[..., 3] = 1.0
    state = np.zeros((N, N * N, 4))
    out = _ws_merge(gpu, cam, g, interval, probes, state, (0.0, 0.0, 0.0), *FORWARD, block, spacing_px)
    right_half = out[:, probes_x * block // 2:]
    np.testing.assert_allclose(right_half[..., :3].reshape(-1, 3).mean(0), W, rtol=0.02)
    assert np.all(right_half[..., 3] < 0.05)                 # far field fully covered
    np.testing.assert_allclose(out[:, : probes_x * block // 2], interval[:, : probes_x * block // 2], atol=1e-3)

    only_world = _ws_merge(gpu, cam, g, interval, probes, state, (0.0, 0.0, 0.0), *FORWARD, block, spacing_px, flags=1)
    np.testing.assert_allclose(only_world[..., :3].reshape(-1, 3).mean(0), W, rtol=0.02)


def test_ws_merge_ignores_cells_behind_the_probes_surface(gpu):
    # A probe on a wall blends the 8 cache cells around it. Cells on the far side of the wall see a different
    # space (outside a room, inside a pillar) and classify can't always mark them buried: from the camera they
    # can line up with empty background. Blending them in let a wall light itself (the Cornell box came out
    # nearly twice the reference with the cache on). Only cells on the side the surface faces may count.
    cam = Camera(width=32, height=24)
    g = render(cam, [Plane((0, 0, -1.2), (0, 0, 1), (0.1, 0.1, 0.1))])   # wall facing the camera, inside the grid
    spacing_px, block = 16, 8
    interval = np.zeros((2 * block, 2 * block, 4)); interval[..., 3] = 1.0  # all open: everything reads the cache
    front, behind = np.array([0.1, 0.5, 0.9]), np.array([5.0, 0.0, 0.0])
    probes = np.zeros((N * WORLD_BLOCK, N * N * WORLD_BLOCK, 4)); probes[..., 3] = 1.0
    base = base_cell_for((0.0, 0.0, 0.0))
    for z in range(base[2], base[2] + N):                 # cell z = -2 (centre -1.5) is behind the wall at -1.2
        for y in range(base[1], base[1] + N):
            for x in range(base[0], base[0] + N):
                sx, sy, sz = x % N, y % N, z % N
                x0, y0 = atlas_texel((sx, sy, sz), (0, 0))
                probes[y0:y0 + WORLD_BLOCK, x0:x0 + WORLD_BLOCK, :3] = behind if (z + 0.5) * SPACING < -1.2 else front
    state = np.zeros((N, N * N, 4))
    out = _ws_merge(gpu, cam, g, interval, probes, state, (0.0, 0.0, 0.0), *FORWARD, block, spacing_px, normal=g["normal"])
    np.testing.assert_allclose(out[..., :3].reshape(-1, 3).mean(0), front, rtol=0.02)