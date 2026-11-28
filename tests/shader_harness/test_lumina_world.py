"""Lumina world-space radiance cache shaders on lavapipe."""

import numpy as np
import pytest

import pc
from harness import Dispatch, Img, groups_for
from scene import LONE_SUPPORT, Camera, Plane, parent_support, render

WORLD_BLOCK = 8
N = 4
LEVELS = 2           # WORLD_LEVELS: level l's cells are SPACING * 2**l apart, stacked down the atlas
SPACING = 1.0
CLASSIFY_GROUPS = (-(-N // 4), -(-N // 4), -(-LEVELS * N // 4))   # 4^3 threads per group, every level's slots
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


def atlas_texel(slot, dt, level=0):
    return ((slot[0] + slot[2] * N) * WORLD_BLOCK + dt[0], (slot[1] + level * N) * WORLD_BLOCK + dt[1])


def base_cell_for(cam_pos, level=0):
    return tuple(int(np.floor(c / (SPACING * 2 ** level))) - N // 2 for c in cam_pos)


def world_pc(cam, cam_pos, right, up, base, a1w=0, screen_block=0, misc_z=RAY_LENGTH, steps=32.0):
    return pc.world((cam.width, cam.height), N, screen_block or WORLD_BLOCK, base, a1w, cam_pos, SPACING,
                    right, 0.25, up, 0.1, cam.proj, cam.depth_params, misc_z, steps)


def radiance_mip_count(cam):
    return int(np.log2(max(cam.width, cam.height))) + 1


def radiance_mip_dispatches(cam):
    """As in the C++: the radiance mip chain (colour plus linear depth, then box-filtered mips) from "color" and
    "depth" into "radiance"."""
    dispatches = [Dispatch(
        "lumina/lumina_radiance_copy.comp.glsl",
        {0: ("color", "sampled_nearest"), 1: ("depth", "sampled_nearest"), 2: ("radiance", "storage", 0), 3: ("color", "sampled_linear"), 4: ("color", "sampled_nearest"), 5: ("color", "sampled_linear"), 6: ("color", "sampled_nearest"),
         7: ("color", "sampled_nearest")},
        pc.radiance_copy(cam.width, cam.height, cam.depth_params), groups_for(cam.width, cam.height))]
    for m in range(1, radiance_mip_count(cam)):
        size = (max(1, cam.width >> m), max(1, cam.height >> m))
        dispatches.append(Dispatch(
            "lumina/lumina_radiance_downsample.comp.glsl",
            {0: ("radiance", "sampled_nearest", m - 1), 1: ("radiance", "storage", m)}, pc.downsample(*size), groups_for(*size)))
    return dispatches


def classify_and_update(gpu, cam, g, cam_pos, right, up, state=None, probes=None, frame=0):
    """As in the C++, the update reads the radiance mip chain."""
    base = base_cell_for(cam_pos)
    state = np.zeros((LEVELS * N, N * N, 4)) if state is None else state
    probes = np.zeros((LEVELS * N * WORLD_BLOCK, N * N * WORLD_BLOCK, 4)) if probes is None else probes
    push = world_pc(cam, cam_pos, right, up, base, a1w=frame)
    mips = radiance_mip_count(cam)
    dispatches = radiance_mip_dispatches(cam)
    dispatches += [
        Dispatch("lumina/lumina_ws_classify.comp.glsl", {0: ("depth", "sampled_nearest"), 1: ("state", "storage")}, push, CLASSIFY_GROUPS),
        Dispatch("lumina/lumina_ws_update.comp.glsl",
                 {0: ("depth", "sampled_nearest"), 1: ("radiance", "sampled_linear"), 2: ("state", "sampled_nearest"),
                  3: ("probes", "storage"), 4: ("normal", "sampled_nearest")},
                 push, groups_for(N * N * WORLD_BLOCK, LEVELS * N * WORLD_BLOCK))]
    out = gpu.run(
        {"depth": Img("r32f", data=g["depth"]), "color": Img("rgba16f", data=g["color"]),
         "normal": Img("rgba8", data=g["normal"]), "radiance": Img("rgba16f", cam.width, cam.height, mips=mips),
         "state": Img("rgba32f", data=state), "probes": Img("rgba16f", data=probes)},
        dispatches, ["state", "probes"])
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


def test_update_takes_a_small_bright_patch_at_its_cones_average(gpu):
    # Each direction stands for a 64th of the sphere. Read at one pixel, a small bright patch was caught at full
    # strength by the direction pointing at it, which then held the patch's peak across its whole cone (the
    # Cornell box's ceiling, just above its light). Sampled at the cone's width it gives the cone's average.
    cam = Camera(width=64, height=48)
    cell = (0, 0, -2)                                       # centre (0.5, 0.5, -1.5)
    centre = (np.array(cell) + 0.5) * SPACING
    texel = dir_texel((0.0, 0.0, -1.0))
    d = oct_decode((np.array(texel) + 0.5) / WORLD_BLOCK * 2.0 - 1.0)
    aim = centre + d * ((-6.0 - centre[2]) / d[2])          # where that direction meets the wall at z = -6
    half = 0.2                                              # a 0.4 m patch, about 3 px wide at 6 m
    patch = Plane((0, 0, -5.99), (0, 0, 1), (10.0, 10.0, 10.0),
                  ((aim[0] - half, aim[0] + half), (aim[1] - half, aim[1] + half), (-7.0, -5.0)))
    g = render(cam, [patch, Plane((0, 0, -6), (0, 0, 1), (0.0, 0.0, 0.0))])
    _, probes, _ = classify_and_update(gpu, cam, g, (0.0, 0.0, 0.0), *FORWARD)
    got = probes[atlas_texel(tuple(c % N for c in cell), texel)[::-1]]
    assert got[3] == 1.0                                    # sampled
    assert 0.02 < got[0] < 2.0, got[0]                      # some of the patch's light, nowhere near its peak of 10


def test_update_keeps_a_nearer_objects_light_out_of_a_hit_beside_it(gpu):
    # At the cone's width, a hit's texel can take in something well in front of it, such as the edge of a bright
    # object beside the hit on screen. The hit steps to finer mips until nothing in its texel is that near, as in
    # the screen-space trace; without that the wall beside the object took the object's light.
    cam = Camera(width=64, height=48)
    cell = (0, 0, -2)                                       # centre (0.5, 0.5, -1.5)
    centre = (np.array(cell) + 0.5) * SPACING
    texel = dir_texel((0.0, 0.0, -1.0))
    d = oct_decode((np.array(texel) + 0.5) / WORLD_BLOCK * 2.0 - 1.0)
    aim = centre + d * ((-6.0 - centre[2]) / d[2])          # where that direction meets the wall at z = -6
    edge = 3.0 * (aim[0] / 6.0 - 0.03)                      # a card at z = -3 ending just left of the hit on screen
    card = Plane((0, 0, -3.0), (0, 0, 1), (10.0, 10.0, 10.0), ((-3.0, edge), (-3.0, 3.0), (-4.0, -2.0)))
    g = render(cam, [card, Plane((0, 0, -6), (0, 0, 1), (0.1, 0.1, 0.1))])
    assert centre[0] + d[0] * ((-3.0 - centre[2]) / d[2]) > edge     # the ray itself passes clear of the card
    _, probes, _ = classify_and_update(gpu, cam, g, (0.0, 0.0, 0.0), *FORWARD)
    got = probes[atlas_texel(tuple(c % N for c in cell), texel)[::-1]]
    assert got[3] == 1.0
    assert got[0] < 0.5, got[0]                             # the wall's 0.1, not the card's 10 blurred in


def test_update_does_not_record_the_lit_front_of_an_object_it_passes_behind(gpu):
    # A ray that passes just behind an object, heading back toward the camera's side of it, lands within the hit
    # thickness behind the object's front face, which the depth buffer can't tell from a hit. That face points
    # away from the ray, so the ray is blocked there. Recording the lit face's colour instead lit the Cornell box's
    # back wall from directions that see nothing.
    cam = Camera(width=64, height=48)
    occluder = Plane((0, 0, -1.3), (0, 0, 1), (4.0, 4.0, 4.0), ((-1.0, 0.3), (-1.0, 1.0), (-2.0, -1.0)))
    g = render(cam, [occluder, Plane((0, 0, -6), (0, 0, 1), (0.1, 0.1, 0.1))])
    cell = (0, 0, -2)                                       # centre (0.5, 0.5, -1.5): seen beside the occluder
    centre = (np.array(cell) + 0.5) * SPACING
    texel = dir_texel((-0.9, -0.2, 0.35))
    d = oct_decode((np.array(texel) + 0.5) / WORLD_BLOCK * 2.0 - 1.0)
    t = (0.3 - centre[0]) / d[0]                            # where the ray passes the occluder's edge
    behind = -1.3 - (centre[2] + d[2] * t)
    assert d[2] > 0.0 and 0.0 < behind < 0.25 * 1.3        # toward the camera, within the hit thickness
    _, probes, _ = classify_and_update(gpu, cam, g, (0.0, 0.0, 0.0), *FORWARD)
    got = probes[atlas_texel(tuple(c % N for c in cell), texel)[::-1]]
    assert got[3] == 1.0                                    # sampled: blocked, not unknown
    assert got[:3].max() < 0.05, got


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


def _ws_merge(gpu, cam, g, interval, probes, state, cam_pos, right, up, screen_block, spacing_px, flags=0):
    base = base_cell_for(cam_pos)
    push = world_pc(cam, cam_pos, right, up, base, a1w=flags, screen_block=screen_block, misc_z=float(spacing_px))
    return gpu.run(
        {"depth": Img("r32f", data=g["depth"]), "probes": Img("rgba16f", data=probes), "state": Img("rgba32f", data=state),
         "lvl": Img("rgba16f", data=interval), "normal": Img("rgba8", data=g["normal"])},
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
    probes = np.zeros((LEVELS * N * WORLD_BLOCK, N * N * WORLD_BLOCK, 4)); probes[..., :3] = W; probes[..., 3] = 1.0
    state = np.zeros((LEVELS * N, N * N, 4))
    out = _ws_merge(gpu, cam, g, interval, probes, state, (0.0, 0.0, 0.0), *FORWARD, block, spacing_px)
    right_half = out[:, probes_x * block // 2:]
    np.testing.assert_allclose(right_half[..., :3].reshape(-1, 3).mean(0), W, rtol=0.02)
    assert np.all(right_half[..., 3] < 0.05)                 # far field fully covered
    np.testing.assert_allclose(out[:, : probes_x * block // 2], interval[:, : probes_x * block // 2], atol=1e-3)

    only_world = _ws_merge(gpu, cam, g, interval, probes, state, (0.0, 0.0, 0.0), *FORWARD, block, spacing_px, flags=1)
    np.testing.assert_allclose(only_world[..., :3].reshape(-1, 3).mean(0), W, rtol=0.02)


def test_ws_merge_lights_only_the_lone_probes_of_a_lower_level(gpu):
    # Below the top level the world merge runs for lone probes alone (flags bit 1): those whose parents can't stand
    # in for them, so they take no parent light (lumina_rc_trace.comp.glsl). What their rays leave clear gets the
    # cache, and they end at transmittance 0 so the screen merge adds nothing after. Without this, light from
    # behind the camera never reached them. Every other probe keeps its interval for the screen merge. The floor
    # and back wall are inside the grid; at 32 x 24 a third of the probes are lone.
    cam = Camera(width=32, height=24)
    g = render(cam, [Plane((0, -0.3, 0), (0, 1, 0), (0.1, 0.1, 0.1)), Plane((0, 0, -1.9), (0, 0, 1), (0.1, 0.1, 0.1))])
    spacing_px, block = 4, 8
    lone = parent_support(g["linear_z"], spacing_px) < LONE_SUPPORT
    assert 0 < lone.sum() < lone.size
    interval = np.zeros((lone.shape[0] * block, lone.shape[1] * block, 4))
    interval[...] = (0.2, 0.0, 0.0, 0.5)
    W = np.array([0.1, 0.5, 0.9])
    probes = np.zeros((LEVELS * N * WORLD_BLOCK, N * N * WORLD_BLOCK, 4)); probes[..., :3] = W; probes[..., 3] = 1.0
    state = np.zeros((LEVELS * N, N * N, 4))
    out = _ws_merge(gpu, cam, g, interval, probes, state, (0.0, 0.0, 0.0), *FORWARD, block, spacing_px, flags=2)
    per_probe = out.reshape(lone.shape[0], block, lone.shape[1], block, 4).transpose(0, 2, 1, 3, 4)
    np.testing.assert_allclose(per_probe[lone][..., :3].reshape(-1, 3).mean(0), (0.2, 0.0, 0.0) + 0.5 * W, rtol=0.02)
    assert np.all(per_probe[lone][..., 3] == 0.0)
    np.testing.assert_allclose(per_probe[~lone], interval[:block, :block][None].repeat((~lone).sum(), 0), atol=1e-3)


def _view_to_world(v, right, up):
    return np.array(right) * v[0] + np.array(up) * v[1] + np.cross(right, up) * v[2]


def _fill_cells(value_for_cell, levels=range(LEVELS)):
    """World-cache atlas where every direction of a live cell holds value_for_cell(cell, centre, level), camera at
    the origin."""
    probes = np.zeros((LEVELS * N * WORLD_BLOCK, N * N * WORLD_BLOCK, 4)); probes[..., 3] = 1.0
    for level in levels:
        base = base_cell_for((0.0, 0.0, 0.0), level)
        for z in range(base[2], base[2] + N):
            for y in range(base[1], base[1] + N):
                for x in range(base[0], base[0] + N):
                    x0, y0 = atlas_texel((x % N, y % N, z % N), (0, 0), level)
                    centre = (np.array([x, y, z]) + 0.5) * SPACING * 2 ** level
                    probes[y0:y0 + WORLD_BLOCK, x0:x0 + WORLD_BLOCK, :3] = value_for_cell((x, y, z), centre, level)
    return probes


@pytest.mark.parametrize("basis", [FORWARD, BACKWARD], ids=["forward", "turned_around"])
def test_ws_merge_ignores_cells_behind_the_probes_surface(gpu, basis):
    # A probe on a wall blends the 8 cache cells around it. Cells on the far side of the wall see a different
    # space (outside a room, inside a pillar) and classify can't always mark them buried: from the camera they
    # can line up with empty background. Blending them in let a wall light itself (the Cornell box came out
    # nearly twice the reference with the cache on). Only cells on the side the surface faces may count.
    # Turned around, view and world axes differ, so a view/world mix-up in the merge would fail here.
    cam = Camera(width=32, height=24)
    g = render(cam, [Plane((0, 0, -1.2), (0, 0, 1), (0.1, 0.1, 0.1))])   # wall facing the camera, inside the grid
    spacing_px, block = 16, 8
    interval = np.zeros((2 * block, 2 * block, 4)); interval[..., 3] = 1.0  # all open: everything reads the cache
    wall_point, wall_normal = _view_to_world((0, 0, -1.2), *basis), _view_to_world((0, 0, 1), *basis)
    front, behind = np.array([0.1, 0.5, 0.9]), np.array([5.0, 0.0, 0.0])
    probes = _fill_cells(lambda cell, centre, level: behind if np.dot(centre - wall_point, wall_normal) < 0 else front)
    state = np.zeros((LEVELS * N, N * N, 4))
    out = _ws_merge(gpu, cam, g, interval, probes, state, (0.0, 0.0, 0.0), *basis, block, spacing_px)
    np.testing.assert_allclose(out[..., :3].reshape(-1, 3).mean(0), front, rtol=0.02)


def test_ws_merge_interpolates_on_a_plane_of_cell_centres(gpu):
    # A wall lying exactly on a plane of cell centres (the Cornell box's back and side walls at 1 m spacing): the
    # cells on that plane carry the interpolation. With no tolerance, normal quantisation alone decided which of
    # them counted as behind, and probes came out wrong by up to a whole cell's value in a noise pattern.
    cam = Camera(width=32, height=24)
    g = render(cam, [Plane((0, 0, -1.5), (0, 0, 1), (0.1, 0.1, 0.1))])   # z = -1.5: cell centres at k + 0.5
    spacing_px, block = 16, 8
    interval = np.zeros((2 * block, 2 * block, 4)); interval[..., 3] = 1.0
    # A linear ramp in world x + y on both levels: trilinear is exact on each, and so is their blend.
    probes = _fill_cells(lambda cell, centre, level: centre[0] / SPACING - 0.5 + centre[1] / SPACING - 0.5)
    state = np.zeros((LEVELS * N, N * N, 4))
    out = _ws_merge(gpu, cam, g, interval, probes, state, (0.0, 0.0, 0.0), *FORWARD, block, spacing_px)
    dirs = cam.ray_dirs()
    for py in range(2):
        for px in range(2):
            cx, cy = min(int((px + 0.5) * spacing_px), cam.width - 1), min(int((py + 0.5) * spacing_px), cam.height - 1)
            p = dirs[cy, cx] * g["linear_z"][cy, cx]                       # FORWARD at the origin: view == world
            expected = (p[0] / SPACING - 0.5) + (p[1] / SPACING - 0.5)
            got = out[py * block:(py + 1) * block, px * block:(px + 1) * block, 0]
            assert np.abs(got - expected).max() < 0.01, f"probe ({px}, {py}): {got.mean():.3f}, expected {expected:.3f}"


def _ws_merge_levels(gpu, cam, g, level_values, n=N, spacing_px=16, block=8):
    """World merge with the camera at the origin looking down -Z, every interval open, and each level's cells holding
    one colour in every direction (level_values[level]). The grid is n^3 cells per level."""
    probes = np.zeros((LEVELS * n * WORLD_BLOCK, n * n * WORLD_BLOCK, 4))
    for level, value in enumerate(level_values):
        probes[level * n * WORLD_BLOCK:(level + 1) * n * WORLD_BLOCK] = (*value, 1.0)
    probes_x, probes_y = -(-cam.width // spacing_px), -(-cam.height // spacing_px)
    interval = np.zeros((probes_y * block, probes_x * block, 4)); interval[..., 3] = 1.0
    push = pc.world((cam.width, cam.height), n, block, (-(n // 2),) * 3, 0, (0.0, 0.0, 0.0), SPACING, *FORWARD[:1], 0.25,
                    *FORWARD[1:], 0.1, cam.proj, cam.depth_params, float(spacing_px), 32.0)
    out = gpu.run(
        {"depth": Img("r32f", data=g["depth"]), "probes": Img("rgba16f", data=probes),
         "state": Img("rgba32f", data=np.zeros((LEVELS * n, n * n, 4))), "lvl": Img("rgba16f", data=interval),
         "normal": Img("rgba8", data=g["normal"])},
        [Dispatch("lumina/lumina_ws_merge.comp.glsl",
                  {0: ("depth", "sampled_nearest"), 1: ("probes", "sampled_nearest"), 2: ("state", "sampled_nearest"),
                   3: ("lvl", "storage"), 4: ("normal", "sampled_nearest")},
                  push, groups_for(interval.shape[1], interval.shape[0]))], ["lvl"])["lvl"]
    return out.reshape(probes_y, block, probes_x, block, 4).mean((1, 3))   # per probe


def test_ws_merge_lights_a_surface_beyond_level_0_from_level_1(gpu):
    # Level 0 reaches N/2 cells around the camera, 8 m at 1 m spacing. Surfaces past it got nothing from the cache:
    # in Sponza the far half of the upper galleries, lit by sun on gallery floors the camera never sees, came out at
    # a third of their light. Level 1, at twice the spacing, reaches twice as far. The wall at z = -3 is past level
    # 0's 4^3 grid (whose cells reach z = -1.5) and inside level 1's.
    cam = Camera(width=32, height=24)
    g = render(cam, [Plane((0, 0, -3.0), (0, 0, 1), (0.1, 0.1, 0.1))])
    level0, level1 = (5.0, 0.0, 0.0), (0.1, 0.5, 0.9)
    per_probe = _ws_merge_levels(gpu, cam, g, [level0, level1])
    np.testing.assert_allclose(per_probe[..., :3].reshape(-1, 3).mean(0), level1, rtol=0.02)
    assert np.all(per_probe[..., 3] < 0.05)                  # fully covered


def test_ws_merge_hands_over_between_levels_without_a_seam(gpu):
    # Down a floor running away from the camera, the answer goes from level 0's to level 1's across level 0's last
    # two cells, not in one step where a line would show. 8^3 cells per level, so level 0 has an inside where it
    # answers alone: its cells reach z = -3.5, and the handover runs from -1.5 to -3.5.
    n = 8
    cam = Camera(width=64, height=48)
    g = render(cam, [Plane((0, -0.5, 0), (0, 1, 0), (0.1, 0.1, 0.1))])
    spacing_px = 2
    per_probe = _ws_merge_levels(gpu, cam, g, [(1.0, 0.0, 0.0), (0.0, 0.0, 0.0)], n=n, spacing_px=spacing_px)
    column = per_probe.shape[1] // 2                          # probes near x = 0
    z = np.array([g["linear_z"][min(int((r + 0.5) * spacing_px), cam.height - 1), int((column + 0.5) * spacing_px)]
                  for r in range(per_probe.shape[0])])
    rows = (z > 0.0) & (z < 6.0)                              # the floor (sky is 0), inside level 1's reach
    z, red = z[rows], per_probe[rows, column, 0]
    order = np.argsort(z)
    z, red = z[order], red[order]
    assert np.all(red[z < 1.4] > 0.98) and np.all(red[z > 3.6] < 0.02), list(zip(z.round(2), red.round(3)))
    assert np.sum((z > 1.5) & (z < 3.5)) >= 4                 # the handover is sampled
    steps = np.diff(red) / np.diff(z)
    assert np.all(steps <= 0.02) and np.all(steps > -0.6), list(zip(z.round(2), red.round(3)))   # 0.5 per metre
