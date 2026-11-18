"""Track B virtual shadow page shaders on lavapipe: page requests and allocation, eviction and invalidation,
clear + cull + raster (with alpha-tested casters), and the PCSS sampler's coarser-level fallback."""

import struct

import numpy as np

import track_b as tb
from harness import Buf, Dispatch, Img, groups_for

REQUEST = "shadows/vsp_page_request_and_invalidate.comp.glsl"


def ground_setup(**kw) -> tb.VspSetup:
    eye = (0.0, 3.0, 3.0)
    return tb.VspSetup(view=tb.look_at_view(eye, (0.0, 0.0, 0.0)), proj=tb.godot_projection(60.0, 1.5, 0.05, 100.0),
                       camera_pos=eye, width=24, height=16, **kw)


def run_request(gpu, setup, page_table=None, free_stack=None, state=None, modes=(0, 1)):
    depth, world = setup.ground_depth()
    pt = Img("r32ui", 128, 128, layers=8, data=page_table) if page_table is not None else Img("r32ui", 128, 128, layers=8)
    bufs = {"ubo0": Buf(data=setup.ubo(pass_mode=0)), "ubo1": Buf(data=setup.ubo(pass_mode=1)), "aabbs": Buf(size=32 * 4),
            "free": Buf(data=free_stack if free_stack is not None else tb.u32s(*range(tb.VSP_MAX_PAGES))),
            "state": Buf(data=state if state is not None else tb.vsp_state()), "queue": Buf(size=8 * setup.max_dirty_pages)}
    dispatches = []
    for mode in modes:
        groups = (16, 16, 8) if mode == 0 else groups_for(setup.width, setup.height)
        dispatches.append(Dispatch(REQUEST, {0: (f"ubo{mode}", "uniform_buffer"), 1: ("depth", "sampled_nearest"), 2: ("pt", "storage"),
                                             3: ("aabbs", "storage_buffer"), 4: ("free", "storage_buffer"), 5: ("state", "storage_buffer"),
                                             6: ("queue", "storage_buffer")}, groups=groups))
    out = gpu.run(images={"depth": Img("r32f", data=depth.astype(np.float32)), "pt": pt}, buffers=bufs,
                  dispatches=dispatches, outputs=["pt", "free", "state", "queue"])
    state_out = tb.unpack_vsp_state(out["state"])
    n = min(state_out["dirty"], setup.max_dirty_pages)
    queue = [struct.unpack_from("<2I", out["queue"], 8 * i) for i in range(n)]
    return out["pt"][..., 0], out["free"], state_out, queue, world


def selected_level(setup, p):
    """select_vsp_clipmap_level for a ground point, or None within 0.1% of a level boundary."""
    view_depth = max(-(setup.view @ np.append(p, 1.0))[2], setup.near)
    base = setup.levels[0].extent
    ratio = max(view_depth * 2.2 * 2 ** setup.lod_bias, base) / base
    if ratio > 1 and abs(np.log2(ratio) - np.round(np.log2(ratio))) < 1e-3:
        return None
    return int(np.clip(np.ceil(np.log2(ratio)), 0, tb.VSP_LEVELS - 1))


def pixel_pages(setup, world, level=None):
    """(level, linear page) under each ground pixel, at `level` or the pixel's selected level; skips pixels within
    0.1% of a page or level boundary."""
    pages = set()
    for p in world.reshape(-1, 3):
        lv = selected_level(setup, p) if level is None else level
        if lv is None:
            continue
        f = setup.levels[lv].uv(p) * tb.VSP_PAGES
        if np.all(np.abs(f - np.round(f)) > 1e-3):
            pages.add((lv, setup.levels[lv].linear_page(p)))
    return pages


def test_request_allocates_and_queues_every_page_shading_reads(gpu):
    setup = ground_setup()
    pt, free, state, queue, world = run_request(gpu, setup)
    valid = tb.PTE_VALID | tb.PTE_DIRTY | tb.PTE_ALLOCATED | tb.PTE_REQUESTED

    # Every queued page is VALID with the physical page it was queued with, all distinct, popped off the stack top
    phys = []
    for packed, page in queue:
        level, x, y = packed >> 24, packed & 0xFF, (packed >> 8) & 0xFF
        assert pt[level, y, x] & 0xF0000000 == valid
        assert pt[level, y, x] & tb.PTE_PAGE_MASK == page
        phys.append(page)
    n = len(queue)
    assert n > 4 and sorted(phys) == list(range(tb.VSP_MAX_PAGES - n, tb.VSP_MAX_PAGES))
    assert np.count_nonzero(pt & tb.PTE_VALID) == n
    assert state["free"] == tb.VSP_MAX_PAGES - n and state["clear_dispatch"] == (n, 1, 1)

    # The page under every pixel, at its selected level and at the coarsest level, is resident
    own = pixel_pages(setup, world)
    assert {lv for lv, _ in own} == {0, 1}           # this view spans two levels
    for level, page in own | pixel_pages(setup, world, tb.VSP_LEVELS - 1):
        x, y = setup.levels[level].toroidal(page)
        assert pt[level, y, x] & tb.PTE_VALID, (level, page)
    # The filter footprint reaches past the pixels' own pages at page edges
    queued = {(p >> 24, (p & 0xFF, (p >> 8) & 0xFF)) for p, _ in queue}
    assert queued >= own and len(queued - own - pixel_pages(setup, world, tb.VSP_LEVELS - 1)) > 0

    # Dirty rects bound the queued pages per level
    for level in range(tb.VSP_LEVELS):
        pages = [(p & 0xFF, (p >> 8) & 0xFF) for p, _ in queue if p >> 24 == level]
        rect = state["rects"][level]
        if pages:
            xs, ys = zip(*pages)
            assert rect == (min(xs), min(ys), max(xs), max(ys))
        else:
            assert rect == (tb.EMPTY, tb.EMPTY, 0, 0)


def test_full_render_queue_leaves_pages_invalid_until_queued(gpu):
    setup = ground_setup(max_dirty_pages=3)
    pt, free, state, queue, _ = run_request(gpu, setup)
    claimed = tb.PTE_DIRTY | tb.PTE_ALLOCATED | tb.PTE_REQUESTED
    assert len(queue) == 3 and state["dirty"] > 3 and state["clear_dispatch"] == (3, 1, 1)
    assert np.count_nonzero(pt & tb.PTE_VALID) == 3                  # nothing unqueued claims to be valid
    waiting = (pt & 0xF0000000) == claimed
    assert np.count_nonzero(waiting) == state["dirty"] - 3            # the rest: allocated, invalid
    total, allocated_before = state["dirty"], state["free"]

    # Next frame, with room: the waiting pages get queued; the three already valid are not redone
    setup2 = ground_setup(max_dirty_pages=4096)
    pt2, _, state2, queue2, _ = run_request(gpu, setup2, page_table=pt, free_stack=free, state=tb.vsp_state(free_count=allocated_before))
    assert len(queue2) == total - 3
    assert np.count_nonzero(pt2 & tb.PTE_VALID) == total
    assert state2["free"] == allocated_before                        # no new allocations


def test_invalidate_ages_evicts_and_invalidates(gpu):
    levels = [tb.VspLevel(16.0 * 2 ** k) for k in range(tb.VSP_LEVELS)]
    levels[1] = tb.VspLevel(32.0, offset=(3, 0), scroll=(3, 0))      # moved 3 pages in +x this frame
    setup = ground_setup(levels=levels, evict_after=8)
    pt = np.zeros((8, 128, 128), dtype=np.uint64)
    alloc, valid, req, dirty = tb.PTE_ALLOCATED, tb.PTE_VALID, tb.PTE_REQUESTED, tb.PTE_DIRTY
    pt[0, 5, 5] = alloc | valid | (7 << tb.PTE_AGE_SHIFT) | 11            # unused for 8 frames now: evicted
    pt[0, 6, 6] = alloc | valid | req | dirty | (5 << tb.PTE_AGE_SHIFT) | 12  # used: age back to 0
    pt[0, 7, 7] = req | dirty                                            # failed claim: cleared
    pt[0, 8, 8] = alloc | (2 << tb.PTE_AGE_SHIFT) | 13                    # ages
    pt[1, 10, 1] = alloc | valid | req | 14                               # linear x 126: scrolled in, invalid
    pt[1, 10, 13] = alloc | valid | req | 15                              # linear x 10: untouched
    free_stack = tb.u32s(*range(100)) + bytes(4 * (tb.VSP_MAX_PAGES - 100))
    out_pt, free, state, _, _ = run_request(gpu, setup, page_table=pt, free_stack=free_stack,
                                            state=tb.vsp_state(free_count=100), modes=(0,))
    assert out_pt[0, 5, 5] == 0 and state["free"] == 101 and struct.unpack_from("<I", free, 4 * 100)[0] == 11
    assert out_pt[0, 6, 6] == alloc | valid | 12
    assert out_pt[0, 7, 7] == 0
    assert out_pt[0, 8, 8] == alloc | (3 << tb.PTE_AGE_SHIFT) | 13
    assert out_pt[1, 10, 1] == alloc | 14
    assert out_pt[1, 10, 13] == alloc | valid | 15

    forced, _, _, _, _ = run_request(gpu, ground_setup(force_invalidate=1), page_table=pt, free_stack=free_stack,
                                     state=tb.vsp_state(free_count=100), modes=(0,))
    assert not np.any(forced & tb.PTE_VALID) and forced[0, 6, 6] == alloc | 12


# --------------------------------------------------------------------------- clear + cull + raster

def quad_mesh(x0, x1, z0, z1, y):
    """Horizontal quad at height y; UV u runs with x over [x0, x1]."""
    pos = [(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)]
    uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
    return pos, uvs, [(0, 1, 2), (0, 2, 3)]


def test_clear_cull_raster_page_with_alpha_tested_caster(gpu):
    page, phys = (64, 64), 37            # level-0 page covering world x, z in [0, 0.125)
    casters = [  # (quad, flags, material)
        (quad_mesh(-0.01, 0.0625, -0.01, 0.135, 2.0), tb.FLAG_VISIBLE | tb.FLAG_CAST_SHADOW, 0),                    # left half, z 0.48
        (quad_mesh(0.0, 0.125, -0.01, 0.135, 3.0), tb.FLAG_VISIBLE | tb.FLAG_CAST_SHADOW | tb.FLAG_ALPHA_TEST, 1),  # alpha: left opaque, z 0.47
        (quad_mesh(-0.01, 0.135, -0.01, 0.135, 1.0), tb.FLAG_VISIBLE | tb.FLAG_CAST_SHADOW, 0),                     # whole page, z 0.49
        (quad_mesh(-0.01, 0.135, -0.01, 0.135, 5.0), tb.FLAG_VISIBLE, 0),                                           # casts no shadow
        (quad_mesh(5.0, 5.1, 5.0, 5.1, 2.0), tb.FLAG_VISIBLE | tb.FLAG_CAST_SHADOW, 0),                             # elsewhere
    ]
    positions, attrs, mverts, mtris, meshlets, instances = b"", b"", b"", b"", b"", b""
    for i, ((pos, uvs, tris), flags, material) in enumerate(casters):
        positions += b"".join(struct.pack("<4f", *p, 1.0) for p in pos)
        attrs += b"".join(tb.gpu_vertex_attributes((0, 1, 0), uv=uv) for uv in uvs)
        mverts += tb.u32s(0, 1, 2, 3)
        mtris += tb.pack_triangles(tris)
        center = np.mean(np.array(pos), axis=0)
        meshlets += tb.gpu_meshlet(4 * i, 2 * i, 4, 2, center, float(np.max(np.linalg.norm(np.array(pos) - center, axis=1))))
        instances += tb.gpu_instance(np.eye(4), i, 1, 4 * i, material, flags)
    materials = tb.gpu_material() + tb.gpu_material(alpha_cutoff=0.5, tex=(0, tb.INVALID_TEX, tb.INVALID_TEX, tb.INVALID_TEX))
    alpha_tex = np.array([[[1, 1, 1, 1], [1, 1, 1, 0]]], float)  # u < 0.5 opaque, u >= 0.5 transparent

    setup = ground_setup(candidates=len(casters))
    pt = np.zeros((8, 128, 128), dtype=np.uint64)
    pt[0, page[1], page[0]] = tb.PTE_VALID | tb.PTE_DIRTY | tb.PTE_ALLOCATED | tb.PTE_REQUESTED | phys
    rects = [(page[0], page[1], page[0], page[1])] + [(tb.EMPTY, tb.EMPTY, 0, 0)] * 7
    state = tb.vsp_state(dirty=1, clear_dispatch=(1, 1, 1), rects=rects)
    ubo = setup.ubo()
    bufs = {"ubo": Buf(data=ubo), "queue": Buf(data=tb.u32s((page[1] << 8) | page[0], phys)),
            "state": Buf(data=state), "meshlets": Buf(data=meshlets), "instances": Buf(data=instances),
            "candidates": Buf(data=b"".join(tb.u32s(i, 0) for i in range(len(casters)))), "items": Buf(size=16 * 64),
            "positions": Buf(data=positions), "mverts": Buf(data=mverts), "mtris": Buf(data=mtris), "attrs": Buf(data=attrs),
            "materials": Buf(data=materials)}
    images = {"pt": Img("r32ui", 128, 128, layers=8, data=pt), "atlas": Img("r32ui", 1024, 256), "alpha": Img("rgba8", data=alpha_tex)}
    out = gpu.run(images=images, buffers=bufs, outputs=["atlas", "state", "pt"], dispatches=[
        Dispatch("shadows/vsp_clear_pages.comp.glsl", {0: ("ubo", "uniform_buffer"), 1: ("queue", "storage_buffer"),
                 2: ("state", "storage_buffer"), 3: ("atlas", "storage")}, indirect=("state", 16)),
        Dispatch("shadows/vsp_cull_meshlets.comp.glsl", {0: ("ubo", "uniform_buffer"), 1: ("meshlets", "storage_buffer"),
                 2: ("instances", "storage_buffer"), 3: ("candidates", "storage_buffer"), 4: ("pt", "storage"),
                 5: ("items", "storage_buffer"), 6: ("state", "storage_buffer")}, groups=(1, 1, 1)),
        Dispatch("shadows/vsp_raster_pages.comp.glsl", {0: ("ubo", "uniform_buffer"), 1: ("meshlets", "storage_buffer"),
                 2: ("instances", "storage_buffer"), 3: ("positions", "storage_buffer"), 4: ("mverts", "storage_buffer"),
                 5: ("mtris", "storage_buffer"), 6: ("items", "storage_buffer"), 7: ("state", "storage_buffer"),
                 8: ("atlas", "storage"), 9: ("attrs", "storage_buffer"), 10: ("materials", "storage_buffer"),
                 11: (["alpha"], "sampled_nearest")}, indirect=("state", 32))])

    state_out = tb.unpack_vsp_state(out["state"])
    assert state_out["raster"] == 3 and state_out["raster_dispatch"] == (3, 1, 1)   # casters 0-2 x one page
    atlas = out["atlas"][..., 0]
    x0, y0 = (phys % 32) * 128, (phys // 32) * 128
    depth = atlas[y0:y0 + 128, x0:x0 + 128].view(np.float32)
    np.testing.assert_allclose(depth[:, :64], 0.47, atol=1e-6)   # alpha-tested caster's opaque half
    np.testing.assert_allclose(depth[:, 64:], 0.49, atol=1e-6)   # seen through its transparent half
    outside = atlas.copy()
    outside[y0:y0 + 128, x0:x0 + 128] = 0
    assert not outside.any(), "wrote outside its page"


def test_raster_queue_overflow_invalidates_the_page(gpu):
    setup = ground_setup(candidates=4, max_raster_items=2)
    pos, uvs, tris = quad_mesh(-0.01, 0.135, -0.01, 0.135, 2.0)
    center = np.array([0.0625, 2.0, 0.0625])
    meshlets = b"".join(tb.gpu_meshlet(0, 0, 4, 2, center, 0.2) for _ in range(4))
    instances = b"".join(tb.gpu_instance(np.eye(4), 0, 1, 0, 0, tb.FLAG_VISIBLE | tb.FLAG_CAST_SHADOW) for _ in range(4))
    pt = np.zeros((8, 128, 128), dtype=np.uint64)
    pt[0, 64, 64] = tb.PTE_VALID | tb.PTE_DIRTY | tb.PTE_ALLOCATED | tb.PTE_REQUESTED | 5
    rects = [(64, 64, 64, 64)] + [(tb.EMPTY, tb.EMPTY, 0, 0)] * 7
    out = gpu.run(images={"pt": Img("r32ui", 128, 128, layers=8, data=pt)},
                  buffers={"ubo": Buf(data=setup.ubo()), "meshlets": Buf(data=meshlets), "instances": Buf(data=instances),
                           "candidates": Buf(data=b"".join(tb.u32s(i, 0) for i in range(4))), "items": Buf(size=16 * 2),
                           "state": Buf(data=tb.vsp_state(dirty=1, rects=rects))},
                  dispatches=[Dispatch("shadows/vsp_cull_meshlets.comp.glsl", {0: ("ubo", "uniform_buffer"), 1: ("meshlets", "storage_buffer"),
                              2: ("instances", "storage_buffer"), 3: ("candidates", "storage_buffer"), 4: ("pt", "storage"),
                              5: ("items", "storage_buffer"), 6: ("state", "storage_buffer")}, groups=(1, 1, 1))],
                  outputs=["pt", "state"])
    state = tb.unpack_vsp_state(out["state"])
    assert state["raster"] == 4 and state["raster_dispatch"] == (2, 1, 1)
    assert out["pt"][0, 64, 64, 0] & tb.PTE_VALID == 0   # half-drawn: not to be sampled


# --------------------------------------------------------------------------- sampler

def sampler_setup():
    eye = (4.0, 4.0, 0.0)
    return tb.VspSetup(view=tb.look_at_view(eye, (4.0, 0.0, 0.0), up=(0.0, 0.0, -1.0)), proj=tb.godot_projection(20.0, 1.0, 0.05, 100.0),
                       camera_pos=eye, width=8, height=8)


def run_sampler(gpu, setup, fill_right, fill_wrong, vis=None, instances=b"", positions=b"", mtris=b"", mverts=b"", meshlets=b""):
    """Level 0 has no valid pages. Level 1 pages under the receivers point at physical page 0 (depth `fill_right`);
    the level-1 pages a start-level UV would wrongly address point at physical page 1 (depth `fill_wrong`)."""
    depth, world = setup.ground_depth()
    pt = np.zeros((8, 128, 128), dtype=np.uint64)
    lvl0, lvl1 = setup.levels[0], setup.levels[1]
    right, wrong = set(), set()
    for p in world.reshape(-1, 3):
        for dx in (-0.01, 0.0, 0.01):
            for dz in (-0.01, 0.0, 0.01):
                q = p + [dx, 0.0, dz]
                right.add(lvl1.linear_page(q))
                u0 = np.clip(np.floor(lvl0.uv(q) * 128), 0, 127).astype(int)   # level-0 page used as if it were level 1's
                wrong.add((int(u0[0]), int(u0[1])))
    assert not right & wrong, "test needs the two page sets apart"
    for (x, y) in wrong:
        pt[1, y, x] = tb.PTE_VALID | tb.PTE_ALLOCATED | 1
    for (x, y) in right:
        pt[1, y, x] = tb.PTE_VALID | tb.PTE_ALLOCATED | 0
    atlas = np.zeros((128, 256), dtype=np.uint64)
    atlas[:, :128] = np.float32(fill_right).view(np.uint32)
    atlas[:, 128:] = np.float32(fill_wrong).view(np.uint32)
    if vis is None:
        vis = np.full((setup.height, setup.width, 2), tb.EMPTY, dtype=np.uint64)
    dummy = tb.u32s(0, 0, 0, 0)
    out = gpu.run(images={"depth": Img("r32f", data=depth.astype(np.float32)), "vis": Img("rg32ui", data=vis),
                          "pt": Img("r32ui", 128, 128, layers=8, data=pt), "atlas": Img("r32ui", data=atlas), "mask": Img("r8", setup.width, setup.height)},
                  buffers={"ubo": Buf(data=setup.ubo()), "meshlets": Buf(data=meshlets or dummy * 4), "instances": Buf(data=instances or dummy * 10),
                           "positions": Buf(data=positions or dummy), "mverts": Buf(data=mverts or dummy), "mtris": Buf(data=mtris or dummy)},
                  dispatches=[Dispatch("shadows/vsp_sample_shadows.comp.glsl", {
                      0: ("ubo", "uniform_buffer"), 1: ("depth", "sampled_nearest"), 2: ("vis", "sampled_nearest"),
                      3: ("meshlets", "storage_buffer"), 4: ("instances", "storage_buffer"), 5: ("positions", "storage_buffer"),
                      6: ("mverts", "storage_buffer"), 7: ("mtris", "storage_buffer"), 8: ("pt", "sampled_nearest"),
                      9: ("atlas", "sampled_nearest"), 10: ("mask", "storage")}, groups=groups_for(setup.width, setup.height))],
                  outputs=["mask"])
    return out["mask"][..., 0]


def test_sampler_falls_back_to_the_right_place_in_a_coarser_level(gpu):
    setup = sampler_setup()
    blocker = (tb.SUN_TOP - 10.0) / tb.SUN_RANGE   # something 10 m up; the ground is at (50 - 0) / 100 = 0.5
    np.testing.assert_array_equal(run_sampler(gpu, setup, fill_right=blocker, fill_wrong=1.0), 0.0)
    np.testing.assert_array_equal(run_sampler(gpu, setup, fill_right=1.0, fill_wrong=blocker), 1.0)


def test_sampler_mirrored_receiver_keeps_its_normal(gpu):
    """A triangle whose front faces down in local space, mirrored by scale(1, -1, 1), faces up (towards the sun).
    Without the determinant correction its normal points down and the pixel reads as facing away (0)."""
    setup = sampler_setup()
    local = [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)]
    n_local = np.cross(np.subtract(local[2], local[0]), np.subtract(local[1], local[0]))
    assert n_local[1] < 0
    positions = b"".join(struct.pack("<4f", *p, 1.0) for p in local)
    instances = tb.gpu_instance(tb.scale(1, -1, 1), 0, 1, 0, 0, tb.FLAG_VISIBLE)
    vis = np.zeros((setup.height, setup.width, 2), dtype=np.uint64)
    vis[...] = tb.payload(0, 0, 0)
    mask = run_sampler(gpu, setup, fill_right=1.0, fill_wrong=1.0, vis=vis, instances=instances, positions=positions,
                       mtris=tb.pack_triangles([(0, 1, 2)]), mverts=tb.u32s(0, 1, 2),
                       meshlets=tb.gpu_meshlet(0, 0, 3, 1, (0, 0, 0), 1.0))
    np.testing.assert_array_equal(mask, 1.0)
