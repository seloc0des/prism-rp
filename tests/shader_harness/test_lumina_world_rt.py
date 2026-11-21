"""Lumina's ray-query world-cache passes on lavapipe: the update that traces the scene mirror (lumina_ws_update_rt)
and the cache's irradiance (lumina_ws_irradiance). Each scene's triangles go into both the mirror's buffers and the
TLAS (mirror.py); what the camera sees comes from scene.render as in the other world-cache tests.

The grid is 4^3 cells of 1 m around the camera at the origin, so cell (x, y, z) has its probe at (x, y, z) + 0.5."""

import numpy as np
import pytest

from harness import Dispatch, Img, groups_for
from mirror import LIGHT_DIRECTIONAL, LIGHT_OMNI, LIGHT_SPOT, Light, Material, Mesh, Scene, cube, quad, sphere
from scene import Camera, Plane, render
from test_lumina_world import (FORWARD, N, WORLD_BLOCK, atlas_texel, base_cell_for, dir_texel, radiance_mip_count,
                               radiance_mip_dispatches, world_pc)

GREY = Material(albedo=(0.5, 0.5, 0.5))


def atlas_shape():
    return (N * WORLD_BLOCK, N * N * WORLD_BLOCK, 4)


def update_rt(gpu, cam, g, scene, irradiance=None, bounce=1.0, classify=True):
    """One frame of classify + ray-traced update from an empty cache, camera at the origin looking down -Z.
    bounce scales the cache's irradiance at hits (misc.w: 1 multi-bounce, 0 one bounce). Without classify, no probe
    is buried behind what the camera sees."""
    cam_pos = (0.0, 0.0, 0.0)
    push = world_pc(cam, cam_pos, *FORWARD, base_cell_for(cam_pos), steps=bounce)
    irradiance = np.zeros(atlas_shape()) if irradiance is None else irradiance
    dispatches = radiance_mip_dispatches(cam)
    if classify:
        dispatches.append(Dispatch("lumina/lumina_ws_classify.comp.glsl", {0: ("depth", "sampled_nearest"), 1: ("state", "storage")}, push, (1, 1, 1)))
    dispatches.append(Dispatch(
        "lumina/lumina_ws_update_rt.comp.glsl",
        {0: ("depth", "sampled_nearest"), 1: ("radiance", "sampled_linear"), 2: ("state", "sampled_nearest"),
         3: ("probes", "storage"), 4: ("irradiance", "sampled_nearest"), **scene.bindings()},
        push, groups_for(N * N * WORLD_BLOCK, N * WORLD_BLOCK)))
    images = {"depth": Img("r32f", data=g["depth"]), "color": Img("rgba16f", data=g["color"]),
              "radiance": Img("rgba16f", cam.width, cam.height, mips=radiance_mip_count(cam)),
              "state": Img("rgba32f", data=np.zeros((N, N * N, 4))), "probes": Img("rgba16f", data=np.zeros(atlas_shape())),
              "irradiance": Img("rgba16f", data=irradiance), **scene.images()}
    out = gpu.run(images, dispatches, ["probes"], buffers=scene.buffers(), acceleration_structures=scene.acceleration_structures())
    return out["probes"]


def irradiance_of(gpu, probes, state=None):
    push = world_pc(Camera(), (0.0, 0.0, 0.0), *FORWARD, base_cell_for((0.0, 0.0, 0.0)))
    state = np.zeros((N, N * N, 4)) if state is None else state
    return gpu.run(
        {"probes": Img("rgba16f", data=probes), "state": Img("rgba32f", data=state), "irradiance": Img("rgba16f", *atlas_shape()[1::-1])},
        [Dispatch("lumina/lumina_ws_irradiance.comp.glsl",
                  {0: ("probes", "sampled_nearest"), 1: ("state", "sampled_nearest"), 2: ("irradiance", "storage")},
                  push, groups_for(N * N * WORLD_BLOCK, N * WORLD_BLOCK))], ["irradiance"])["irradiance"]


def at(atlas, cell, direction):
    """The atlas texel of cell's probe for a world direction (or, in the irradiance atlas, a surface normal)."""
    slot = tuple(c % N for c in cell)
    return atlas[atlas_texel(slot, dir_texel(direction))[::-1]]


def wall_behind_camera(material=GREY, z=6.0, normal=(0.0, 0.0, -1.0)):
    return Mesh(quad((0.0, 0.0, z), normal, 40.0), material)


# Shines along +Z, onto the side of a wall behind the camera that faces the camera.
SUN_BEHIND = Light(LIGHT_DIRECTIONAL, color=(2.0, 2.0, 2.0), direction=(0.0, 0.0, 1.0))


def test_rt_update_lights_a_wall_behind_the_camera(gpu):
    # The march can't see behind the camera at all: those directions stay unknown. Traced, the probe finds the wall
    # and lights it from the scene's light: albedo 0.5 x colour 2 x N.L 1.
    cam = Camera(width=32, height=24)
    probes = update_rt(gpu, cam, render(cam, []), Scene([wall_behind_camera()], [SUN_BEHIND]))
    got = at(probes, (0, 0, 0), (0, 0, 1))
    assert got[3] == 1.0
    assert got[:3] == pytest.approx((1.0, 1.0, 1.0), rel=0.01)


def test_rt_update_finds_nothing_through_an_open_boxs_open_front(gpu):
    # The camera stands in a box whose open side is behind it (+Z). Directions out through the opening meet nothing.
    # The march couldn't tell that from "out of view", so the merge filled them in from neighbouring cells, which
    # over-lit the Cornell box. Traced, they are known and black, while the same probe sees the box the other way
    # (its walls glow, so any wall a ray meets reads the same).
    cam = Camera(width=32, height=24)
    glowing = Material(albedo=(0.5, 0.5, 0.5), emission=(1.0, 1.0, 1.0), emission_energy=1.0)
    walls = [Mesh(quad(c, n, 6.0), glowing) for c, n in [
        ((0.0, -2.0, -2.0), (0.0, 1.0, 0.0)), ((0.0, 2.0, -2.0), (0.0, -1.0, 0.0)),
        ((-2.0, 0.0, -2.0), (1.0, 0.0, 0.0)), ((2.0, 0.0, -2.0), (-1.0, 0.0, 0.0)), ((0.0, 0.0, -5.0), (0.0, 0.0, 1.0))]]
    probes = update_rt(gpu, cam, render(cam, []), Scene(walls))
    out_front = at(probes, (0, 0, 0), (0, 0, 1))
    assert out_front[3] == 1.0
    assert np.all(out_front[:3] == 0.0)
    assert at(probes, (0, 0, 0), (0, 0, -1))[:3] == pytest.approx((1.0, 1.0, 1.0), rel=0.01)


@pytest.mark.parametrize("shadows, expected", [(True, 0.0), (False, 0.5 * 2.0 * np.sqrt(0.5))], ids=["shadowed", "unshadowed"])
def test_rt_update_shadow_rays_stop_light_that_casts_shadows(gpu, shadows, expected):
    # The light comes down at 45 degrees onto the wall behind the camera; a roof high above (out of the probe's view)
    # stands between them. Only a light that casts shadows is stopped by it, as in Godot.
    cam = Camera(width=32, height=24)
    roof = Mesh(quad((0.0, 8.0, 0.0), (0.0, -1.0, 0.0), 60.0), GREY)
    light = Light(LIGHT_DIRECTIONAL, color=(2.0, 2.0, 2.0), direction=(0.0, -1.0, 1.0), shadows=shadows)
    probes = update_rt(gpu, cam, render(cam, []), Scene([wall_behind_camera(), roof], [light]))
    got = at(probes, (0, 0, 0), (0, 0, 1))
    assert got[3] == 1.0
    assert got[:3] == pytest.approx((expected,) * 3, abs=0.01)


def test_rt_update_probe_inside_an_object_sees_nothing_through_it(gpu):
    # A probe whose centre lies inside a closed object only meets the object's back faces. Culling them, as the
    # albedo view does, would let it see the lit wall beyond and light the object's surroundings from inside it.
    cam = Camera(width=32, height=24)
    pillar = Mesh(cube((0.5, 0.5, 0.5), 0.3), GREY, casts_shadows=False)
    probes = update_rt(gpu, cam, render(cam, []), Scene([wall_behind_camera(), pillar], [SUN_BEHIND]))
    inside = at(probes, (0, 0, 0), (0, 0, 1))
    assert inside[3] == 1.0
    assert np.all(inside[:3] == 0.0)
    assert at(probes, (-1, 0, 0), (0, 0, 1))[:3] == pytest.approx((1.0, 1.0, 1.0), rel=0.01)   # beside it


def test_rt_update_lights_the_back_of_a_double_sided_surface(gpu):
    # The wall's front faces away from the camera, but its material draws both sides: the probe sees a surface, lit.
    cam = Camera(width=32, height=24)
    wall = wall_behind_camera(Material(albedo=(0.5, 0.5, 0.5), double_sided=True), normal=(0.0, 0.0, 1.0))
    probes = update_rt(gpu, cam, render(cam, []), Scene([wall], [SUN_BEHIND]))
    assert at(probes, (0, 0, 0), (0, 0, 1))[:3] == pytest.approx((1.0, 1.0, 1.0), rel=0.01)


def test_rt_update_reads_a_hit_the_camera_sees_from_the_screen(gpu):
    # The screen holds Godot's own lighting of what the camera sees (red here); the scene has no lights, so a traced
    # hit lit by the mirror would be black. A hit the camera sees takes the screen's value.
    cam = Camera(width=64, height=48, fov_y_deg=90.0)
    g = render(cam, [Plane((0.0, 0.0, -3.0), (0.0, 0.0, 1.0), (2.0, 0.0, 0.0))])
    wall = Mesh(quad((0.0, 0.0, -3.0), (0.0, 0.0, 1.0), 40.0), GREY)
    probes = update_rt(gpu, cam, g, Scene([wall]))
    got = at(probes, (0, 0, -1), (0, 0, -1))
    assert got[:3] == pytest.approx((2.0, 0.0, 0.0), abs=0.02)


def test_rt_update_doesnt_take_the_screens_view_of_a_surfaces_other_side(gpu):
    # The camera sees the lit (red) front of a double-sided curtain; the probe meets its back, which no light reaches.
    # The screen's red belongs to the other side. Classify would bury a probe straight behind the curtain on screen,
    # so it is skipped: this probe stands for one behind the curtain off to the side, out of view.
    cam = Camera(width=64, height=48, fov_y_deg=90.0)
    g = render(cam, [Plane((0.0, 0.0, -1.0), (0.0, 0.0, 1.0), (2.0, 0.0, 0.0))])
    curtain = Mesh(quad((0.0, 0.0, -1.0), (0.0, 0.0, 1.0), 40.0), Material(albedo=(0.5, 0.5, 0.5), double_sided=True))
    probes = update_rt(gpu, cam, g, Scene([curtain]), classify=False)
    behind = at(probes, (0, 0, -2), (0, 0, 1))
    assert behind[3] == 1.0
    assert np.all(behind[:3] == 0.0)


def test_rt_update_adds_emission_out_of_view(gpu):
    cam = Camera(width=32, height=24)
    panel = wall_behind_camera(Material(albedo=(0.2, 0.2, 0.2), emission=(1.5, 0.5, 0.0), emission_energy=2.0))
    probes = update_rt(gpu, cam, render(cam, []), Scene([panel]))
    assert at(probes, (0, 0, 0), (0, 0, 1))[:3] == pytest.approx((3.0, 1.0, 0.0), abs=0.01)


@pytest.mark.parametrize("multiply, texture, expected", [
    (False, (0.5, 0.25, 0.0, 1.0), (3.0, 2.5, 2.0)),    # (colour + texture) x energy
    (True, (0.5, 0.25, 0.0, 1.0), (1.0, 0.5, 0.0)),     # (colour x texture) x energy
    (True, None, (0.0, 0.0, 0.0)),                      # no texture reads black, as Godot's hint_default_black
], ids=["add", "multiply", "multiply_without_texture"])
def test_rt_update_emission_follows_godots_operators(gpu, multiply, texture, expected):
    cam = Camera(width=32, height=24)
    material = Material(albedo=(0.2, 0.2, 0.2), emission=(1.0, 1.0, 1.0), emission_energy=2.0,
                        emission_texture=1 if texture else 0, emission_multiply=multiply)
    scene = Scene([wall_behind_camera(material)], textures=[texture] if texture else [])
    probes = update_rt(gpu, cam, render(cam, []), scene)
    assert at(probes, (0, 0, 0), (0, 0, 1))[:3] == pytest.approx(expected, abs=0.02)


# A probe at the centre of a closed sphere facing inward, with the light at the probe: every ray meets the sphere at
# the same distance, facing the light, whichever way inside its texel's cone it goes.
PROBE = (0.5, 0.5, 0.5)
SPHERE_RADIUS = 1.5


def godot_omni_attenuation(distance, light_range, decay):
    """Godot's get_omni_attenuation (scene_forward_lights_inc.glsl)."""
    return max(1.0 - (distance / light_range) ** 4, 0.0) ** 2 * distance ** -decay


def inside_sphere(gpu, lights, irradiance=None):
    cam = Camera(width=32, height=24)
    room = Mesh(sphere(PROBE, SPHERE_RADIUS, inward=True), GREY)
    return update_rt(gpu, cam, render(cam, []), Scene([room], lights), irradiance=irradiance)


@pytest.mark.parametrize("light_range, decay", [(3.0, 2.0), (1.8, 1.0)], ids=["inverse_square", "near_the_range"])
def test_rt_update_omni_light_falls_off_as_godots(gpu, light_range, decay):
    lamp = Light(LIGHT_OMNI, color=(2.0, 2.0, 2.0), position=PROBE, range=light_range, attenuation=decay)
    probes = inside_sphere(gpu, [lamp])
    expected = 0.5 * 2.0 * godot_omni_attenuation(SPHERE_RADIUS, light_range, decay)
    for direction in ((0, 0, 1), (1, 0, 0), (0, -1, 0)):
        assert at(probes, (0, 0, 0), direction)[0] == pytest.approx(expected, rel=0.02), direction


def test_rt_update_spot_light_lights_its_cone_only(gpu):
    # Pointing along +Z, 60 degrees wide. Godot raises the rim to 1 / spot attenuation (here 20): every direction in
    # the +Z texel (within 35 degrees) is fully lit, and behind the light nothing is.
    lamp = Light(LIGHT_SPOT, color=(2.0, 2.0, 2.0), position=PROBE, direction=(0.0, 0.0, 1.0), range=3.0, attenuation=2.0,
                 spot_angle_deg=60.0, spot_attenuation=0.05)
    probes = inside_sphere(gpu, [lamp])
    full = 0.5 * 2.0 * godot_omni_attenuation(SPHERE_RADIUS, 3.0, 2.0)
    assert at(probes, (0, 0, 0), (0, 0, 1))[0] == pytest.approx(full, rel=0.02)
    assert at(probes, (0, 0, 0), (0, 0, -1))[0] == 0.0


@pytest.mark.parametrize("negative_color, share", [(-1.0, 0.5), (-4.0, 0.0)], ids=["subtracts", "stops_at_no_light"])
def test_rt_update_negative_light_subtracts(gpu, negative_color, share):
    lights = [Light(LIGHT_OMNI, color=(c, c, c), position=PROBE, range=3.0, attenuation=2.0) for c in (2.0, negative_color)]
    probes = inside_sphere(gpu, lights)
    expected = share * 0.5 * 2.0 * godot_omni_attenuation(SPHERE_RADIUS, 3.0, 2.0)
    assert at(probes, (0, 0, 0), (0, 0, 1))[0] == pytest.approx(expected, rel=0.02, abs=1e-3)


@pytest.mark.parametrize("bounce, expected", [(1.0, 0.4), (0.0, 0.0)], ids=["multi_bounce", "single_bounce"])
def test_rt_update_bounces_the_caches_irradiance_off_a_hit(gpu, bounce, expected):
    # No light reaches the wall, but the cache holds irradiance around it: it sends out albedo x that, the bounce that
    # lights surfaces only other surfaces light (Sponza's gallery undersides). Held to one bounce (Gate 2's
    # single-bounce reference), it sends out nothing. The wall is inside the grid, behind the camera.
    cam = Camera(width=32, height=24)
    irradiance = np.zeros(atlas_shape())
    irradiance[...] = (0.8, 0.8, 0.8, 1.0)
    probes = update_rt(gpu, cam, render(cam, []), Scene([wall_behind_camera(z=1.75)]), irradiance=irradiance, bounce=bounce)
    assert at(probes, (0, 0, 0), (0, 0, 1))[:3] == pytest.approx((expected,) * 3, abs=0.01)


def block_directions():
    """World direction of each texel of an 8 x 8 octahedral block, (y, x, 3)."""
    def decode(e):
        v = np.array([e[0], e[1], 1.0 - abs(e[0]) - abs(e[1])])
        if v[2] < 0:
            v[:2] = (1.0 - np.abs(v[[1, 0]])) * np.where(v[:2] >= 0, 1.0, -1.0)
        return v / np.linalg.norm(v)
    return np.array([[decode((np.array([x, y]) + 0.5) / WORLD_BLOCK * 2.0 - 1.0) for x in range(WORLD_BLOCK)]
                     for y in range(WORLD_BLOCK)])


def test_irradiance_of_uniform_radiance_is_that_radiance(gpu):
    probes = np.zeros(atlas_shape())
    probes[...] = (0.7, 0.3, 0.1, 1.0)
    irradiance = irradiance_of(gpu, probes)
    assert irradiance[..., :3] == pytest.approx(np.broadcast_to((0.7, 0.3, 0.1), irradiance[..., :3].shape), abs=0.003)
    assert np.all(irradiance[..., 3] == 1.0)


def test_irradiance_is_the_cosine_weighted_mean_over_the_normals_hemisphere(gpu):
    # Light from above only: a surface facing up gets all of it, one facing down none, one facing sideways half.
    from_above = (block_directions()[..., 1] > 0.0).astype(float)
    probes = np.zeros(atlas_shape())
    probes[..., 0] = np.tile(from_above, (N, N * N))
    probes[..., 3] = 1.0
    irradiance = irradiance_of(gpu, probes)
    assert at(irradiance, (0, 0, 0), (0, 1, 0))[0] > 0.9
    assert at(irradiance, (0, 0, 0), (0, -1, 0))[0] < 0.1
    assert at(irradiance, (0, 0, 0), (1, 0, 0))[0] == pytest.approx(0.5, abs=0.1)
    # Tilted up: a normal at elevation e gets (1 + sin e) / 2 of light from above under cosine weighting, 0.82 for the
    # texel nearest 45 degrees (its centre is at 40); weighting the hemisphere evenly would give 0.72.
    tilted = block_directions()[dir_texel((1, 1, 0))[::-1]]
    assert at(irradiance, (0, 0, 0), (1, 1, 0))[0] == pytest.approx((1.0 + tilted[1]) / 2.0, abs=0.03)


def test_irradiance_of_a_cell_new_this_frame_is_empty(gpu):
    # Its slot still holds the radiance of the cell that left.
    probes = np.zeros(atlas_shape())
    probes[...] = (1.0, 1.0, 1.0, 1.0)
    state = np.zeros((N, N * N, 4))
    state[1, 2 + 3 * N, 1] = 1.0   # slot (2, 1, 3) changed cells
    irradiance = irradiance_of(gpu, probes, state)
    x0, y0 = atlas_texel((2, 1, 3), (0, 0))
    new = irradiance[y0:y0 + WORLD_BLOCK, x0:x0 + WORLD_BLOCK]
    assert np.all(new == 0.0)
    assert np.all(irradiance[..., 0][irradiance[..., 3] > 0] == pytest.approx(1.0, abs=0.003))
    assert (irradiance[..., 3] > 0).sum() == (N ** 3 - 1) * WORLD_BLOCK ** 2


def test_rt_update_negative_light_darkens_the_bounce_too(gpu):
    # Godot sums a negative light into the same diffuse light the indirect light goes into, then clamps the total.
    irradiance = np.zeros(atlas_shape())
    irradiance[...] = (0.8, 0.8, 0.8, 1.0)
    shade = Light(LIGHT_OMNI, color=(-1.0, -1.0, -1.0), position=PROBE, range=3.0, attenuation=2.0)
    probes = inside_sphere(gpu, [shade], irradiance)
    expected = 0.5 * (0.8 - godot_omni_attenuation(SPHERE_RADIUS, 3.0, 2.0))
    assert at(probes, (0, 0, 0), (0, 0, 1))[0] == pytest.approx(expected, rel=0.02)
