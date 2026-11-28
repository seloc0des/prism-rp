"""Lumina's albedo rays on lavapipe (lumina_rt_albedo): one camera ray per pixel through the scene mirror, which the
screen bounce reflects light off and the Ray-Traced Albedo view shows. The ray looks for the surface Godot drew at
the pixel only near the depth buffer's depth there. The camera is at the origin looking down -Z, as in scene.render;
a = 0 is a pixel the ray left alone (the bounce keeps the proxy albedo there)."""

import numpy as np
import pytest

import pc
from harness import Dispatch, Img, groups_for
from mirror import Material, Mesh, Scene, quad
from scene import Camera, Plane, render

ALBEDO = (0.2, 0.4, 0.6)
WALL = Mesh(quad((0.0, 0.0, -3.0), (0.0, 0.0, 1.0), 40.0), Material(albedo=ALBEDO))
WALL_PLANE = Plane((0.0, 0.0, -3.0), (0.0, 0.0, 1.0), (1.0, 1.0, 1.0))


def trace_albedo(gpu, cam, g, scene):
    images = {"albedo": Img("rgba16f", data=np.zeros((cam.height, cam.width, 4))), "depth": Img("r32f", data=g["depth"]),
              **scene.images()}
    dispatch = Dispatch("lumina/lumina_rt_albedo.comp.glsl", {0: ("albedo", "storage"), 1: ("depth", "sampled_nearest"), **scene.bindings()},
                        pc.rt_albedo(cam.width, cam.height, cam.proj, cam.depth_params), groups_for(cam.width, cam.height))
    return gpu.run(images, [dispatch], ["albedo"], buffers=scene.buffers(), acceleration_structures=scene.acceleration_structures())["albedo"]


def test_albedo_rays_find_the_surface_godot_drew(gpu):
    cam = Camera(width=32, height=24)
    out = trace_albedo(gpu, cam, render(cam, [WALL_PLANE]), Scene([WALL]))
    assert (out[..., 3] == 1.0).all()
    np.testing.assert_allclose(out[..., :3], np.broadcast_to(ALBEDO, out[..., :3].shape), rtol=2e-3)


def test_albedo_rays_miss_where_the_mirror_lacks_what_godot_drew(gpu):
    # Godot draws a card in front of the wall that the mirror leaves out (as it does a mesh with gi_mode disabled).
    # A ray from the near plane to the far plane went past the card and gave its pixels the wall's albedo.
    cam = Camera(width=32, height=24)
    card = Plane((0.0, 0.0, -2.0), (0.0, 0.0, 1.0), (1.0, 1.0, 1.0), bounds=((-0.4, 0.4), (-0.4, 0.4), (-2.1, -1.9)))
    g = render(cam, [WALL_PLANE, card])
    on_card = np.isclose(g["linear_z"], 2.0)
    assert on_card.sum() > 20 and (~on_card).sum() > 20
    out = trace_albedo(gpu, cam, g, Scene([WALL]))
    assert (out[on_card] == 0.0).all()
    assert (out[~on_card, 3] == 1.0).all()


def test_albedo_rays_leave_the_sky_alone(gpu):
    # Godot drew nothing; the wall in the mirror isn't what the pixel shows.
    cam = Camera(width=32, height=24)
    out = trace_albedo(gpu, cam, render(cam, []), Scene([WALL]))
    assert (out == 0.0).all()


@pytest.mark.parametrize("depth_off", [0.99, 1.01])
def test_albedo_rays_find_the_surface_a_little_off_the_depth_buffer(gpu, depth_off):
    # The raster and the mirror can disagree by a little (a lower level of detail, depth precision): a surface 1%
    # nearer or farther than the depth buffer says is still the one Godot drew.
    cam = Camera(width=32, height=24)
    g = render(cam, [WALL_PLANE])
    g["depth"] = np.where(g["depth"] > 0.0, cam.depth_from_linear(g["linear_z"] * depth_off), 0.0)
    out = trace_albedo(gpu, cam, g, Scene([WALL]))
    assert (out[..., 3] == 1.0).all()
