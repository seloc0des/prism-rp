from pathlib import Path
import math
import sys

import numpy as np
import pytest

TESTS_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(TESTS_DIR / "reference"))
import godot_to_blender as g2b  # noqa: E402
import metrics  # noqa: E402


def test_vector_mapping_is_a_proper_rotation():
    m = np.array([g2b.vec(e) for e in np.eye(3)]).T     # columns = images of basis vectors
    assert np.allclose(m @ m.T, np.eye(3)) and np.isclose(np.linalg.det(m), 1.0)
    assert g2b.vec((0, 1, 0)) == (0, 0, 1)                # Godot up -> Blender up


def test_identity_godot_camera_looks_along_blender_plus_y():
    mat = np.array(g2b.camera_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 2, 5]))
    forward = -mat[:3, 2]                                 # both engines' cameras look down local -Z
    up = mat[:3, 1]
    assert np.allclose(forward, (0, 1, 0)) and np.allclose(up, (0, 0, 1))
    assert np.allclose(mat[:3, 3], (0, -5, 2))


def test_winding_flip_keeps_godot_front_face_outward():
    # Godot CW front face seen from +Y (a floor): front normal = (c - a) x (b - a).
    a, b, c = np.array([0, 0, 0.0]), np.array([1, 0, 0.0]), np.array([0, 0, 1.0])
    godot_front = np.cross(c - a, b - a)
    assert np.allclose(godot_front / np.linalg.norm(godot_front), (0, 1, 0))
    tri = g2b.triangles_ccw([0, 1, 2])[0]
    pts = [np.array(g2b.vec(p)) for p in (a, b, c)]
    blender_normal = np.cross(pts[tri[1]] - pts[tri[0]], pts[tri[2]] - pts[tri[0]])   # Blender: CCW right-hand
    assert np.allclose(blender_normal / np.linalg.norm(blender_normal), g2b.vec((0, 1, 0)))


def test_sun_rotation_points_minus_z_along_direction():
    d = np.array(g2b.vec((0.3, -1.0, -0.2)))
    rot = np.array(g2b.look_rotation_matrix(tuple(d)))
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-9)
    assert np.allclose(-rot[:, 2], d / np.linalg.norm(d))


def test_uv_flips_v():
    assert g2b.uv(0.25, 0.0) == (0.25, 1.0) and g2b.uv(1.0, 1.0) == (1.0, 0.0)


def test_light_unit_conversions():
    assert g2b.omni_power_watts(1.0) == pytest.approx(4 * math.pi ** 2)
    assert g2b.sun_strength(2.5) == 2.5


def test_metrics():
    rng = np.random.default_rng(0)
    ref = rng.random((16, 16, 3)) * 0.8
    assert metrics.psnr(ref, ref) == float("inf")
    m = metrics.gate2_metrics(ref * 0.5, ref)            # right shape, wrong units
    assert m["fitted_scale"] == pytest.approx(2.0)
    assert m["psnr_fitted_db"] > 80 and m["psnr_db"] < 25
    assert metrics.srgb_to_linear(np.array([0.0, 1.0]))[1] == pytest.approx(1.0)
    assert metrics.srgb_to_linear(np.array([0.5]))[0] == pytest.approx(0.2140, abs=1e-3)
