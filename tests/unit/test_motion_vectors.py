"""tests/check_motion_vectors.py's expected motion vectors, checked against projecting actual 3D points with
each camera (no shared formula), plus the parsing of what the benchmark rig logs."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("flip_evaluator")  # check_motion_vectors imports run_visual_regression, which needs it
import check_motion_vectors as mv  # noqa: E402


def look_basis(forward, up=(0.0, 1.0, 0.0)):
    """Godot camera basis: columns right, up, back (the camera looks down -Z)."""
    back = -np.asarray(forward, float) / np.linalg.norm(forward)
    right = np.cross(up, back)
    right /= np.linalg.norm(right)
    return np.column_stack([right, np.cross(back, right), back])


def yaw(basis, degrees):
    a = np.radians(degrees)
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])  # Node3D.rotate_y
    return ry @ basis


def project(basis, point, fov_deg, aspect, keep_aspect):
    """uv (y down) of a world point (camera at the origin), the way Godot's projection maps it."""
    p = basis.T @ point
    t = np.tan(np.radians(fov_deg) / 2)
    tx, ty = (t, t / aspect) if keep_aspect == mv.KEEP_WIDTH else (t * aspect, t)
    return np.array([(p[0] / -p[2] / tx) * 0.5 + 0.5, (-p[1] / -p[2] / ty) * 0.5 + 0.5])


@pytest.mark.parametrize("keep_aspect", [0, 1])
def test_expected_velocity_matches_projected_points(keep_aspect):
    w, h, fov = 160, 90, 50.0
    prev = look_basis((0.1, -0.2, -1.0))
    curr = yaw(prev, 0.8)
    field = mv.expected_velocity(w, h, fov, keep_aspect, curr, prev)
    rng = np.random.default_rng(0)
    for _ in range(50):
        # A point seen at a random pixel of the current frame, at a random depth (rotation: depth doesn't matter)
        px, py = rng.integers(0, w), rng.integers(0, h)
        u, v = (px + 0.5) / w, (py + 0.5) / h
        t = np.tan(np.radians(fov) / 2)
        tx, ty = (t, t * h / w) if keep_aspect == mv.KEEP_WIDTH else (t * w / h, t)
        point = curr @ (np.array([(2 * u - 1) * tx, -(2 * v - 1) * ty, -1.0]) * rng.uniform(0.5, 50.0))
        assert np.allclose(project(curr, point, fov, w / h, keep_aspect), [u, v], atol=1e-9)
        expected = project(prev, point, fov, w / h, keep_aspect) - np.array([u, v])  # Godot: prev_uv - curr_uv
        np.testing.assert_allclose(field[py, px], expected, atol=1e-9)


def test_turning_left_moves_the_scene_right():
    """rotate_y(+) turns the camera left, so the scene moves right: curr_uv > prev_uv, velocity.x < 0."""
    prev = look_basis((0.0, 0.0, -1.0))
    field = mv.expected_velocity(64, 64, 60.0, 1, yaw(prev, 1.0), prev)
    assert np.all(field[..., 0] < 0)
    assert abs(field[32, 32, 1]) < 1e-3


def test_no_rotation_no_motion():
    b = look_basis((0.3, -0.1, -1.0))
    assert np.abs(mv.expected_velocity(32, 18, 50.0, 1, b, b)).max() < 1e-12


def test_parses_the_rig_log():
    line = ("[Prism RP QA] motion-test camera fov=50.000000 keep_aspect=1 "
            "curr=[1.000000,0.000000,0.000000;0.000000,0.995037,0.099504;0.000000,-0.099504,0.995037] "
            "prev=[0.999962,0.000000,-0.008727;0.000868,0.995037,0.099500;0.008684,-0.099504,0.995000]")
    m = mv.CAMERA_RE.search(line)
    assert m and float(m.group(1)) == 50.0 and int(m.group(2)) == 1
    curr = mv.parse_basis(m.group(3))
    np.testing.assert_allclose(curr[:, 1], [0.0, 0.995037, 0.099504])   # columns are the basis vectors
    np.testing.assert_allclose(curr[:, 2], [0.0, -0.099504, 0.995037])
