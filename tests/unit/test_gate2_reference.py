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
    assert g2b.sun_strength(2.5) == pytest.approx(2.5 * math.pi)   # Godot scales directional energy by pi


def test_authored_colours_are_decoded_from_srgb_like_godot():
    # Godot decodes authored colours with Color::srgb_to_linear() before lighting; the reference must too.
    assert g2b.linear((0.5, 0.5, 0.5, 1.0)) == pytest.approx((0.214041,) * 3, abs=1e-6)
    assert g2b.linear((0.0, 0.02, 1.0)) == pytest.approx((0.0, 0.02 / 12.92, 1.0))
    assert g2b.linear((0.8, 0.1, 0.1)) == pytest.approx((0.603827, 0.010023, 0.010023), abs=1e-6)


def test_metrics():
    rng = np.random.default_rng(0)
    ref = rng.random((16, 16, 3)) * 0.8
    assert metrics.psnr(ref, ref) == float("inf")
    m = metrics.gate2_metrics(ref * 0.5, ref)            # right shape, wrong units
    assert m["fitted_scale"] == pytest.approx(2.0)
    assert m["psnr_fitted_db"] > 80 and m["psnr_db"] < 25
    assert metrics.srgb_to_linear(np.array([0.0, 1.0]))[1] == pytest.approx(1.0)
    assert metrics.srgb_to_linear(np.array([0.5]))[0] == pytest.approx(0.2140, abs=1e-3)


def test_metrics_on_a_float_capture_are_unclipped():
    ref = np.full((20, 20, 3), 0.5)
    ref[:5] = 3.0                                       # bright bounce light near an emitter
    m = metrics.gate2_metrics(ref.copy(), ref, clipped_test=False)
    assert m["psnr_fitted_db"] == float("inf") and m["fitted_scale"] == pytest.approx(1.0)
    clipped = metrics.gate2_metrics(np.clip(ref, 0, 1), ref)   # 8-bit path: the reference is clipped too
    assert clipped["psnr_fitted_db"] == float("inf")
    edges = ref.copy()
    edges[:, :2] = 0.0                                  # error only at the frame's edge
    e = metrics.gate2_metrics(edges, ref, clipped_test=False)
    assert e["psnr_center_db"] == float("inf") and e["psnr_fitted_db"] < 30
    assert e["mean_reference"] == pytest.approx(float(ref.mean()))


def test_indirect_adds_emitter_light_when_rendered(tmp_path, monkeypatch):
    import exr
    images = {"full": np.full((2, 2, 3), 0.7), "direct": np.full((2, 2, 3), 0.5), "emitters": np.full((2, 2, 3), 0.25)}
    monkeypatch.setattr(exr, "load_rgb", lambda path: images[str(path).rsplit("_", 1)[1].split(".")[0]])
    prefix = tmp_path / "ref"
    np.testing.assert_allclose(exr.load_indirect(prefix), 0.2)        # no emitters file: lights' bounce only
    (tmp_path / "ref_emitters.exr").write_bytes(b"")
    np.testing.assert_allclose(exr.load_indirect(prefix), 0.45)       # plus the emitters' direct light


def test_gate2_results_serialize_and_compare_image(tmp_path):
    pytest.importorskip("flip_evaluator")   # run_gate2 imports run_visual_regression
    import json
    import run_gate2
    from PIL import Image
    row = run_gate2.jsonable({"passed": np.bool_(False), "psnr": np.float64(21.4), "errors": [np.int64(3)]})
    assert json.loads(json.dumps(row)) == {"passed": False, "psnr": 21.4, "errors": [3]}
    ref = np.random.default_rng(1).random((6, 8, 3))
    run_gate2.write_comparison(tmp_path / "compare.png", ref, ref * 0.5, 2.0)
    assert Image.open(tmp_path / "compare.png").size == (24, 6)


def test_settling_error_and_frame_count():
    pytest.importorskip("flip_evaluator")   # check_gi_convergence imports run_visual_regression
    import check_gi_convergence as gc
    settled = np.full((4, 4, 3), 0.5)
    frames = [settled * f for f in (0.0, 0.5, 0.8, 0.95, 1.0)]
    errors = gc.settling_errors(frames, settled)
    assert errors == pytest.approx([1.0, 0.5, 0.2, 0.05, 0.0])
    assert gc.frames_to_settle(errors, 0.10) == 3
    assert gc.frames_to_settle(errors[:3], 0.10) is None


def test_gate2_captures_share_a_matte_scene_and_measure_sdfgi_in_colour():
    # Godot's lighting-only debug view swaps every material for its grey default, also in the pass where SDFGI
    # captures colours, so Gate 2 measured SDFGI in a colourless scene. No capture may use it, and Lumina, SDFGI
    # and the albedo capture must all render the reference's scene: matte, no emission, no environment light, and
    # one GI bounce (SDFGI's default bounce feedback computes several).
    pytest.importorskip("flip_evaluator")
    import run_gate2
    on = run_gate2.sdfgi_capture_args(Path("x.exr"), True)
    off = run_gate2.sdfgi_capture_args(Path("x.exr"), False)
    lumina = run_gate2.lumina_capture_args(Path("e"), Path("x.exr"))
    albedo = run_gate2.albedo_capture_args(Path("a.exr"))
    for args in (on, off, lumina, albedo):
        assert not any(a.startswith("--prism-debug-draw=lighting") for a in args)
        for flag in ("--prism-chronos=off", "--prism-ambient=off", "--prism-ssr=off", "--prism-emission=off",
                     "--prism-diffuse-only", "--prism-no-jitter", "--prism-gi-single-bounce"):
            assert flag in args, (flag, args)
    assert "--prism-sdfgi=on" in on and "--prism-sdfgi=off" in off and "--prism-lumina=off" in on
    assert "--prism-debug-draw=unshaded" in albedo and "--prism-sdfgi=off" in albedo
