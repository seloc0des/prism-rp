"""Chronos compute shaders run on lavapipe with synthetic inputs and checked numerically."""

import numpy as np
import pytest

import pc
from harness import Dispatch, Img, groups_for

RNG = np.random.default_rng(1234)


def test_copy_is_exact(gpu):
    src = RNG.random((21, 35, 4)) * 8.0
    out = gpu.run({"src": Img("rgba16f", data=src), "dst": Img("rgba16f", 35, 21)},
                  [Dispatch("util/prism_copy.comp.glsl", {0: ("src", "sampled_nearest"), 1: ("dst", "storage")},
                            pc.copy(35, 21), groups_for(35, 21))], ["dst"])["dst"]
    np.testing.assert_array_equal(out, src.astype(np.float16).astype(np.float32))


def _dilate(gpu, depth, velocity, jitter_delta=(0.0, 0.0)):
    h, w = depth.shape
    return gpu.run(
        {"depth": Img("r32f", data=depth), "vel": Img("rg16f", data=velocity), "out": Img("rg16f", w, h)},
        [Dispatch("chronos/chronos_dilate_velocity.comp.glsl",
                  {0: ("depth", "sampled_nearest"), 1: ("vel", "sampled_nearest"), 2: ("out", "storage")},
                  pc.dilate(w, h, jitter_delta), groups_for(w, h))], ["out"])["out"]


def test_dilation_takes_velocity_of_closest_surface(gpu):
    # Reverse-Z: the near object has the larger depth. A single near pixel at (5, 5) moving left;
    # everything else far and static. Its 3x3 neighbourhood must inherit its velocity.
    depth = np.full((12, 12), 0.1)
    depth[5, 5] = 0.9
    vel = np.zeros((12, 12, 2))
    vel[5, 5] = (-0.05, 0.0)
    out = _dilate(gpu, depth, vel)
    np.testing.assert_allclose(out[4:7, 4:7, 0], -0.05, atol=1e-3)
    assert np.all(np.abs(out[0:3, 0:3]) < 1e-6)


def test_dilation_at_screen_edge_reads_inside_the_image(gpu):
    # Closest surface at the corner pixel itself: offsets that point outside the image must not be
    # used to fetch velocity (the original code fetched at pixel + offset, unclamped).
    depth = np.full((8, 8), 0.1)
    depth[0, 0] = 0.9
    vel = np.zeros((8, 8, 2))
    vel[0, 0] = (0.02, 0.03)
    out = _dilate(gpu, depth, vel)
    np.testing.assert_allclose(out[0, 0], (0.02, 0.03), atol=1e-3)
    np.testing.assert_allclose(out[1, 1], (0.02, 0.03), atol=1e-3)


def test_dilation_adds_jitter_delta(gpu):
    depth = np.full((8, 8), 0.5)
    vel = np.zeros((8, 8, 2))
    out = _dilate(gpu, depth, vel, jitter_delta=(0.001, -0.002))
    np.testing.assert_allclose(out[..., 0], 0.001, atol=1e-4)
    np.testing.assert_allclose(out[..., 1], -0.002, atol=1e-4)


def _taau(gpu, current, history, velocity, reactive=None, flags=0, fb_min=0.95, fb_max=0.95):
    h, w = current.shape[:2]
    reactive = np.zeros((h, w)) if reactive is None else reactive
    return gpu.run(
        {
            "cur": Img("rgba16f", data=current), "hist": Img("rgba16f", data=history),
            "depth": Img("r32f", data=np.full((h, w), 0.5)), "vel": Img("rg16f", data=velocity),
            "react": Img("r8", data=reactive), "out": Img("rgba16f", w, h),
        },
        [Dispatch("chronos/chronos_taau.comp.glsl",
                  {0: ("cur", "sampled_nearest"), 1: ("hist", "sampled_linear"), 2: ("depth", "sampled_nearest"),
                   3: ("vel", "sampled_nearest"), 4: ("react", "sampled_linear"), 5: ("out", "storage")},
                  pc.taau(w, h, fb_min=fb_min, fb_max=fb_max, flags=flags), groups_for(w, h))], ["out"])["out"]


def _pattern(h, w, shift_x=0):
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    x = x + shift_x
    v = 0.5 + 0.4 * np.sin(x * 0.7) * np.cos(y * 0.45)
    rgb = np.stack([v, 1.0 - v, 0.5 * v + 0.2], axis=-1)
    return np.concatenate([rgb, np.ones((h, w, 1))], axis=-1)


def test_taa_reset_outputs_current_frame(gpu):
    cur = _pattern(16, 16)
    out = _taau(gpu, cur, np.zeros_like(cur), np.zeros((16, 16, 2)), flags=1)
    # Reconstruction filter softens slightly; it must not pull in the (black) history.
    assert abs(out[..., :3].mean() - cur[..., :3].mean()) < 0.02


def test_taa_reprojects_with_godot_velocity_sign(gpu):
    """History holds the scene shifted by +3 px; Godot's velocity (prev_uv - curr_uv) is +3 px.

    With prev = uv + velocity the history lines up with the current frame; with the old
    uv - velocity it is off by 6 px and the blend error is much larger.
    """
    w = h = 32
    shift = 3
    cur = _pattern(h, w)
    hist = _pattern(h, w, shift_x=-shift)    # hist(x + 3) == cur(x)
    hist[..., 3] = 1.0                        # full confidence
    vel = np.zeros((h, w, 2))
    vel[..., 0] = shift / w

    good = _taau(gpu, cur, hist, vel)
    flipped = _taau(gpu, cur, hist, -vel)
    inner = (slice(4, -4), slice(8, -8))
    err_good = np.abs(good[inner][..., :3] - cur[inner][..., :3]).mean()
    err_flip = np.abs(flipped[inner][..., :3] - cur[inner][..., :3]).mean()
    assert err_good < 0.03
    assert err_flip > 2.0 * err_good


def test_taa_reactive_pixels_drop_history(gpu):
    # The current frame needs local variation, or the neighbourhood clamp alone would reject
    # the history in both runs. History sits inside that range, so only feedback decides.
    cur = _pattern(16, 16)
    hist = np.zeros((16, 16, 4)); hist[..., :3] = 0.5; hist[..., 3] = 1
    vel = np.zeros((16, 16, 2))
    err_static = np.abs(_taau(gpu, cur, hist, vel)[4:12, 4:12, :3] - cur[4:12, 4:12, :3]).mean()
    err_reactive = np.abs(_taau(gpu, cur, hist, vel, reactive=np.ones((16, 16)))[4:12, 4:12, :3] - cur[4:12, 4:12, :3]).mean()
    assert err_reactive < 0.5 * err_static


def _bloom(gpu, src, vert_pc, threshold=0.0, intensity=1.0):
    h, w = src.shape[:2]
    imgs = {
        "src": Img("rgba16f", data=src),
        "ping_rg": Img("rgba32f", 256, 256), "ping_b": Img("rgba32f", 256, 256),
        "pong_rg": Img("rgba32f", 256, 256), "pong_b": Img("rgba32f", 256, 256),
        "bloom": Img("rgba16f", 256, 256),
    }
    return gpu.run(imgs, [
        Dispatch("chronos/chronos_fft_bloom_horizontal.comp.glsl",
                 {0: ("src", "sampled_linear"), 1: ("ping_rg", "storage"), 2: ("ping_b", "storage")},
                 pc.fft_horiz(threshold=threshold), (1, 256, 1)),
        Dispatch("chronos/chronos_fft_bloom_vertical.comp.glsl",
                 {0: ("ping_rg", "storage"), 1: ("ping_b", "storage"), 2: ("pong_rg", "storage"), 3: ("pong_b", "storage")},
                 vert_pc, (256, 1, 1)),
        Dispatch("chronos/chronos_fft_bloom_inv_horizontal.comp.glsl",
                 {0: ("pong_rg", "storage"), 1: ("pong_b", "storage"), 2: ("bloom", "storage")},
                 pc.fft_inv(intensity), (1, 256, 1)),
    ], ["bloom"])["bloom"]


def _blob(size=224, value=4.0, radius=3):
    src = np.zeros((size, size, 4)); src[..., 3] = 1
    c = size // 2
    src[c - radius:c + radius, c - radius:c + radius, :3] = value
    return src


def test_fft_bloom_core_conserves_energy(gpu):
    # Core kernel only: the forward/inverse FFT pair and its 1/N^2 normalisation must give back
    # exactly the input energy (OTF(0) = 1). A grey source stays grey in ACEScg.
    src = _blob()
    out = _bloom(gpu, src, pc.fft_vert(diffraction=0.0, halation=0.0))
    for ch in range(3):
        assert out[..., ch].sum() == pytest.approx(src[..., 0].sum(), rel=0.005), ch


def test_fft_bloom_full_kernel_energy_and_tint(gpu):
    # Full kernel: DC gain is 1 + d + h (per-channel weights as in evaluate_optical_otf). The
    # diffraction spikes have negative spatial lobes that the final max(0) clips, which can only add
    # energy; it adds roughly 15-20% here.
    d, h = 0.45, 0.35
    src = _blob()
    out = _bloom(gpu, src, pc.fft_vert(diffraction=d, halation=h))
    total_in = src[..., 0].sum()
    gains = [1 + d + h, 1 + 0.96 * d + 0.14 * h, 1 + 0.92 * d + 0.02 * h]
    sums = [out[..., ch].sum() for ch in range(3)]
    for ch, gain in enumerate(gains):
        assert total_in * gain * 0.99 <= sums[ch] <= total_in * gain * 1.25, (ch, sums[ch], total_in * gain)
    assert sums[0] > sums[1] > sums[2]   # red halation


def test_fft_bloom_is_point_symmetric_for_odd_blades(gpu):
    # A real, even OTF gives a point-symmetric PSF. Odd blade counts used to break this.
    src = _blob(value=8.0, radius=1)
    out = _bloom(gpu, src, pc.fft_vert(blades=5.0))[..., 1]
    c = 16 + 112   # blob centre in grid coordinates (16 px padding)
    patch = out[c - 40:c + 40, c - 40:c + 40]
    np.testing.assert_allclose(patch, patch[::-1, ::-1], atol=0.02 * patch.max())


def test_fft_bloom_aspect_narrows_horizontally(gpu):
    src = _blob(value=8.0, radius=1)
    def widths(img):
        g = img[..., 1]
        c = 16 + 112
        row, col = g[c, c - 60:c + 60], g[c - 60:c + 60, c]
        xs = np.arange(-60, 60)
        return np.sqrt((row * xs**2).sum() / row.sum()), np.sqrt((col * xs**2).sum() / col.sum())
    wx1, wy1 = widths(_bloom(gpu, src, pc.fft_vert(aspect=1.0, diffraction=0.0)))
    wx2, wy2 = widths(_bloom(gpu, src, pc.fft_vert(aspect=2.0, diffraction=0.0)))
    assert wx1 == pytest.approx(wy1, rel=0.1)
    assert wx2 < 0.75 * wy2


def _composite(gpu, src, flags, exposure=1.0, debug=0, max_linear=0.0):
    h, w = src.shape[:2]
    out = gpu.run(
        {"res": Img("rgba16f", data=src), "bloom": Img("rgba16f", 256, 256), "vel": Img("rg16f", w, h),
         "react": Img("r8", w, h), "out": Img("rgba16f", w, h)},
        [Dispatch("chronos/chronos_aces2_odt.comp.glsl",
                  {0: ("res", "sampled_nearest"), 1: ("bloom", "sampled_linear"), 2: ("vel", "sampled_nearest"),
                   3: ("react", "sampled_nearest"), 4: ("out", "storage")},
                  pc.composite_chronos(w, h, exposure=exposure, flags=flags, debug=debug, max_linear=max_linear), groups_for(w, h))], ["out"])
    return out["out"]


def test_composite_passthrough_roundtrips_color(gpu):
    src = np.concatenate([RNG.random((8, 8, 3)) * 4.0, np.ones((8, 8, 1))], axis=-1)
    out = _composite(gpu, src, flags=0)
    np.testing.assert_allclose(out[..., :3], src[..., :3], rtol=2e-3, atol=2e-3)
    out2 = _composite(gpu, src, flags=0, exposure=2.0)
    np.testing.assert_allclose(out2[..., :3], 2.0 * src[..., :3], rtol=3e-3, atol=3e-3)


def test_sdr_look_is_monotonic_and_in_range(gpu):
    lum = np.geomspace(1e-4, 100.0, 64)
    src = np.zeros((1, 64, 4)); src[0, :, :3] = lum[:, None]; src[..., 3] = 1
    out = _composite(gpu, src, flags=2)[0, :, 0]
    assert np.all(np.diff(out) >= -1e-3)
    assert out.min() >= 0.0 and out.max() <= 1.0
    mid = out[np.argmin(np.abs(lum - 0.18))]
    assert 0.05 < mid < 0.3      # mid grey lands in a sane display-linear range
    assert out[-1] > 0.9         # very bright values approach white


def test_sdr_look_handles_saturated_and_out_of_gamut_colors(gpu):
    cols = np.array([[8, 0, 0], [0, 8, 0], [0, 0, 8], [4, 4, 0], [1e-3, 0, 0], [0.0, 0.0, 0.0]], dtype=np.float64)
    src = np.concatenate([cols[None], np.ones((1, len(cols), 1))], axis=-1)
    out = _composite(gpu, src, flags=2)
    assert np.all(np.isfinite(out)) and out.min() >= 0.0 and out[..., :3].max() <= 1.0


def test_signed_velocity_debug_view_encodes_sign_and_size(gpu):
    """Debug view 5 (tests/check_motion_vectors.py reads it): rg = 0.5 + velocity * 10."""
    h, w = 4, 8
    vel = np.stack(np.meshgrid(np.linspace(-0.03, 0.03, w), np.linspace(0.02, -0.02, h)), axis=-1)
    out = gpu.run(
        {"res": Img("rgba16f", data=np.ones((h, w, 4))), "bloom": Img("rgba16f", 256, 256), "vel": Img("rg16f", data=vel),
         "react": Img("r8", w, h), "out": Img("rgba16f", w, h)},
        [Dispatch("chronos/chronos_aces2_odt.comp.glsl",
                  {0: ("res", "sampled_nearest"), 1: ("bloom", "sampled_linear"), 2: ("vel", "sampled_nearest"),
                   3: ("react", "sampled_nearest"), 4: ("out", "storage")},
                  pc.composite_chronos(w, h, debug=5), groups_for(w, h))], ["out"])["out"]
    vel16 = vel.astype(np.float16).astype(np.float64)
    np.testing.assert_allclose(out[..., :2], 0.5 + vel16 * 10.0, atol=1e-3)
    np.testing.assert_allclose(out[..., 2], 0.5, atol=1e-3)


def _luminance_ramp(n=96):
    lum = np.geomspace(1e-4, 100.0, n)
    src = np.zeros((1, n, 4)); src[0, :, :3] = lum[:, None]; src[..., 3] = 1
    return lum, src


def test_sdr_look_unchanged_on_sdr_displays(gpu):
    """No HDR (0, the old padding value) and a peak of 1.0 must both give exactly today's SDR output."""
    _, src = _luminance_ramp()
    sdr = _composite(gpu, src, flags=2)
    np.testing.assert_array_equal(_composite(gpu, src, flags=2, max_linear=1.0), sdr)
    np.testing.assert_array_equal(_composite(gpu, src, flags=2, max_linear=0.5), sdr)


def test_sdr_look_extends_highlights_on_hdr_displays(gpu):
    lum, src = _luminance_ramp()
    sdr = _composite(gpu, src, flags=2)[0, :, 0].astype(float)
    hdr = _composite(gpu, src, flags=2, max_linear=4.0)[0, :, 0].astype(float)
    assert np.all(np.diff(hdr) >= -1e-3)                       # still monotonic
    assert hdr.max() <= 4.0 + 1e-3 and hdr[-1] > 3.0           # reaches towards the display peak
    shadows = lum <= 0.05
    np.testing.assert_allclose(hdr[shadows], sdr[shadows], rtol=5e-3, atol=1e-4)   # shadows where SDR puts them
    mid = np.argmin(np.abs(lum - 0.18))
    assert hdr[mid] == pytest.approx(sdr[mid], rel=5e-3)       # mid grey unchanged
    assert np.all(hdr >= sdr - 1e-3) and hdr[lum >= 5.0].min() > 1.0   # highlights only ever brighter


def test_hdr_look_handles_saturated_and_out_of_gamut_colors(gpu):
    cols = np.array([[80, 0, 0], [0, 80, 0], [0, 0, 80], [40, 40, 0], [1e-3, 0, 0], [0.0, 0.0, 0.0]], dtype=np.float64)
    src = np.concatenate([cols[None], np.ones((1, len(cols), 1))], axis=-1)
    out = _composite(gpu, src, flags=2, max_linear=6.0)
    assert np.all(np.isfinite(out)) and out.min() >= 0.0 and out[..., :3].max() <= 6.0 + 1e-3


def _reactive(gpu, color, prev, depth, opaque=None, depth_params=(0.1, 0.001), transparency_scale=1.0):
    h, w = color.shape[:2]
    flags = 1 if opaque is not None else 0
    imgs = {
        "color": Img("rgba16f", data=color), "prev": Img("rgba16f", data=prev), "depth": Img("r32f", data=depth),
        "vel": Img("rg16f", w, h), "out": Img("r8", w, h),
        "opaque": Img("rgba16f", data=opaque if opaque is not None else color),
    }
    return gpu.run(imgs, [Dispatch(
        "chronos/chronos_reactive_mask.comp.glsl",
        {0: ("color", "sampled_nearest"), 1: ("prev", "sampled_linear"), 2: ("depth", "sampled_nearest"),
         3: ("vel", "sampled_nearest"), 4: ("out", "storage"), 5: ("opaque", "sampled_nearest")},
        pc.reactive(w, h, depth_params=depth_params, transparency_scale=transparency_scale, flags=flags),
        groups_for(w, h))], ["out"])["out"][..., 0]


def _flat(h, w, value):
    a = np.zeros((h, w, 4)); a[..., :3] = value; a[..., 3] = 1
    return a


def test_reactive_static_scene_is_calm(gpu):
    c = _flat(16, 16, 0.3)
    out = _reactive(gpu, c, c, np.full((16, 16), 0.5))
    assert out.max() < 0.05


def test_reactive_flags_pixels_changed_by_transparents(gpu):
    opaque = _flat(16, 16, 0.3)
    final = opaque.copy()
    final[4:8, 4:8, :3] = (1.5, 0.8, 0.2)          # a spark / glass drawn over the opaque frame
    depth = np.full((16, 16), 0.5)
    out = _reactive(gpu, final, final, depth, opaque=opaque)   # same as last frame: only transparency explains it
    assert out[5:7, 5:7].min() > 0.6
    assert out[12:, 12:].max() < 0.05
    # Without the opaque capture the same pixels stay calm (that's the ghosting the capture fixes).
    out_no_capture = _reactive(gpu, final, final, depth)
    assert out_no_capture[5:7, 5:7].max() < 0.1


def test_reactive_depth_edges_scale_with_distance(gpu):
    """A 50% depth step is flagged the same near the camera and far away (linearised depth)."""
    near, far = 0.1, 100.0
    a, b = far * near / (far - near), near / (far - near)
    def depth_for(z):
        return a / z - b
    out = []
    for z in (1.0, 40.0):
        d = np.full((16, 16), depth_for(z))
        d[:, 8:] = depth_for(z * 1.5)
        out.append(_reactive(gpu, _flat(16, 16, 0.3), _flat(16, 16, 0.3), d, depth_params=(a, b)))
    edge_near, edge_far = out[0][:, 7:9].mean(), out[1][:, 7:9].mean()
    assert edge_near > 0.3
    assert edge_far == pytest.approx(edge_near, abs=0.05)
