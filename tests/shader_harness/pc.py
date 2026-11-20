"""Push-constant packers, field for field the same as the C++ structs (and the GLSL blocks).

tests/unit/test_push_constants.py already checks the sizes; these tests check the values land
where the shaders read them.
"""

import struct


def pack(fmt: str, *values) -> bytes:
    data = struct.pack("<" + fmt, *values)
    assert len(data) % 16 == 0, f"push constants should be 16-byte multiples, got {len(data)}"
    return data


def copy(w, h):
    return pack("ii2f", w, h, 0.0, 0.0)


def dilate(w, h, jitter_delta=(0.0, 0.0)):
    return pack("ii2f", w, h, *jitter_delta)


def reactive(w, h, sensitivity=1.0, spark=2.5, jitter_delta=(0.0, 0.0), depth_params=(0.05, 0.0), transparency_scale=1.0, flags=0):
    return pack("iiff2f2ffI2f", w, h, sensitivity, spark, *jitter_delta, *depth_params, transparency_scale, flags, 0.0, 0.0)


def taau(w, h, jitter=(0.0, 0.0), fb_min=0.05, fb_max=0.95, motion_scale=18.0, frame=0, flags=0):
    return pack("iiii2ffffIIf", w, h, w, h, *jitter, fb_min, fb_max, motion_scale, frame, flags, 0.0)


def fft_horiz(threshold=0.0, knee=0.5, exposure=1.0):
    return pack("4f", threshold, knee, exposure, 0.0)


def fft_vert(radius=1.0, diffraction=0.45, halation=0.35, blades=6.0, aspect=1.0):
    return pack("8f", radius, diffraction, halation, blades, aspect, 0.0, 0.0, 0.0)


def fft_inv(intensity=1.0):
    return pack("4f", intensity, 0.0, 0.0, 0.0)


def composite_chronos(w, h, exposure=1.0, sharpness=0.0, debug=0, flags=0, max_linear=0.0):
    return pack("iiffiI2f", w, h, exposure, sharpness, debug, flags, max_linear, 0.0)


def trace(w, h, probes, block, spacing, start, length, steps, thickness, frame, normal_bias, proj, depth_params, sub_rays):
    return pack("iiiiifffifIf4f2fif", w, h, probes[0], probes[1], block, spacing, start, length, steps,
                thickness, frame, normal_bias, *proj, *depth_params, sub_rays, 0.0)


def downsample(dst_w, dst_h):
    return pack("iiii", dst_w, dst_h, 0, 0)


def radiance_copy(w, h, depth_params):
    return pack("ii2f", w, h, *depth_params)


def merge(probes, parent_probes, block, parent_block, spacing, parent_spacing, resolution, depth_params):
    return pack("iiiiiiffii2f", *probes, *parent_probes, block, parent_block, spacing, parent_spacing, *resolution, *depth_params)


def integrate(w, h, probes, block, spacing, depth_params, proj):
    return pack("iiiiif2f4f", w, h, *probes, block, spacing, *depth_params, *proj)


def temporal(w, h, depth_params, flags=0, alpha_min=0.1, moments_alpha_min=0.2, max_history=32.0):
    return pack("ii2fIfff", w, h, *depth_params, flags, alpha_min, moments_alpha_min, max_history)


def atrous(w, h, step, phi_color=4.0, phi_normal=128.0, phi_depth=1.0):
    return pack("iiifff2f", w, h, step, phi_color, phi_normal, phi_depth, 0.0, 0.0)


def composite_lumina(w, h, intensity=1.0, debug=0, albedo_mode=0, max_history=32.0):
    return pack("iifiif2f", w, h, intensity, debug, albedo_mode, max_history, 0.0, 0.0)


def world(res, grid_n, block_or_screen_block, base_cell, a1w, cam_pos, spacing, right, thickness, up, alpha_min,
          proj, depth_params, misc_z, misc_w):
    """Shared 112-byte WorldPushConstants (lumina_world_common.glsl)."""
    return pack("4i4i4f4f4f4f4f",
                res[0], res[1], grid_n, block_or_screen_block,
                *base_cell, a1w,
                *cam_pos, spacing,
                *right, thickness,
                *up, alpha_min,
                *proj,
                *depth_params, misc_z, misc_w)
