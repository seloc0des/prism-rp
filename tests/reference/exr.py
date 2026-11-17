"""Read one pass from a Blender multilayer EXR (needs `pip install OpenEXR`)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_pass(path: str | Path, pass_name: str = "DiffInd") -> np.ndarray:
    """Returns the pass as float32 (H, W, 3), top row first. Channels are named
    "<view layer>.<pass>.<R|G|B>" in Blender's multilayer EXRs."""
    import OpenEXR

    with OpenEXR.File(str(path), separate_channels=True) as f:
        channels = f.channels()
        rgb = []
        for c in "RGB":
            matches = [k for k in channels if k.endswith(f".{pass_name}.{c}") or k == f"{pass_name}.{c}"]
            if not matches:
                raise KeyError(f"pass {pass_name!r} not in {path}; channels: {sorted(channels)[:12]}...")
            rgb.append(np.asarray(channels[matches[0]].pixels, dtype=np.float32))
    return np.stack(rgb, axis=-1)
