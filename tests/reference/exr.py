"""Read Blender EXR renders for Gate 2 (needs `pip install OpenEXR`)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_rgb(path: str | Path) -> np.ndarray:
    """float32 (H, W, 3), top row first. Accepts plain "R"/"G"/"B" channels or
    "<layer>.<pass>.R"-style names (the first match wins)."""
    import OpenEXR

    with OpenEXR.File(str(path), separate_channels=True) as f:
        channels = f.channels()
        rgb = []
        for c in "RGB":
            matches = sorted(k for k in channels if k == c or k.endswith(f".{c}"))
            if not matches:
                raise KeyError(f"no {c} channel in {path}; channels: {sorted(channels)[:12]}")
            rgb.append(np.asarray(channels[matches[0]].pixels, dtype=np.float32))
    return np.stack(rgb, axis=-1)


def load_indirect(prefix: str | Path) -> np.ndarray:
    """What Lumina has to add, from cycles_reference.py's renders: <prefix>_full.exr - <prefix>_direct.exr
    (the lights' bounced light), plus <prefix>_emitters.exr (emissive surfaces' direct light) if present."""
    full = load_rgb(f"{prefix}_full.exr")
    direct = load_rgb(f"{prefix}_direct.exr")
    indirect = np.maximum(full - direct, 0.0)
    emitters = Path(f"{prefix}_emitters.exr")
    if emitters.exists():
        indirect = indirect + np.maximum(load_rgb(emitters), 0.0)
    return indirect
