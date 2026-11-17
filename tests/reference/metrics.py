"""Image metrics for Gate 2 (pure numpy)."""

from __future__ import annotations

import numpy as np


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def psnr(test: np.ndarray, reference: np.ndarray, peak: float = 1.0) -> float:
    mse = float(np.mean((np.asarray(test, np.float64) - np.asarray(reference, np.float64)) ** 2))
    return float("inf") if mse == 0.0 else 10.0 * np.log10(peak * peak / mse)


def fit_scale(test: np.ndarray, reference: np.ndarray) -> float:
    """Least-squares s minimising |s * test - reference|^2 (absorbs light-unit calibration)."""
    t = np.asarray(test, np.float64).ravel()
    r = np.asarray(reference, np.float64).ravel()
    denom = float(t @ t)
    return float(t @ r) / denom if denom > 0 else 1.0


def gate2_metrics(test: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    reference = np.clip(reference, 0.0, 1.0)   # the Godot side is an 8-bit capture, clipped at 1
    s = fit_scale(test, reference)
    return {
        "psnr_db": psnr(test, reference),
        "psnr_fitted_db": psnr(np.clip(test * s, 0.0, 1.0), reference),
        "fitted_scale": s,
    }
