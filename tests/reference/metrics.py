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


def central(img: np.ndarray, keep: float = 0.7) -> np.ndarray:
    h, w = img.shape[:2]
    y0, x0 = int(h * (1 - keep) / 2), int(w * (1 - keep) / 2)
    return img[y0:h - y0, x0:w - x0]


def gate2_metrics(test: np.ndarray, reference: np.ndarray, clipped_test: bool = True) -> dict[str, float]:
    """clipped_test: the Godot side is an 8-bit capture, clipped at 1, so the reference is clipped to
    match. A float capture is compared unclipped (PSNR's peak stays 1.0)."""
    if clipped_test:
        reference = np.clip(reference, 0.0, 1.0)
    s = fit_scale(test, reference)
    fitted = np.clip(test * s, 0.0, 1.0) if clipped_test else test * s
    return {
        "psnr_db": psnr(test, reference),
        "psnr_fitted_db": psnr(fitted, reference),
        "psnr_center_db": psnr(central(fitted), central(reference)),
        "fitted_scale": s,
        "mean_test": float(np.mean(test)),
        "mean_reference": float(np.mean(reference)),
    }
