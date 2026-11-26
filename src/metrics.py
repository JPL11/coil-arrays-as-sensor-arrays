"""Reconstruction quality metrics, computed on magnitude images."""
from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def _mag(a):
    return np.abs(a).astype(np.float64)


def nmse(ref, rec):
    r, x = _mag(ref), _mag(rec)
    return float(np.sum((r - x) ** 2) / np.sum(r ** 2))


def psnr(ref, rec):
    r, x = _mag(ref), _mag(rec)
    mse = np.mean((r - x) ** 2)
    return float('inf') if mse == 0 else float(10 * np.log10(r.max() ** 2 / mse))


def ssim(ref, rec):
    r, x = _mag(ref), _mag(rec)
    return float(structural_similarity(r, x, data_range=r.max() - r.min()))


def roi_nmse(ref, rec, center, half=16):
    """NMSE restricted to a box -- used for the small lesion, where global
    metrics are least sensitive and clinical damage matters most."""
    cy, cx = center
    s = (slice(cy - half, cy + half), slice(cx - half, cx + half))
    return nmse(ref[s], rec[s])


def summarize(ref, rec, lesion_center=None):
    out = {"nmse": nmse(ref, rec), "psnr": psnr(ref, rec), "ssim": ssim(ref, rec)}
    if lesion_center is not None:
        out["lesion_nmse"] = roi_nmse(ref, rec, lesion_center)
    return out
