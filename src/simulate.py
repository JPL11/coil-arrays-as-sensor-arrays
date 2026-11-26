"""Multicoil MRI forward model and retrospective undersampling.

Forward model (Cartesian):  y_c = P . F . (S_c * x) + n_c

  x    : complex image                      (ny, nx)
  S_c  : coil sensitivity of coil c         (nc, ny, nx)
  F    : centered 2-D DFT
  P    : k-space sampling mask              (ny, nx)  [undersampled along ky]
  n_c  : complex circular Gaussian noise
"""
from __future__ import annotations

import numpy as np

# --- centered FFT pair -------------------------------------------------------

def fft2c(img):
    """Centered orthonormal 2-D FFT over the last two axes."""
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(img, axes=(-2, -1)), norm="ortho"),
        axes=(-2, -1))


def ifft2c(ksp):
    """Centered orthonormal 2-D inverse FFT over the last two axes."""
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(ksp, axes=(-2, -1)), norm="ortho"),
        axes=(-2, -1))


# --- phantom -----------------------------------------------------------------

def shepp_logan(n=256):
    """Complex Shepp-Logan-style phantom with a small high-contrast lesion.

    The lesion is deliberately small: it is the structure that aggressive
    acceleration destroys first, which is the clinically interesting failure.
    """
    ellipses = [
        # (intensity, a,   b,   x0,    y0,   theta_deg)
        (1.00, 0.69, 0.92, 0.0, 0.0, 0),
        (-0.80, 0.6624, 0.874, 0.0, -0.0184, 0),
        (-0.20, 0.1100, 0.310, 0.22, 0.0, -18),
        (-0.20, 0.1600, 0.410, -0.22, 0.0, 18),
        (0.10, 0.2100, 0.250, 0.0, 0.35, 0),
        (0.10, 0.0460, 0.046, 0.0, 0.10, 0),
        (0.10, 0.0460, 0.046, 0.0, -0.10, 0),
        (0.10, 0.0460, 0.023, -0.08, -0.605, 0),
        (0.10, 0.0230, 0.023, 0.0, -0.606, 0),
        (0.10, 0.0230, 0.046, 0.06, -0.605, 0),
        # small high-contrast "lesion"
        (0.35, 0.0320, 0.032, -0.28, 0.28, 0),
    ]
    g = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(g, -g)
    img = np.zeros((n, n), dtype=float)
    for inten, a, b, x0, y0, th in ellipses:
        t = np.deg2rad(th)
        xr = (X - x0) * np.cos(t) + (Y - y0) * np.sin(t)
        yr = -(X - x0) * np.sin(t) + (Y - y0) * np.cos(t)
        img[(xr / a) ** 2 + (yr / b) ** 2 <= 1] += inten
    img = np.clip(img, 0, None)
    # mild smooth phase -- real MRI images are complex, and a real-valued
    # phantom would make the problem unrealistically easy for SENSE.
    ph = 0.6 * np.pi * (X ** 2 - Y ** 2)
    return (img * np.exp(1j * ph)).astype(np.complex128)


# --- coil sensitivities ------------------------------------------------------

def birdcage_maps(nc=8, n=256, radius=1.35, width=0.55, seed=0):
    """Smooth, spatially distinct complex coil sensitivities on a ring.

    A Biot-Savart-like magnitude falloff with a position-dependent phase --
    enough spatial diversity for SENSE to unfold, without pretending to be a
    physical coil model.
    """
    rng = np.random.default_rng(seed)
    g = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(g, -g)
    maps = np.zeros((nc, n, n), dtype=np.complex128)
    for c in range(nc):
        ang = 2 * np.pi * c / nc
        cx, cy = radius * np.cos(ang), radius * np.sin(ang)
        d2 = (X - cx) ** 2 + (Y - cy) ** 2
        mag = np.exp(-d2 / (2 * width ** 2))
        phase = 1.4 * ((X - cx) * np.cos(ang) + (Y - cy) * np.sin(ang))
        maps[c] = mag * np.exp(1j * phase)
    # normalize so sum-of-squares is 1 where the object is -- keeps the
    # combined image on the same scale as the phantom.
    rss = np.sqrt((np.abs(maps) ** 2).sum(0))
    maps /= np.maximum(rss, 1e-8)
    return maps


# --- forward model -----------------------------------------------------------

def multicoil_kspace(img, maps, noise_std=0.0, seed=0):
    """Fully sampled multicoil k-space, optionally with complex Gaussian noise."""
    ksp = fft2c(maps * img[None])
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        ksp = ksp + noise_std * (rng.normal(size=ksp.shape)
                                 + 1j * rng.normal(size=ksp.shape)) / np.sqrt(2)
    return ksp


def uniform_mask(ny, nx, R, n_acs=24):
    """Uniform every-R-th ky line plus a fully sampled ACS block.

    Returns (mask, acs_slice). The ACS block is what GRAPPA calibrates on and
    what sensitivity estimation uses, so R_eff > R is reported separately.
    """
    mask = np.zeros((ny, nx), dtype=bool)
    mask[::R, :] = True
    if n_acs > 0:
        c0 = ny // 2 - n_acs // 2
        acs = slice(c0, c0 + n_acs)
        mask[acs, :] = True
    else:
        acs = slice(0, 0)
    return mask, acs


def effective_acceleration(mask):
    """R_eff = sampled fraction^-1, counting ACS lines (the honest number)."""
    return mask.shape[0] / np.count_nonzero(mask[:, 0])
