"""Load fastMRI multicoil brain k-space into this study's array conventions.

Three things about real fastMRI data differ from the simulation and every one
of them changes the code:

1. **Axis order.** fastMRI stores k-space as (slice, coil, kx, ky) with the
   phase-encode direction LAST and 2x readout oversampling along kx. `recon.py`
   undersamples axis -2, so k-space is transposed to (coil, ky, kx) on load and
   the oversampled readout is cropped in image space afterwards.

2. **Not every volume is fully sampled.** Many brain volumes are acquired with
   an elliptical / partial-Fourier k-space and carry all-zero phase-encode
   lines (96 of 320 in some AXFLAIR and AXT1 volumes). Those cannot serve as a
   fully-sampled reference -- retrospectively undersampling an already-gapped
   k-space measures the acquisition, not the reconstructor. `is_fully_sampled`
   filters them out and the count of rejected volumes is reported.

3. **Coil noise is correlated.** A real receiver array has a non-diagonal noise
   covariance; the simulation assumed white noise, which is what let SENSE use
   an unweighted least-squares solve. `noise_covariance` estimates Psi from the
   signal-free k-space corners and `prewhiten` applies Psi^-1/2, which is the
   array-processing-correct step and is studied as its own condition.
"""
from __future__ import annotations

import os, glob, re
import numpy as np
import h5py

from simulate import fft2c, ifft2c

PE_STD = 320          # phase-encode lines every volume is standardized to
IMG_STD = 320         # final image size (square), matches fastMRI's own target


# --- discovery ---------------------------------------------------------------

def is_fully_sampled(ksp_slice, crop_pe=None):
    """True if no phase-encode line (axis -1 in fastMRI order) is entirely zero.

    `crop_pe` tests the centre `crop_pe` lines instead of the whole array, which
    is what should be tested when the loader is going to crop to that width
    anyway. It matters for knee: those volumes carry ~36 all-zero lines at the
    *edges* of a 368/372-line acquisition (asymmetric echo) while the centre 320
    -- the only part `load_slice` keeps -- is fully sampled. Testing the raw array
    rejects every knee volume for data the pipeline never touches.

    It is off by default so the brain results stay exactly reproducible. There
    the rejected volumes have zeros spread through a 320-line acquisition, so a
    centre crop is a no-op and the verdict is unchanged either way.
    """
    k = ksp_slice
    if crop_pe is not None and k.shape[-1] > crop_pe:
        c0 = k.shape[-1] // 2 - crop_pe // 2
        k = k[..., c0:c0 + crop_pe]
    return not bool((np.abs(k).sum(axis=(0, 1)) == 0).any())


def list_volumes(root="datasets/fastmri/multicoil_val", require_full=True,
                 crop_pe=None):
    """Discover volumes, returning metadata and the fully-sampled subset."""
    vols, rejected = [], []
    for p in sorted(glob.glob(os.path.join(root, "*.h5"))):
        try:
            with h5py.File(p, "r") as f:
                ns, nc, nro, npe = f["kspace"].shape
                mid = f["kspace"][ns // 2]
                full = is_fully_sampled(mid, crop_pe=crop_pe)
                rec = dict(path=p, n_slices=ns, n_coils=nc, n_ro=nro, n_pe=npe,
                           acquisition=str(f.attrs["acquisition"]), fully_sampled=full)
        except Exception as e:
            rejected.append((p, f"unreadable: {e}")); continue
        if require_full and not full:
            rejected.append((p, "zero-filled phase-encode lines")); continue
        vols.append(rec)
    return vols, rejected


# --- loading -----------------------------------------------------------------

def load_slice(path, sl, pe=PE_STD, coils=None):
    """One slice as (n_coils, ky, kx) with phase-encode on axis -2.

    The phase-encode axis is centre-cropped in k-space to `pe` lines so that
    every volume has the same PE length and R divides it exactly. This lowers
    PE resolution slightly on the 396-line volumes; it does not change FOV.
    """
    with h5py.File(path, "r") as f:
        k = np.asarray(f["kspace"][sl])          # (nc, kx, ky)
    k = np.moveaxis(k, -1, -2)                   # -> (nc, ky, kx)
    ny = k.shape[1]
    if ny > pe:                                  # centre-crop phase encode
        c0 = ny // 2 - pe // 2
        k = k[:, c0:c0 + pe, :]
    elif ny < pe:
        raise ValueError(f"{path} slice {sl}: only {ny} PE lines, need {pe}")
    if coils is not None:
        k = k[coils]
    return k.astype(np.complex128)


def crop_center(img, n=IMG_STD):
    """Centre-crop the (oversampled) readout axis to a square n x n image."""
    ny, nx = img.shape[-2:]
    y0, x0 = ny // 2 - n // 2, nx // 2 - n // 2
    return img[..., y0:y0 + n, x0:x0 + n]


def reference(ksp):
    """Fully-sampled root-sum-of-squares reference image, cropped and real.

    This is the same construction fastMRI uses for `reconstruction_rss`, which
    is what `verify_against_rss` checks it against.
    """
    coil = ifft2c(ksp)
    return crop_center(np.sqrt((np.abs(coil) ** 2).sum(0)))


# --- noise handling ----------------------------------------------------------

def noise_covariance(ksp, frac=0.06):
    """Coil noise covariance Psi from the signal-free corners of k-space.

    Uses the four outer corners, where a brain acquisition has no object energy.
    Returns (Psi, correlation_matrix); the off-diagonal magnitude of the
    correlation matrix is the thing the simulation had no analogue for.
    """
    nc, ny, nx = ksp.shape
    hy, hx = max(int(ny * frac), 4), max(int(nx * frac), 4)
    corners = np.concatenate([
        ksp[:, :hy, :hx].reshape(nc, -1), ksp[:, :hy, -hx:].reshape(nc, -1),
        ksp[:, -hy:, :hx].reshape(nc, -1), ksp[:, -hy:, -hx:].reshape(nc, -1)], axis=1)
    corners = corners - corners.mean(axis=1, keepdims=True)
    Psi = (corners @ corners.conj().T) / corners.shape[1]
    d = np.sqrt(np.real(np.diag(Psi)))
    corr = Psi / np.maximum(np.outer(d, d), 1e-30)
    return Psi, corr


def prewhiten(ksp, Psi=None):
    """Apply Psi^-1/2 across coils so the noise becomes white and unit-variance.

    After this the plain least-squares SENSE solve in `recon.py` is the correct
    (maximum-likelihood) estimator; without it, it is not.
    """
    if Psi is None:
        Psi, _ = noise_covariance(ksp)
    # Cholesky of Psi; L^-1 whitens.  Regularized for numerical safety.
    Psi = Psi + 1e-12 * np.trace(Psi).real / Psi.shape[0] * np.eye(Psi.shape[0])
    L = np.linalg.cholesky(Psi)
    W = np.linalg.inv(L)
    return np.einsum('ij,jyx->iyx', W, ksp)


# --- verification ------------------------------------------------------------

def verify_against_rss(path, sl):
    """Relative error between our reference() and fastMRI's own reconstruction_rss.

    The real-data analogue of the simulation's forward-model round-trip check:
    it confirms the axis convention and cropping against the dataset's own
    target. Because `load_slice` transposes k-space to put phase-encode on
    axis -2 (what `recon.py` needs), our image is the transpose of fastMRI's,
    so the comparison transposes back. Agreement is limited by the complex64
    storage of the archive, not by the pipeline.
    """
    with h5py.File(path, "r") as f:
        target = np.asarray(f["reconstruction_rss"][sl], dtype=np.float64)
        k = np.moveaxis(np.asarray(f["kspace"][sl]), -1, -2).astype(np.complex128)
    n = target.shape[-1]
    ours = crop_center(np.sqrt((np.abs(ifft2c(k)) ** 2).sum(0)), n)
    return float(np.linalg.norm(ours.T - target) / np.linalg.norm(target))


# --- evaluation masking ------------------------------------------------------

def body_mask(ref, thresh=0.08):
    """Binary support of the head, from the fully-sampled reference.

    Real evaluation needs this and the simulation did not. A root-sum-of-squares
    reference carries a positive (Rayleigh) noise bias outside the object, while
    a sensitivity-weighted reconstruction is ~zero there, because the estimated
    maps are explicitly zeroed in the background. That mismatch is systematic,
    has nothing to do with parallel imaging, and dominates the error: measured
    against an unmasked RSS reference, a *fully sampled* matched-filter combine
    scores SSIM 0.48 with 80% of its error energy outside the head.

    The mask must therefore be tight. An earlier version thresholded and then
    called `binary_fill_holes` on the result, which welded 154 isolated noise
    specks into a single blob covering 0.83 of the frame -- roughly twice the
    head -- so metrics were still averaging over background. Opening first
    removes the specks, and keeping only the largest connected component before
    filling holes stops the background from being absorbed.
    """
    from scipy import ndimage
    m = ref > thresh * ref.max()
    m = ndimage.binary_opening(m, np.ones((3, 3)))          # drop noise specks
    lab, n = ndimage.label(m)
    if n:                                                   # head = largest blob
        m = lab == (1 + int(np.argmax(ndimage.sum(m, lab, range(1, n + 1)))))
    m = ndimage.binary_closing(m, np.ones((7, 7)))
    return ndimage.binary_fill_holes(m)


def apply_mask(img, mask):
    """Zero everything outside the body so metrics see anatomy, not noise floor."""
    return np.where(mask, img, 0.0)


def remove_readout_oversampling(ksp, factor=2):
    """Discard the 2x readout oversampling: (nc, ky, kx) -> (nc, ky, kx/2).

    fastMRI acquires the readout at twice the reconstructed FOV; every pipeline
    throws the outer half away, and this study already crops to 320 before
    computing any metric. Removing it up front is exact -- an inverse FFT along
    the readout only, a centre crop, and a forward FFT back -- and it touches
    neither the phase-encode axis nor anything parallel imaging depends on. It
    halves the pixel count, which matters once per-pixel eigendecompositions
    (ESPIRiT) enter the picture.
    """
    nx = ksp.shape[-1]
    n = nx // factor
    img = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(ksp, axes=-1),
                                      axis=-1, norm="ortho"), axes=-1)
    x0 = nx // 2 - n // 2
    img = img[..., x0:x0 + n]
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(img, axes=-1),
                                      axis=-1, norm="ortho"), axes=-1)
