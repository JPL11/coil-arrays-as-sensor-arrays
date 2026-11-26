"""Parallel-imaging reconstructors: zero-filled, SENSE, GRAPPA.

Both SENSE and GRAPPA are implemented directly from the original papers
(Pruessmann 1999; Griswold 2002) rather than called from a library -- the point
of the study is the array-processing mechanics.

Aliasing convention (verified numerically against a point source): with every
R-th ky line retained, `ifft2c(k*mask)*R` places replicas of image row y at rows
(y + r*ny/R) mod ny for r = 0..R-1, with unit weight and no phase modulation.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from simulate import fft2c, ifft2c


def _windows(blk, kw):
    """All length-kw windows along the last axis of blk -> (n_col, ..., kw).

    Vectorized replacement for stacking `blk[..., c-hw:c+hw+1]` over every
    centre column c. Identical output, but it does not build one array per
    column -- which matters on real data, where the readout axis is 640 wide
    instead of the simulation's 192.
    """
    return np.moveaxis(sliding_window_view(blk, kw, axis=-1), -2, 0)


# --- coil combination --------------------------------------------------------

def rss(coil_imgs):
    """Root-sum-of-squares combine (no sensitivity knowledge required)."""
    return np.sqrt((np.abs(coil_imgs) ** 2).sum(0))


def sense_combine(coil_imgs, maps):
    """Matched-filter (sensitivity-weighted) combine."""
    return (np.conj(maps) * coil_imgs).sum(0)


def optimal_combine(coil_imgs, maps, Psi):
    """Psi-weighted R=1 estimate, (S^H Psi^-1 S)^-1 S^H Psi^-1 x.

    This, not the matched filter above, is the fully sampled reconstruction the
    g-factor is defined against. On correlated coil noise the matched filter is
    noisier, so using it as the reference deflates a measured g by a constant
    factor (0.86 on the fastMRI brain arrays; `diagnose_g_gap.py`).
    """
    nc = Psi.shape[0]
    Pinv = np.linalg.inv(Psi + 1e-12 * np.trace(Psi).real / nc * np.eye(nc))
    w = np.einsum('ij,jyx->iyx', Pinv, maps)
    num = (np.conj(w) * coil_imgs).sum(0)
    den = (np.conj(w) * maps).sum(0).real
    return np.where(den > 1e-12, num / np.maximum(den, 1e-12), 0)


# --- sensitivity estimation --------------------------------------------------

def estimate_maps_acs(ksp, acs, thresh=0.02, hann=True):
    """Low-resolution sensitivity maps from the fully sampled ACS block.

    This is what a real autocalibrated scan has access to -- deliberately not
    the true maps, so sensitivity-estimation error is part of the experiment.
    """
    nc, ny, nx = ksp.shape
    k_acs = np.zeros_like(ksp)
    k_acs[:, acs, :] = ksp[:, acs, :]
    if hann:  # taper to suppress Gibbs ringing in the low-res maps
        n_acs = acs.stop - acs.start
        w = np.hanning(n_acs)[None, :, None]
        k_acs[:, acs, :] *= w
    low = ifft2c(k_acs)
    body = rss(low)
    maps = low / np.maximum(body, 1e-12)[None]
    maps[:, body < thresh * body.max()] = 0  # mask out noise-only background
    return maps


# --- zero-filled -------------------------------------------------------------

def zerofill_recon(ksp_us, maps=None):
    """Inverse FFT of the zero-filled k-space, then coil-combine."""
    coil = ifft2c(ksp_us)
    return sense_combine(coil, maps) if maps is not None else rss(coil)


# --- SENSE -------------------------------------------------------------------

def sense_recon(ksp, maps, R, lam=0.0, return_gfactor=False, Psi=None):
    """Cartesian SENSE (Pruessmann 1999), uniform undersampling by R.

    Solves, for each reduced-FOV pixel, the nc x R problem a = S rho as

        rho_hat = (S^H Psi^-1 S + lam I)^-1 S^H Psi^-1 a

    with the receiver noise covariance Psi, exactly as in the paper. Passing
    Psi=None substitutes the identity, which is the white-noise special case;
    that was the only version implemented at first, and on real arrays it is
    *not* the paper's estimator -- measured off-diagonal coil noise correlation
    on this fastMRI data reaches 0.54.

    The g-factor follows the paper's definition on the same weighted matrix,

        g_rho = sqrt[ (S^H Psi^-1 S)^-1_rho,rho * (S^H Psi^-1 S)_rho,rho ]

    and is scale-invariant in S, so the RSS normalization of the maps does not
    affect it. `validate_snr_law.py` checks this g against the noise actually
    measured by Monte-Carlo replicas, which is the paper's central quantitative
    claim (SNR_R = SNR_full / (g * sqrt(R))).
    """
    nc, ny, nx = ksp.shape
    Pinv = None
    if Psi is not None:
        Pinv = np.linalg.inv(Psi + 1e-12 * np.trace(Psi).real / nc * np.eye(nc))
    m = np.zeros((ny, nx), dtype=bool)
    m[::R, :] = True                       # strictly uniform: SENSE needs it
    alias = ifft2c(ksp * m) * R            # (nc, ny, nx)

    step = ny // R
    out = np.zeros((ny, nx), dtype=np.complex128)
    gmap = np.zeros((ny, nx), dtype=float) if return_gfactor else None

    for y in range(step):
        idx = [(y + r * step) % ny for r in range(R)]
        S = maps[:, idx, :]                        # (nc, R, nx)
        a = alias[:, y, :]                         # (nc, nx)
        S = np.moveaxis(S, 2, 0)                   # (nx, nc, R)
        a = np.moveaxis(a, 1, 0)[..., None]        # (nx, nc, 1)
        Sh = np.conj(np.swapaxes(S, 1, 2))         # (nx, R, nc)
        if Pinv is not None:                       # S^H Psi^-1
            Sh = Sh @ Pinv[None]
        A = Sh @ S                                 # (nx, R, R)
        if lam:
            # Diagonal loading, scaled to the local array gain -- the direct
            # analogue of loaded MVDR/Capon beamforming under steering-vector
            # mismatch. Relative scaling keeps one lam meaningful across pixels.
            trace = np.einsum('xrr->x', A).real / R
            A = A + lam * np.maximum(trace, 1e-12)[:, None, None] * np.eye(R)[None]
        # pinv keeps pixels where a coil set is rank-deficient from blowing up
        Ainv = np.linalg.pinv(A)
        rho = (Ainv @ (Sh @ a))[..., 0]            # (nx, R)
        out[idx, :] = rho.T
        if return_gfactor:
            diagA = np.einsum('xrr->xr', A).real
            diagAinv = np.einsum('xrr->xr', Ainv).real
            g = np.sqrt(np.maximum(diagAinv * diagA, 0.0))
            gmap[idx, :] = g.T
    return (out, gmap) if return_gfactor else out


# --- GRAPPA ------------------------------------------------------------------

def cg_sense(ksp, maps, mask, Psi=None, n_iter=250, lam=0.0, tol=1e-6, R=None):
    """Iterative (conjugate-gradient) SENSE for an arbitrary Cartesian mask.

    Cartesian `sense_recon` requires strictly uniform every-Rth sampling,
    because it solves the R-way aliasing problem pixel by pixel. That made it
    the one reconstructor in this study that could not consume an ACS block,
    so it silently ran at a higher effective acceleration than everything it
    was compared against -- the defect the 2026-08-30 fairness audit named.
    This solves the same estimator by normal equations instead,

        (E^H Psi^-1 E + lam I) x = E^H Psi^-1 y,    E = M F S

    which places no structure requirement on M, so every method can be handed
    the identical acquired line set. With Psi given, the system is whitened
    first (Psi = L L^H, S <- L^-1 S, y <- L^-1 y), which makes the operator
    Hermitian positive-definite and lets plain CG run on it.

    At strictly uniform sampling and lam=0 this converges to what `sense_recon`
    returns, but only in the limit: the operator is ill-conditioned because the
    RSS-normalized maps leave a near-null space outside the object, so the
    relative residual falls roughly a decade per 60 iterations (2.3e-6 at 200,
    2.9e-9 at 400 on the birdcage check in `validate_cg_sense.py`). The default
    is therefore deliberately generous, and `lam > 0` both regularizes and
    accelerates it. Because a fully converged unregularized solve costs several
    seconds per slice, the tumour pipeline uses this only for the shared-ACS
    sampling convention; the uniform-only convention keeps the direct solve.
    """
    nc, ny, nx = ksp.shape
    if Psi is not None:
        L = np.linalg.cholesky(Psi + 1e-12 * np.trace(Psi).real / nc * np.eye(nc))
        Linv = np.linalg.inv(L)
        ksp = np.einsum("ij,jkl->ikl", Linv, ksp)
        maps = np.einsum("ij,jkl->ikl", Linv, maps)
    m = mask[None].astype(bool)
    ksp = ksp * m

    # Loading relative to the local array gain, exactly as `sense_recon` does
    # it, so that one `lam` means one thing in both solvers: lam times the
    # sampled fraction times sum_c |S_c|^2 (whitened), averaged over the R
    # pixels that alias together under the underlying every-Rth pattern. At
    # strictly uniform sampling this reproduces the direct regularised solve to
    # 2e-5 in magnitude. The absolute `lam * x` used here before was ~2x
    # stronger at the study's settings, so "SENSE+DL" changed regulariser
    # between the strictly uniform point of the R_eff sweep and every other.
    load = 0.0
    if lam:
        w = (np.abs(maps) ** 2).sum(0)
        if R and ny % R == 0:
            w = w.reshape(R, ny // R, nx).mean(0, keepdims=True).repeat(R, 0).reshape(ny, nx)
        load = lam * m.mean() * np.maximum(w, 1e-12)

    def A(x):
        return np.sum(np.conj(maps) * ifft2c(fft2c(maps * x[None]) * m), axis=0) + load * x

    b = np.sum(np.conj(maps) * ifft2c(ksp), axis=0)
    x = np.zeros((ny, nx), dtype=np.complex128)
    r = b - A(x)
    p = r.copy()
    rs = np.vdot(r, r).real
    rs0 = rs
    # `tol` is a relative *residual norm*; rs tracks its square.
    for _ in range(n_iter):
        if rs <= (tol ** 2) * rs0 or rs == 0:
            break
        Ap = A(p)
        alpha = rs / np.vdot(p, Ap).real
        x += alpha * p
        r -= alpha * Ap
        rs_new = np.vdot(r, r).real
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x


def grappa_recon(ksp, acs, R, kh=4, kw=5, lam=1e-4, maps=None, return_info=False,
                 restore_acs=True, restore_slice=None):
    """Cartesian GRAPPA (Griswold 2002).

    Learns, from the ACS block, linear kernels that predict each missing ky
    line from a kh x kw neighbourhood of acquired lines across all coils, then
    applies them to fill k-space. Coil-combines with RSS unless maps are given.

    kh : number of acquired ky lines in the kernel (spaced R apart)
    kw : kernel width along kx (odd)
    """
    nc, ny, nx = ksp.shape
    assert kw % 2 == 1, "kernel width must be odd"
    hw = kw // 2

    m = np.zeros((ny, nx), dtype=bool)
    m[::R, :] = True
    k_us = ksp * m                                  # what the scanner acquired
    acs_data = ksp[:, acs, :]                       # (nc, n_acs, nx)
    n_acs = acs_data.shape[1]

    # Kernel geometry: source lines at offsets, targets at the R-1 gaps.
    # A tall kernel needs a tall ACS block; shrink kh until the block can
    # calibrate it. The kh actually used is reported, since it is a real
    # consequence of the ACS budget rather than a free parameter.
    kh_used = None
    for k in range(kh, 1, -1):
        off = (np.arange(k) - (k // 2 - 1)) * R
        n_rows = (n_acs - off.max()) - (-off.min())
        if n_rows >= 8:
            kh_used, src_off = k, off
            break
    if kh_used is None:
        raise ValueError(
            f"ACS block of {n_acs} lines cannot calibrate any GRAPPA kernel at R={R} "
            f"(need >= 8 calibration rows for kh=2, span {3*R})")
    kh = kh_used
    lo, hi = src_off.min(), src_off.max()

    # --- calibrate ---------------------------------------------------------
    rows = list(range(-lo, n_acs - hi))
    cols = np.arange(hw, nx - hw)

    # Source matrix (n_samples, nc*kh*kw), built in one shot rather than one
    # slab per calibration row.
    rows_a = np.asarray(rows)
    blk = acs_data[:, rows_a[:, None] + src_off[None, :], :]   # (nc,nrow,kh,nx)
    blk = np.moveaxis(blk, 0, 1)                               # (nrow,nc,kh,nx)
    win = sliding_window_view(blk, kw, axis=-1)                # (nrow,nc,kh,ncol,kw)
    win = np.moveaxis(win, 3, 1)                               # (nrow,ncol,nc,kh,kw)
    src = win.reshape(len(rows) * len(cols), -1)

    G = src.conj().T @ src
    G += lam * np.trace(G).real / G.shape[0] * np.eye(G.shape[0])  # Tikhonov

    # Solve G w = src^H tgt per target offset. Forming G^-1 src^H explicitly
    # would build an (nc*kh*kw, n_samples) matrix and cost O(p^2 n); reducing
    # src^H tgt to (nc*kh*kw, nc) first makes the solve O(p^2 nc) instead, for
    # the same weights.
    weights = {}
    for dy in range(1, R):                          # each missing offset
        tg = acs_data[:, rows_a + dy, :][:, :, cols]           # (nc,nrow,ncol)
        tgt = np.moveaxis(tg, 0, 2).reshape(-1, nc)
        weights[dy] = np.linalg.solve(G, src.conj().T @ tgt)   # (nc*kh*kw, nc)

    # --- apply -------------------------------------------------------------
    # k-space is zero-padded before the kernel is applied. Without padding, the
    # first and last `hw` readout columns fall outside `cols` and the phase-
    # encode lines near the edges fail the in-bounds test, so those regions of
    # k-space are never filled and stay zero. The residual that leaves is not
    # small and it *grows with kernel size*, which inverts the expected
    # convergence: bigger kernels should fit better, not worse. Padding is
    # benign here because the k-space periphery being padded is already
    # near-zero. `validate_snr_law.py` checks that error now decreases with
    # kernel size and ACS, which is the behaviour Griswold's derivation implies.
    py = max(-lo, hi, 0) + R
    px = hw
    kp = np.pad(k_us, ((0, 0), (py, py), (px, px)))
    k_out = kp.copy()
    cols_p = np.arange(px, nx + px)                 # every original column
    base = py + np.arange(0, ny, R)
    nyp = kp.shape[1]
    coil_ix = np.arange(nc)[:, None, None]
    for dy in range(1, R):
        W = weights[dy]
        # Every target line for this offset is filled in one matmul. Looping
        # line by line rebuilt a (n_col, nc, kh, kw) window stack per line,
        # which dominates the runtime once R is large (R-1 offsets x ny/R lines
        # per slice, over four modalities and every slice of a volume).
        y0 = base[(base + src_off.min() >= 0) & (base + src_off.max() < nyp)
                  & (base + dy < nyp)]
        if not len(y0):
            continue
        blk = kp[:, y0[:, None] + src_off[None, :], :]        # (nc, nv, kh, nxp)
        blk = np.moveaxis(blk, 0, 1)                          # (nv, nc, kh, nxp)
        win = sliding_window_view(blk, kw, axis=-1)           # (nv, nc, kh, ncol, kw)
        win = np.moveaxis(win, 3, 1)                          # (nv, ncol, nc, kh, kw)
        res = win.reshape(len(y0), len(cols_p), -1) @ W       # (nv, ncol, nc)
        k_out[coil_ix, (y0 + dy)[None, :, None], cols_p[None, None, :]] = \
            np.moveaxis(res, 2, 0)
    k_full = k_out[:, py:py + ny, px:px + nx]

    if restore_acs:
        # Autocalibrating GRAPPA normally keeps the ACS lines it measured. That
        # is only fair if every other arm is handed them too: under the
        # uniform-only sampling convention the ACS block is a separate
        # calibration survey, so the kernel is still fitted on it but the image
        # is reconstructed from the shared uniform samples alone.
        # `restore_slice` separates what the kernel is fitted on from what the
        # output keeps. They are the same block in a normal autocalibrated
        # acquisition, but a sweep that holds calibration fixed while varying
        # how many centre lines are actually acquired needs them apart --
        # otherwise GRAPPA silently ignores the extra lines every other arm is
        # consuming and its score is constant across the sweep.
        keep = acs if restore_slice is None else restore_slice
        k_full[:, keep, :] = ksp[:, keep, :]        # keep true acquired lines
    coil = ifft2c(k_full)
    img = sense_combine(coil, maps) if maps is not None else rss(coil)
    if return_info:
        return img, {"kh_used": kh, "kw": kw, "n_acs": n_acs,
                     "n_calib_rows": len(rows), "kernel_span": int(hi - lo)}
    return img


# --- ESPIRiT ------------------------------------------------------------------

def espirit_maps(ksp, acs, k=6, c=0.02, eig_thresh=0.92, n_maps=1,
                 order=None, return_info=False):
    """ESPIRiT sensitivity maps (Uecker et al., Magn. Reson. Med. 71:990, 2014).

    A subspace steering-vector estimator, and the natural array-processing
    upgrade to `estimate_maps_acs`. That estimator forms sensitivities as a
    ratio of low-resolution coil images, so its output is only as sharp as the
    ACS block -- which is why it caps SENSE at SSIM ~0.60 on real coils, whose
    sensitivities are ~14x rougher than a birdcage model.

    ESPIRiT instead never forms a ratio. It builds a block-Hankel calibration
    matrix from every k x k patch of the ACS block, splits it into signal and
    null subspaces by SVD, transforms the retained signal-subspace kernels into
    image space, and takes the sensitivity at each pixel as the dominant
    eigenvector of the local nc x nc Gram matrix. The eigenvalue is a natural
    confidence measure -- it is 1 exactly where the data locally lies in the
    calibration subspace -- so the object support falls out of the method rather
    than being thresholded in by hand.

    This is the same signal/noise-subspace split that underlies MUSIC and other
    subspace direction finders; here the "array manifold" is learned from the
    ACS data instead of assumed from geometry.

    k           : calibration kernel width (k x k)
    c           : singular-value threshold, relative to the largest
    eig_thresh  : eigenvalue below which a pixel is treated as outside the object
    n_maps      : number of eigenvector sets to return (1 = standard SENSE use)
    """
    nc, ny, nx = ksp.shape
    cal = ksp[:, acs, :]                                # (nc, n_acs, nx)
    n_acs = cal.shape[1]
    if n_acs < k or nx < k:
        raise ValueError(f"ACS block {n_acs}x{nx} too small for a {k}x{k} kernel")

    # --- calibration matrix: every k x k patch, across all coils -----------
    W = sliding_window_view(cal, (k, k), axis=(1, 2))   # (nc, ry, rx, k, k)
    A = np.ascontiguousarray(np.moveaxis(W, 0, 2)).reshape(-1, nc * k * k)

    # --- signal subspace ---------------------------------------------------
    # eigh on A^H A is equivalent to the SVD of A and far cheaper, since
    # A has many more rows (patches) than columns (nc*k*k).
    G = A.conj().T @ A
    w, V = np.linalg.eigh(G)                            # ascending
    sv = np.sqrt(np.maximum(w, 0.0))
    if order in ("mdl", "aic"):
        # subspace size chosen by model order selection instead of a threshold;
        # eigenvalues of the sample covariance are those of G / n_patches
        nk, _ = mdl_aic_order(w / A.shape[0], A.shape[0], order)
        nk = int(np.clip(nk, 1, V.shape[1]))
        V = V[:, -nk:]
    elif isinstance(order, (int, np.integer)):
        nk = int(np.clip(order, 1, V.shape[1]))
        V = V[:, -nk:]
    else:
        V = V[:, sv > c * sv.max()]                     # (nc*k*k, n_kernels)
        nk = V.shape[1]

    # --- kernels to image space -------------------------------------------
    # Each retained right-singular vector is a convolution kernel. Paired with
    # the centred *inverse* transform below, the correlation-to-convolution
    # adjustment is a conjugation alone -- conjugating *and* flipping applies it
    # twice and returns maps orthogonal to the true sensitivities while leaving
    # every eigenvalue at 1, so it cannot be caught by inspecting eigenvalues.
    # Guard: `test_espirit.py` checks alignment against known birdcage maps, and
    # checks that retaining *all* singular vectors makes the Gram exactly the
    # identity (Parseval), which pins the transform and the sqrt(N)/k scaling
    # independently of the subspace choice.
    ker = V.T.reshape(nk, nc, k, k).conj()
    scale = np.sqrt(ny * nx) / k
    y0, x0 = ny // 2 - k // 2, nx // 2 - k // 2

    # Accumulate the per-pixel Gram matrix one kernel at a time: holding every
    # kernel image at once would cost nk*nc*ny*nx complex entries.
    Gram = np.zeros((ny, nx, nc, nc), dtype=np.complex128)
    buf = np.zeros((nc, ny, nx), dtype=np.complex128)
    for j in range(nk):
        buf[:] = 0
        buf[:, y0:y0 + k, x0:x0 + k] = ker[j]
        m = ifft2c(buf) * scale                         # (nc, ny, nx)
        Gram += np.einsum('cyx,dyx->yxcd', m, m.conj(), optimize=True)

    # --- per-pixel eigendecomposition --------------------------------------
    evals, evecs = np.linalg.eigh(Gram)                 # ascending over last axis
    maps = np.empty((n_maps, nc, ny, nx), dtype=np.complex128)
    lams = np.empty((n_maps, ny, nx), dtype=float)
    for i in range(n_maps):
        col = -1 - i                                    # dominant first
        S = np.moveaxis(evecs[..., col], -1, 0)         # (nc, ny, nx)
        lam = evals[..., col]
        # Gauge fix: an eigenvector is defined only up to phase, which would
        # otherwise vary pixel to pixel. Align each pixel to a real, positive
        # first coil. Magnitude images are unaffected either way.
        ph = S[0] / np.maximum(np.abs(S[0]), 1e-12)
        S = S * np.conj(ph)[None]
        S[:, lam < eig_thresh] = 0                      # outside the object
        maps[i], lams[i] = S, lam

    out = maps[0] if n_maps == 1 else maps
    if return_info:
        return out, {"n_kernels": nk, "n_total": nc * k * k, "k": k,
                     "n_patches": A.shape[0],
                     "eig_max": float(lams[0].max()),
                     "support_frac": float((lams[0] >= eig_thresh).mean())}
    return out


# --- worst-case robust SENSE --------------------------------------------------

def sense_robust(ksp, maps, R, eps=0.1, n_iter=6, return_lam=False):
    """SENSE with worst-case robustness to steering-vector mismatch.

    Diagonal loading with a fixed lambda is the classic robust-beamforming
    remedy, and this study's real-data result is that it repairs *conditioning*
    but buys essentially nothing against *mismatch* (SENSE -0.064 vs SENSE+DL
    -0.066 SSIM over a 0->40% map error). The reason is that a fixed lambda is
    not aimed at mismatch at all: it is chosen once, globally, with no reference
    to how wrong the steering vectors might be.

    The array-processing remedy that *is* aimed at mismatch solves the
    worst-case problem over an uncertainty set around the steering vectors,

        min_rho  max_{||dS||_F <= eps||S||}  || (S + dS) rho - a ||

    whose solution (Chandrasekaran et al. 1998; El Ghaoui & Lebret 1997) is
    Tikhonov-regularized least squares with lambda chosen to satisfy a secular
    equation rather than fixed by hand:

        lambda* = eps * ||S|| * ||S rho - a|| / ||rho||

    so the loading becomes **per-pixel and data-adaptive** -- large where the
    residual is big relative to the solution norm, small where the fit is
    already good. That is the structural difference from `sense_recon(lam=...)`,
    which applies one lambda everywhere. The fixed point is reached by a few
    sweeps of the iteration below; `eps` is the assumed relative uncertainty in
    the sensitivity maps and is the single knob.

    eps    : relative steering-vector uncertainty (0 recovers plain SENSE)
    n_iter : fixed-point sweeps for the secular equation
    """
    nc, ny, nx = ksp.shape
    m = np.zeros((ny, nx), dtype=bool)
    m[::R, :] = True
    alias = ifft2c(ksp * m) * R

    step = ny // R
    out = np.zeros((ny, nx), dtype=np.complex128)
    lam_map = np.zeros((ny, nx), dtype=float) if return_lam else None

    for y in range(step):
        idx = [(y + r * step) % ny for r in range(R)]
        S = np.moveaxis(maps[:, idx, :], 2, 0)          # (nx, nc, R)
        a = np.moveaxis(alias[:, y, :], 1, 0)[..., None]  # (nx, nc, 1)
        Sh = np.conj(np.swapaxes(S, 1, 2))
        A = Sh @ S
        Sha = Sh @ a
        nrmS = np.sqrt(np.maximum(np.einsum('xrr->x', A).real, 0.0))   # ||S||_F
        eye = np.eye(R)[None]

        # start from the unregularized solve, then chase the fixed point
        lam = np.zeros(nx)
        rho = (np.linalg.pinv(A) @ Sha)[..., 0]
        for _ in range(n_iter):
            resid = np.linalg.norm((S @ rho[..., None])[..., 0] - a[..., 0], axis=1)
            nrho = np.linalg.norm(rho, axis=1)
            lam = eps * nrmS * resid / np.maximum(nrho, 1e-12)
            Areg = A + lam[:, None, None] * eye
            rho = (np.linalg.pinv(Areg) @ Sha)[..., 0]
        out[idx, :] = rho.T
        if return_lam:
            lam_map[idx, :] = np.repeat(lam[None, :], R, axis=0)
    return (out, lam_map) if return_lam else out


# --- model order selection ----------------------------------------------------

def mdl_aic_order(eigvals, n_snapshots, criterion="mdl"):
    """Number of signal eigenvalues by MDL or AIC (Wax & Kailath, 1985).

    ESPIRiT's subspace size is set by a hand-picked singular-value threshold
    `c`, and the 48-point sweep in `run_espirit_tune.py` shows reconstruction
    quality swings 0.14 SSIM across plausible settings -- so the threshold is
    doing real work and there is no principled reason for any particular value.
    Model order selection is the array-processing answer: decide how many
    eigenvalues belong to the signal subspace by asking which split best
    explains the data, rather than by picking a number.

    Both criteria assume the *noise* eigenvalues are equal in expectation and
    score the split k by the log ratio of the arithmetic to geometric mean of
    the discarded tail (which is 0 when they are exactly equal and grows as
    they are not), penalized by the number of free parameters:

        L(k)   = (p - k) N log( AM(lam_{k+1..p}) / GM(lam_{k+1..p}) )
        AIC(k) = 2 L(k) + 2 k (2p - k)
        MDL(k) = L(k) + (1/2) k (2p - k) log N

    MDL is consistent (it converges to the true order as N grows); AIC is not,
    and tends to overestimate. Both are returned so the difference is visible.
    """
    lam = np.sort(np.maximum(np.asarray(eigvals, dtype=float), 0.0))[::-1]
    p = lam.size
    N = float(n_snapshots)
    crit = np.full(p, np.inf)
    for k in range(p):
        tail = lam[k:]
        m = tail.size
        if m < 1:
            continue
        am = tail.mean()
        if am <= 0:
            continue
        gm = np.exp(np.mean(np.log(np.maximum(tail, 1e-300))))
        Lk = m * N * np.log(am / max(gm, 1e-300))
        free = k * (2 * p - k)
        crit[k] = (2 * Lk + 2 * free) if criterion == "aic" else (Lk + 0.5 * free * np.log(N))
    return int(np.argmin(crit)), crit
