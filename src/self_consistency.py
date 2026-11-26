"""Predictive-coding residuals as reference-free quality signals.

GRAPPA is a linear predictor: it learns, from the calibration block, weights
that reconstruct one k-space line from its acquired neighbours. That is a
predictive-coding model of k-space, and predictive coding's native quantity is
the *prediction error*, which here can be measured without any reference at all:
fit the kernel on part of the ACS block, predict the part held out, and look at
the residual.

Why this should carry what the g-factor cannot. `g` is computed from the
estimated sensitivities and the sampling pattern; it knows the geometry of the
inverse problem and nothing about whether the calibration data actually supports
it. The held-out residual measures precisely that -- how well the acquisition
predicts itself -- which is the bias term that dominates at high acceleration
and that `corr(|bias|, g) = 0.197` says the geometry factor misses.

Nothing here needs a fully sampled reference: acquired lines predict acquired
lines. It is computable on the scanner from the calibration block alone.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def grappa_self_consistency(ksp, acs, R, kh=4, kw=5, lam=1e-4, frac_fit=0.5):
    """Held-out prediction residual of a GRAPPA kernel fitted on the ACS block.

    The ACS rows are split: the kernel is calibrated on the first `frac_fit`
    and scored on the rest, so the residual is genuinely out-of-sample. A kernel
    scored on its own calibration rows would report the Tikhonov residual, which
    is a property of the regulariser rather than of the acquisition.

    Returns a dict of residual statistics, all reference-free.
    """
    nc, ny, nx = ksp.shape
    hw = kw // 2
    acs_data = ksp[:, acs, :]
    n_acs = acs_data.shape[1]

    # same kernel-geometry fallback as `grappa_recon`: shrink kh until the block
    # can calibrate it, so a starved ACS does not silently produce nonsense
    kh_used, src_off = None, None
    for k in range(kh, 1, -1):
        off = (np.arange(k) - (k // 2 - 1)) * R
        if (n_acs - off.max()) - (-off.min()) >= 16:      # 16: 8 fit + 8 held out
            kh_used, src_off = k, off
            break
    if kh_used is None:
        return dict(pc_resid=np.nan, pc_resid_p90=np.nan, pc_resid_ratio=np.nan)

    lo, hi = src_off.min(), src_off.max()
    rows = np.arange(-lo, n_acs - hi)
    n_fit = max(int(len(rows) * frac_fit), 4)
    fit_rows, held_rows = rows[:n_fit], rows[n_fit:]
    if len(held_rows) < 4:
        return dict(pc_resid=np.nan, pc_resid_p90=np.nan, pc_resid_ratio=np.nan)
    cols = np.arange(hw, nx - hw)

    def design(rr):
        blk = acs_data[:, rr[:, None] + src_off[None, :], :]
        blk = np.moveaxis(blk, 0, 1)
        win = sliding_window_view(blk, kw, axis=-1)
        win = np.moveaxis(win, 3, 1)
        return win.reshape(len(rr) * len(cols), -1)

    src_fit = design(fit_rows)
    G = src_fit.conj().T @ src_fit
    G += lam * np.trace(G).real / G.shape[0] * np.eye(G.shape[0])

    src_held = design(held_rows)
    res, den = [], []
    for dy in range(1, R):
        tg_fit = acs_data[:, fit_rows + dy, :][:, :, cols]
        tgt_fit = np.moveaxis(tg_fit, 0, 2).reshape(-1, nc)
        w = np.linalg.solve(G, src_fit.conj().T @ tgt_fit)

        tg_held = acs_data[:, held_rows + dy, :][:, :, cols]
        tgt_held = np.moveaxis(tg_held, 0, 2).reshape(-1, nc)
        pred = src_held @ w
        res.append(np.abs(pred - tgt_held).ravel())
        den.append(np.abs(tgt_held).ravel())

    r = np.concatenate(res); d = np.concatenate(den)
    scale = max(float(np.mean(d)), 1e-30)
    return dict(
        pc_resid=float(np.mean(r) / scale),
        pc_resid_p90=float(np.percentile(r, 90) / scale),
        # residual relative to the signal it is predicting, per sample, which
        # is scale-free in a different way and less dominated by bright lines
        pc_resid_ratio=float(np.mean(r / np.maximum(d, 1e-30))),
    )


def coil_predictability(ksp, acs, lam=1e-4, frac_fit=0.5):
    """How well does each coil's k-space predict from the *other* coils?

    This is predictive coding applied across the array rather than across
    k-space, and it measures array diversity directly. A coil whose signal is
    linearly predictable from its neighbours carries no independent information;
    an array of such coils cannot separate aliased pixels no matter how many
    elements it has. That is the same property the g-factor summarises, arrived
    at without inverting anything and without the sampling pattern -- so it
    should correlate with g, and the interesting question is whether it carries
    anything g does not.

    Fitted and scored on disjoint halves of the ACS block, so the residual is
    out-of-sample. Reference-free: coils predict coils.
    """
    nc, ny, nx = ksp.shape
    A = ksp[:, acs, :].reshape(nc, -1).T                  # (samples, coils)
    n = A.shape[0]
    if n < 4 * nc or nc < 3:
        return dict(coil_pred_mean=np.nan, coil_pred_min=np.nan,
                    coil_pred_max=np.nan, coil_pred_spread=np.nan)
    cut = int(n * frac_fit)
    fit, held = A[:cut], A[cut:]

    res = []
    for c in range(nc):
        others = [j for j in range(nc) if j != c]
        X, y = fit[:, others], fit[:, c]
        G = X.conj().T @ X
        G += lam * np.trace(G).real / max(G.shape[0], 1) * np.eye(G.shape[0])
        w = np.linalg.solve(G, X.conj().T @ y)
        pred = held[:, others] @ w
        denom = max(float(np.mean(np.abs(held[:, c]))), 1e-30)
        res.append(float(np.mean(np.abs(pred - held[:, c])) / denom))
    r = np.asarray(res)
    return dict(coil_pred_mean=float(r.mean()), coil_pred_min=float(r.min()),
                coil_pred_max=float(r.max()), coil_pred_spread=float(r.max() - r.min()))


def multiscale_residual(ksp, acs, R, lam=1e-4):
    """Prediction error at two kernel scales, and their ratio.

    Predictive coding is hierarchical: the same data is predicted at several
    scales and what matters is how the error changes between them. A small
    kernel sees only the immediate neighbourhood; a large one sees broader
    context. If the large kernel does not improve on the small one, the extra
    context carries no information -- the acquisition is locally predictable but
    not globally consistent, which is what a badly calibrated or badly
    conditioned scan looks like.
    """
    small = grappa_self_consistency(ksp, acs, R, kh=2, kw=3, lam=lam)
    large = grappa_self_consistency(ksp, acs, R, kh=4, kw=7, lam=lam)
    s, l = small.get("pc_resid", np.nan), large.get("pc_resid", np.nan)
    ratio = float(l / s) if (np.isfinite(s) and s > 0 and np.isfinite(l)) else np.nan
    return dict(pc_small=s, pc_large=l, pc_scale_ratio=ratio)
