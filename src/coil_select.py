"""Sensor selection for a receiver array: which K of N coils should you keep?

The aperture sweep shows SENSE improving monotonically with coil count
(0.862 -> 0.925 for 4 -> 16 coils on fixed slices), which answers "how many"
and says nothing about "which". That second question is the classical
sensor-selection problem from array processing, and in MRI it is not the same
question as coil compression: compression forms K linear *combinations* of N
channels and needs all N receivers physically present, while selection keeps K
of the actual elements and is what applies when channels are limited by
hardware, by reconstruction cost, or by a coil that has failed.

The objective comes from the same expression the g-factor does. For a
reduced-FOV pixel p aliasing R positions, SENSE solves a = S rho with
    G(p) = S(p)^H Psi^-1 S(p)          (R x R)
and the noise amplification at that pixel is
    g(p) = sqrt( [G^-1]_pp [G]_pp ).
Selecting coils means selecting rows of S, so the objective is a function of a
principal submatrix -- exactly the structure that makes greedy selection
sensible, since mean g over the FOV is a set function that is cheap to
evaluate incrementally and empirically close to supermodular.

Three selectors are provided so the comparison is against real alternatives
rather than against nothing:

* `greedy_g`      -- add the coil that most reduces mean g over the body.
* `greedy_snr`    -- add the coil with the most signal energy. The obvious
                     heuristic, and the one to beat: it ignores geometry
                     entirely and should therefore pick redundant neighbours.
* `random_subset` -- the null.

Coil compression (`svd_compress`) is included separately because it is what a
scanner actually does, and because it is *not* a selection rule -- reporting it
alongside makes clear how much of the gap is "which elements" and how much is
"you are allowed to mix them".

Nothing here uses the reference image: g is built from the estimated
sensitivities and the sampling pattern, both available at scan time.
"""
from __future__ import annotations

import numpy as np


def gmap_for(S, R, Psi=None, body=None):
    """Mean analytic g over the body for a given coil subset."""
    nc, ny, nx = S.shape
    if Psi is None:
        Psi = np.eye(nc)
    Pinv = np.linalg.inv(Psi + 1e-12 * np.trace(Psi).real / nc * np.eye(nc))
    ry = ny // R
    tot, n = 0.0, 0
    for y in range(ry):
        idx = [(y + r * ry) % ny for r in range(R)]
        Sy = S[:, idx, :]                              # (nc, R, nx)
        # G[x] = Sy[:,:,x]^H Pinv Sy[:,:,x]
        A = np.einsum('ij,jrx->irx', Pinv, Sy)
        G = np.einsum('irx,isx->rsx', Sy.conj(), A)    # (R, R, nx)
        G = np.moveaxis(G, -1, 0)                      # (nx, R, R)
        # `estimate_maps_acs` zeroes the sensitivities outside the object, so G
        # is *exactly* zero there and a regularizer scaled by trace(G) is zero
        # too -- every pixel then throws and the whole subset scores inf. The
        # floor is therefore an absolute one, set from the global scale of this
        # slice, and degenerate pixels are skipped rather than inverted.
        tr = np.real(np.trace(G, axis1=1, axis2=2))
        scale = float(tr.max()) if tr.size else 0.0
        if scale <= 0:
            continue
        ok = tr > 1e-8 * scale
        if not ok.any():
            continue
        Gk = G[ok] + 1e-9 * scale * np.eye(R)
        try:
            Gi = np.linalg.inv(Gk)
        except np.linalg.LinAlgError:
            return np.inf
        g = np.sqrt(np.maximum(
            np.real(np.einsum('xrr->xr', Gi) * np.einsum('xrr->xr', Gk)), 0.0))
        for r in range(R):
            m = (body[idx[r], :] if body is not None else np.ones(nx, bool))[ok]
            if m.any():
                tot += float(g[m, r].sum()); n += int(m.sum())
    return tot / max(n, 1)


def greedy_g(S, R, K, Psi=None, body=None):
    """Greedily add the coil that most reduces mean g over the body.

    Cold start matters more than the greedy step. g is undefined for a subset
    smaller than R -- the system is under-determined -- and scoring those
    subsets by a placeholder meant the first R-1 coils were picked by index
    order, which for K=6 at R=4 left half the selection arbitrary and made this
    selector lose to a plain SNR ranking. Instead each partial subset is scored
    at the *highest acceleration it can actually support*, R_eff = |subset|, so
    the criterion is the same quantity throughout and is discriminative from
    the second coil onwards. The first coil, where no acceleration is
    supportable and every subset scores alike, is taken by signal energy.
    """
    nc = S.shape[0]
    chosen: list[int] = list(greedy_snr(S, R, 1, Psi, body))
    while len(chosen) < K:
        best, best_val = None, np.inf
        remaining = [c for c in range(nc) if c not in chosen]
        for c in remaining:
            cand = chosen + [c]
            R_eff = min(R, len(cand))         # highest R this subset supports
            sub = np.ascontiguousarray(S[cand])
            P = None if Psi is None else Psi[np.ix_(cand, cand)]
            val = gmap_for(sub, R_eff, P, body)
            if val < best_val:
                best, best_val = c, val
        # every candidate scoring inf would otherwise append None and corrupt
        # the index list on the next pass
        chosen.append(best if best is not None else remaining[0])
    return sorted(chosen)


def greedy_snr(S, R, K, Psi=None, body=None):
    """Pick the K coils with the most signal energy inside the body."""
    e = np.array([float(np.sum(np.abs(S[c][body if body is not None
                                        else slice(None)]) ** 2))
                  for c in range(S.shape[0])])
    return sorted(np.argsort(e)[::-1][:K].tolist())


def random_subset(S, R, K, Psi=None, body=None, seed=0):
    return sorted(np.random.default_rng(seed).choice(
        S.shape[0], K, replace=False).tolist())


def svd_compress(ksp, K, return_matrix=False, calibration=None):
    """Standard SVD coil compression: K linear combinations of all channels.

    Not a selection rule -- included as the upper reference, since it may mix
    every physical channel and therefore answers a strictly easier question.
    """
    nc = ksp.shape[0]
    if not 1 <= K <= nc:
        raise ValueError(f"cannot retain K={K} from {nc} channels")
    # Fit on explicitly available calibration data, then apply the frozen
    # transform to the acquisition. The legacy default fits on all of ksp;
    # that is an oracle if ksp includes unavailable retrospective samples.
    fit = ksp if calibration is None else np.asarray(calibration)
    if fit.shape[0] != nc or fit.size == 0:
        raise ValueError("calibration must contain nonempty data from every channel")
    X = fit.reshape(nc, -1)
    U, _, _ = np.linalg.svd(X @ X.conj().T)
    W = U[:, :K].conj().T
    out = np.einsum('ij,jyx->iyx', W, ksp)
    return (out, W) if return_matrix else out


def sense_variance_risk(S, R, Psi=None, body=None, aggregate="mean-log"):
    """Predicted SENSE noise risk for a coil subset.

    For one folded pixel set the covariance of the generalized least-squares
    SENSE estimate is, up to the common acquisition-noise scale,

        cov(rho_hat) = (S^H Psi^-1 S)^-1.

    Unlike a g-only criterion, its diagonal already contains both receive
    strength and unfolding geometry.  ``mean-log`` is the geometric mean of
    the voxel variances (returned in log units); it is less dominated by a few
    nearly singular boundary pixels than the arithmetic mean.  ``mean`` and
    ``p90`` are exposed for ablations.

    The sensitivity maps must retain a common normalization across candidate
    subsets.  In practice this means estimating maps with the full survey array
    and then selecting rows, rather than renormalizing each candidate subset.
    """
    nc, ny, nx = S.shape
    if ny % R:
        raise ValueError(f"R={R} must divide ny={ny}")
    if Psi is None:
        Psi = np.eye(nc)
    if Psi.shape != (nc, nc):
        raise ValueError(f"Psi has shape {Psi.shape}, expected {(nc, nc)}")
    body = np.ones((ny, nx), bool) if body is None else np.asarray(body, bool)

    floor = 1e-12 * max(float(np.trace(Psi).real) / nc, 1e-30)
    Pinv = np.linalg.inv(Psi + floor * np.eye(nc))
    step = ny // R
    values = []
    for y in range(step):
        idx = [(y + r * step) % ny for r in range(R)]
        Sy = S[:, idx, :]                              # (nc, R, nx)
        G = np.einsum('crx,cd,dsx->xrs', Sy.conj(), Pinv, Sy)
        tr = np.trace(G, axis1=1, axis2=2).real
        scale = float(tr.max()) if tr.size else 0.0
        if scale <= 0:
            continue
        ok = tr > 1e-8 * scale
        if not ok.any():
            continue
        Gi = np.linalg.inv(G[ok] + 1e-9 * scale * np.eye(R))
        var = np.real(np.diagonal(Gi, axis1=1, axis2=2))
        for r, iy in enumerate(idx):
            m = body[iy, ok]
            if m.any():
                values.append(var[m, r])

    if not values:
        return np.inf
    v = np.concatenate(values)
    v = v[np.isfinite(v) & (v > 0)]
    if not len(v):
        return np.inf
    if aggregate == "mean-log":
        return float(np.mean(np.log(v)))
    if aggregate == "mean":
        return float(np.mean(v))
    if aggregate == "p90":
        return float(np.quantile(v, 0.90))
    raise ValueError(f"unknown aggregate {aggregate!r}")


def greedy_noise_risk(S, R, K, Psi=None, body=None, aggregate="mean-log"):
    """Greedily minimize the exact predicted SENSE variance.

    Partial subsets are evaluated at the highest acceleration they can support,
    avoiding a rank-deficient cold start while preserving the same criterion at
    every greedy step.
    """
    if K > S.shape[0]:
        raise ValueError(f"cannot select K={K} from {S.shape[0]} coils")
    chosen = []
    while len(chosen) < K:
        remaining = [c for c in range(S.shape[0]) if c not in chosen]
        vals = []
        for c in remaining:
            cand = chosen + [c]
            # Cartesian folding requires R_eff to divide ny.  For example,
            # three selected coils cannot be scored at R=3 on a 320-line
            # acquisition, so retain the largest supported divisor instead.
            limit = min(R, len(cand))
            r_eff = max(r for r in range(1, limit + 1) if S.shape[1] % r == 0)
            P = None if Psi is None else Psi[np.ix_(cand, cand)]
            vals.append(sense_variance_risk(
                np.ascontiguousarray(S[cand]), r_eff, P, body, aggregate))
        chosen.append(remaining[int(np.argmin(vals))])
    return sorted(chosen)


def coil_images(ksp, acs):
    """Low-resolution coil images from the ACS block, un-normalised.

    `estimate_maps_acs` divides the sensitivities by their root-sum-of-squares,
    so `sum_c |S_c|^2 = 1` everywhere inside the object and the maps carry no
    information about how much signal a coil actually receives. Selecting on
    them alone is therefore blind to sensitivity magnitude, which is half of
    Pruessmann's SNR expression. These are the same coil images before that
    normalisation, and they supply the missing half.
    """
    from simulate import ifft2c
    nc, ny, nx = ksp.shape
    blk = np.zeros_like(ksp)
    blk[:, acs, :] = ksp[:, acs, :]
    return ifft2c(blk)


def greedy_snr_law(S, R, K, Psi=None, body=None, C=None):
    """Greedy selection maximising Pruessmann's reconstructed SNR.

    The replication in `validate_snr_law.py` confirmed the paper's central
    claim to within 0.03%:

        SNR_R = SNR_full / (g sqrt(R))

    Both selectors above optimise one factor and ignore the other -- `greedy_g`
    minimises the geometry penalty while ignoring how much signal the chosen
    coils receive, `greedy_snr` maximises received signal while ignoring
    conditioning -- so each loses on the axis it does not see. This uses the
    validated law itself as the objective, maximising the mean of

        sqrt( sum_c |C_c|^2 ) / g

    over the body, with C the un-normalised coil images. sqrt(R) is common to
    every subset and drops out of the argmax.
    """
    nc = S.shape[0]
    if C is None:
        raise ValueError("greedy_snr_law needs un-normalised coil images C")
    body = body if body is not None else np.ones(S.shape[1:], bool)

    def score(cand):
        R_eff = min(R, len(cand))
        P = None if Psi is None else Psi[np.ix_(cand, cand)]
        g = gmap_for(np.ascontiguousarray(S[cand]), R_eff, P, body)
        if not np.isfinite(g) or g <= 0:
            return -np.inf
        sig = float(np.mean(np.sqrt((np.abs(C[cand][:, body]) ** 2).sum(0))))
        return sig / g                        # SNR_full / g, up to sqrt(R)

    chosen = list(greedy_snr(S, R, 1, Psi, body))
    while len(chosen) < K:
        remaining = [c for c in range(nc) if c not in chosen]
        vals = [score(chosen + [c]) for c in remaining]
        chosen.append(remaining[int(np.argmax(vals))])
    return sorted(chosen)


SELECTORS = {"greedy-g": greedy_g, "greedy-snr": greedy_snr,
             "greedy-noise-risk": greedy_noise_risk, "random": random_subset}
