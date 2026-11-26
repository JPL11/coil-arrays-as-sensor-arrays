"""An effective g-factor for a nonlinear reconstructor, and what it hides.

The g-factor is *defined* only for linear reconstruction: it comes out of the
SENSE normal equations as sqrt[(S^H Psi^-1 S)^-1_pp (S^H Psi^-1 S)_pp], and a
neural network has no such matrix. But the quantity g measures -- noise
amplification relative to a fully sampled acquisition -- is defined for any
estimator, and the pseudo-replica method (Robson et al., Magn. Reson. Med.
60:895, 2008) measures it by pushing many independent noise realizations
through the reconstructor and taking the per-pixel standard deviation. That is
exactly what `validate_snr_law.py` already does to confirm SENSE's analytic g
to within 0.03%, which is what licenses using it on a method where no analytic
g exists.

    g_eff(p) = sigma_R(p) / ( sigma_full(p) * sqrt(R) )

**Variance alone is not a quality measure, and for a learned reconstructor it
is actively misleading.** A network that ignored the data and emitted a fixed
prior would have sigma_R = 0 and therefore g_eff = 0 -- a perfect score for a
useless reconstruction. Any low g_eff reported for a nonlinear method is
uninterpretable without the companion term, so the error is decomposed here:

    E[(x_hat - x)^2]  =  bias^2  +  variance

over the same replicas. A linear, unregularized reconstructor puts nearly
everything in the variance term; a regularized or learned one trades variance
for bias, and the trade is only visible if both are reported. `mse_check`
confirms the decomposition closes.

The third measurement is *where* the noise lands. The analytic g-map has
characteristic banding at multiples of ny/R -- `paper_P1_gfactor_maps.png` --
and the SENSE difference images in P2 show the same pattern, which is how the
study establishes that g localizes error rather than merely summarizing it. If
a learned reconstructor's noise is uncorrelated with the g-map, its errors are
not governed by array geometry at all; if it bands the same way, they are. That
correlation is reported per method.

Noise is added to the *acquired* k-space with the covariance measured from that
same acquisition's signal-free corners, scaled by `--alpha`, so the added noise
is a physically consistent perturbation of the real array rather than white
noise imposed on it.
"""
from __future__ import annotations

import os, sys, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import numpy as np
import pandas as pd

from scipy import ndimage

import fastmri_data as fd
from simulate import ifft2c, uniform_mask
from recon import (sense_recon, grappa_recon, zerofill_recon,
                   estimate_maps_acs, optimal_combine)
from learned_recon import (load_varnet, varnet_recon, DEFAULT_CKPT,
                           load_unet, unet_recon, UNET_CKPT)
from gpu_guard import gpu_slot

ROOT = "../datasets/fastmri/multicoil_val"


def correlated_noise(shape, Psi, rng):
    """White complex noise coloured by the Cholesky factor of Psi (as in
    `validate_snr_law.py`, so the two noise models are the same one)."""
    nc = Psi.shape[0]
    w = (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2)
    L = np.linalg.cholesky(Psi + 1e-12 * np.trace(Psi).real / nc * np.eye(nc))
    return np.einsum('ij,j...->i...', L, w)


def replicas_for_slice(kf, R, n_acs, lam, alpha, n_rep, seed, varnet, unet=None):
    """Every method's replica stack for one slice at one acceleration.

    Noise is added to the full-readout k-space -- the acquisition as it exists --
    and the oversampling-removed copy the classical estimators use is derived
    from the *same* noisy array, so no method sees a different noise draw.
    """
    Psi, _ = fd.noise_covariance(kf)
    k0 = fd.remove_readout_oversampling(kf)
    nc, ny, nx = k0.shape
    if ny % R:
        return None

    truth = np.sqrt((np.abs(ifft2c(k0)) ** 2).sum(0))
    bm = fd.body_mask(truth)
    mask_s, acs = uniform_mask(ny, nx, R, n_acs)
    mask_f, _   = uniform_mask(ny, kf.shape[-1], R, n_acs)

    # Maps and the analytic g come from the noiseless acquisition: they are
    # properties of the array and the sampling, not of a particular replica.
    S = estimate_maps_acs(k0, acs)
    # Psi must be passed here and in the replicas below. The injected noise is
    # coloured by chol(Psi), so reconstructing white solves a different
    # estimator than the one whose g is being predicted, and the analytic g is
    # then computed on S^H S rather than S^H Psi^-1 S. That mismatch is the
    # likeliest source of the ~5x gap between g_analytic and the pseudo-replica
    # g_eff, which was never reported because only the scale-free correlation
    # was quoted.
    _, gmap = sense_recon(k0, S, R, return_gfactor=True, Psi=Psi)

    # Matched sampling. SENSE reconstructs the strictly uniform lines, so every
    # arm that can does too: zero-filling and the U-Net get the uniform mask and
    # GRAPPA does not paste its calibration block back in. They used to receive
    # uniform+ACS -- at R=8 on 320 lines, 61 lines against SENSE's 40 -- while
    # every g_eff was normalised by the same nominal sqrt(R). The two networks
    # were trained on masks with a sampled centre (VarNet returns NaN without
    # one, the U-Net's NRMSE doubles), so they keep the ACS mask and are
    # normalised by their own R_eff; `R_eff` is recorded per method.
    uni_s = np.zeros_like(mask_s); uni_s[::R, :] = True
    reff = {"VarNet": mask_f.shape[0] / mask_f[:, 0].sum(),
            "U-Net": mask_s.shape[0] / mask_s[:, 0].sum()}

    rng = np.random.default_rng(seed)
    stacks = {"R=1 combine": [], "Zero-filled": [], "SENSE": [],
              "SENSE+DL": [], "GRAPPA": []}
    if varnet is not None:
        stacks["VarNet"] = []
    if unet is not None:
        stacks["U-Net"] = []

    for _ in range(n_rep):
        kf_i = kf + alpha * correlated_noise(kf.shape, Psi, rng)
        k_i  = fd.remove_readout_oversampling(kf_i)
        # Psi-optimal, because that is the reference g is defined against; the
        # unweighted matched filter used here before is noisier on correlated
        # coils and deflated every g_eff by ~14% (`diagnose_g_gap.py`).
        stacks["R=1 combine"].append(np.abs(optimal_combine(ifft2c(k_i), S, Psi)))
        stacks["Zero-filled"].append(np.abs(zerofill_recon(k_i * uni_s, S)))
        stacks["SENSE"].append(np.abs(sense_recon(k_i, S, R, Psi=Psi)))
        stacks["SENSE+DL"].append(np.abs(sense_recon(k_i, S, R, lam=lam, Psi=Psi)))
        try:
            stacks["GRAPPA"].append(np.abs(grappa_recon(k_i, acs, R, maps=S,
                                                        restore_acs=False)))
        except ValueError:
            stacks.pop("GRAPPA", None)
        if varnet is not None:
            stacks["VarNet"].append(varnet_recon(kf_i, mask_f, n_acs,
                                                 crop_ro=True, model=varnet))
        if unet is not None:
            stacks["U-Net"].append(unet_recon(k_i, mask_s, model=unet))

    return {k: np.stack(v) for k, v in stacks.items() if v}, truth, bm, gmap, reff


def predictors(truth, bm, gmap):
    """Candidate explanations for where a reconstructor's noise lands.

    `g` is array geometry -- the analytic prediction, which the SENSE result
    says governs classical error. `grad` is local image structure, the obvious
    alternative: Kleineisel et al. describe learned reconstructors as
    suppressing noise "almost everywhere except around hard edges", which is a
    statement about edges rather than about coils. `signal` and `dist` are
    controls -- brightness and position -- so that a structure effect cannot be
    claimed when the real driver is that edges occur in bright central tissue.
    """
    gy, gx = np.gradient(truth)
    grad = np.sqrt(gy ** 2 + gx ** 2)
    ys, xs = np.nonzero(bm)
    cy, cx = ys.mean(), xs.mean()
    Y, X = np.mgrid[0:truth.shape[0], 0:truth.shape[1]]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    return {"g": gmap, "grad": grad, "signal": truth, "dist": dist}


def _z(v):
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)


def regress(sig, preds, keys):
    """Standardized OLS of noise on the predictors, plus each one alone.

    Coefficients are on z-scored variables so they are directly comparable, and
    the single-predictor R^2 values are reported next to the joint fit because
    `g` and `grad` are not orthogonal -- a small joint coefficient can mean
    "explains nothing" or "explains the same thing as its neighbour", and the
    two are told apart by comparing the solo fits.
    """
    y = _z(sig)
    A = np.column_stack([_z(preds[k]) for k in keys])
    beta, *_ = np.linalg.lstsq(np.column_stack([A, np.ones(len(y))]), y, rcond=None)
    resid = y - np.column_stack([A, np.ones(len(y))]) @ beta
    out = {f"beta_{k}": float(b) for k, b in zip(keys, beta[:-1])}
    out["r2_joint"] = float(1 - resid.var() / max(y.var(), 1e-30))
    for i, k in enumerate(keys):
        r = float(np.corrcoef(A[:, i], y)[0, 1])
        out[f"corr_{k}"] = r
        out[f"r2_{k}_only"] = r ** 2
    return out


def analyse(stacks, truth, bm, gmap, R, alpha, sig_floor=0.05, reff=None):
    """Effective g, bias/variance split, and g-map correlation, per method.

    Two pixel sets, deliberately different. The error decomposition runs over
    the whole body mask, so it stays comparable with every other metric in the
    study. `g_eff` runs over a subset, because it is a *ratio* and is undefined
    wherever the fully sampled reconstruction carries no noise to divide by --
    which is not a rare edge case: `estimate_maps_acs` zeroes the coil maps
    outside the object, the matched-filter combine is then identically zero
    there for every replica, and that support does not coincide with the body
    mask. Left in, those pixels produced mean g_eff of 1e22 while the median
    stayed at 0.26; the mean is reported here only because the analytic g is
    itself heavily right-skewed, so a median-to-mean comparison would flatter
    the measurement. Both statistics are given for both quantities, on the same
    pixels, and `valid_frac` says how many survived.

    `g_eff` carries the per-method gain and `g_eff_raw` does not, and the two
    answer different questions. The gain is fitted to the replica *mean*, so it
    is a property of the bias: SENSE at R=8 on 24-line maps fits a = 0.43, and
    that factor alone took a measured g of 12.8 (analytic 12.8) down to 4.5 --
    the whole of the "5x gap" the 2026-09-12 panel flagged. `g_eff_raw` is the
    quantity the analytic g predicts and is the one to compare against it. It
    is only meaningful where the output scale is physical, which holds for the
    classical estimators and is not guaranteed for a network, so `gain` is
    recorded and any method with gain far from 1 should be read on `g_eff`.
    """
    tb = truth[bm]

    def gain(stack):
        """One scalar per *method*, fitted on the replica mean and applied to
        every replica -- fitting per replica would absorb part of the very
        variance being measured."""
        # Slope of the reconstruction on the reference, inverted -- not the
        # least-squares gain of the reference on the reconstruction. The replica
        # mean still holds the acquisition's native noise draw, and a gain with
        # that noise in the *regressor* is attenuated by it: SENSE at R=8 fitted
        # 0.40 and had its g_eff cut from 13.6 to 5.2 by nothing but its own
        # noise. With the noise in the dependent variable the slope is unbiased.
        m = stack.mean(0)[bm]
        d = float(np.sum(tb * m))
        return float(np.sum(tb * tb)) / d if d > 0 else 1.0

    sig_full_raw = stacks["R=1 combine"].std(0)[bm]
    sig_full = sig_full_raw * gain(stacks["R=1 combine"])
    valid = sig_full > sig_floor * np.median(sig_full)
    gb = gmap[bm]
    P = predictors(truth, bm, gmap)
    Pv = {k: v[bm][valid] for k, v in P.items()}
    KEYS = ["g", "grad", "signal", "dist"]

    rows = []
    for name, stack in stacks.items():
        if name == "R=1 combine":
            continue
        a = gain(stack)
        sig = stack.std(0)[bm] * a
        bias = stack.mean(0)[bm] * a - tb
        mse = float(np.mean((stack[:, bm] * a - tb[None]) ** 2))
        Re = (reff or {}).get(name, R)
        g_eff = sig[valid] / sig_full[valid] / np.sqrt(Re)
        g_raw = stack.std(0)[bm][valid] / sig_full_raw[valid] / np.sqrt(Re)
        rows.append(dict(
            method=name, R=R, R_eff=float(Re), alpha=alpha,
            g_eff=float(np.mean(g_eff)), g_eff_med=float(np.median(g_eff)),
            g_eff_raw=float(np.mean(g_raw)), g_eff_raw_med=float(np.median(g_raw)),
            gain=a,
            g_analytic=float(np.mean(gb[valid])),
            g_analytic_med=float(np.median(gb[valid])),
            valid_frac=float(valid.mean()),
            bias2=float(np.mean(bias ** 2)), variance=float(np.mean(sig ** 2)),
            mse=mse, mse_check=float(np.mean(bias ** 2) + np.mean(sig ** 2)),
            bias_frac=float(np.mean(bias ** 2) / max(mse, 1e-30)),
            corr_sigma_g=float(np.corrcoef(sig[valid], gb[valid])[0, 1]),
            corr_abserr_g=float(np.corrcoef(np.abs(bias[valid]), gb[valid])[0, 1]),
            nrmse=float(np.sqrt(mse) / np.sqrt(np.mean(tb ** 2))),
            **regress(sig[valid], Pv, KEYS),
        ))
    return rows


def main():
    """Admission-controlled entry point; see `src/gpu_guard`."""
    with gpu_slot(need_mb=3500, label="effective_g"):
        return _main()


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-vols", type=int, default=4)
    ap.add_argument("--n-slices", type=int, default=1)
    ap.add_argument("--R", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--n-rep", type=int, default=64)
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.1],
                    help="added noise std as a multiple of the measured array "
                         "noise. Pseudo-replica assumes the perturbation is "
                         "small enough that a nonlinear reconstructor is "
                         "locally linear; pass several values to check that "
                         "g_eff does not depend on which one was chosen.")
    ap.add_argument("--n-acs", type=int, default=24)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--take", type=int, default=None)
    ap.add_argument("--out", default="results/effective_g.csv")
    a = ap.parse_args()

    vols, _ = fd.list_volumes(ROOT)
    vols = vols[:a.n_vols]
    if a.take:
        vols = vols[a.skip:a.skip + a.take]
    varnet = load_varnet() if os.path.exists(DEFAULT_CKPT) else None
    unet = load_unet() if os.path.exists(UNET_CKPT) else None
    print(f"{len(vols)} volumes | {a.n_rep} replicas | alpha={a.alpha} "
          f"| VarNet {'on ' + str(next(varnet.parameters()).device) if varnet else 'SKIPPED'}\n")

    rows = []
    for vi, v in enumerate(vols):
        mid = v["n_slices"] // 2
        for sl in range(mid - a.n_slices // 2, mid - a.n_slices // 2 + a.n_slices):
            kf = fd.load_slice(v["path"], sl)
            for R in a.R:
              for alpha in a.alpha:
                t0 = time.time()
                got = replicas_for_slice(kf, R, a.n_acs, a.lam, alpha,
                                         a.n_rep, 7000 + 13 * vi + sl, varnet, unet)
                if got is None:
                    continue
                stacks, truth, bm, gmap, reff = got
                for r in analyse(stacks, truth, bm, gmap, R, alpha, reff=reff):
                    r.update(volume=os.path.basename(v["path"]), slice=sl)
                    rows.append(r)
                print(f"  {os.path.basename(v['path'])[:28]:28s} sl={sl} R={R} "
                      f"alpha={alpha} {time.time()-t0:6.1f}s", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv(a.out, index=False)

    order = ["Zero-filled", "SENSE", "SENSE+DL", "GRAPPA", "VarNet", "U-Net"]
    for R in a.R:
      for alpha in a.alpha:
        d = df[(df.R == R) & (df.alpha == alpha)]
        if not len(d):
            continue
        g = d.groupby("method").agg(
            g_eff=("g_eff", "mean"), g_eff_med=("g_eff_med", "mean"),
            g_eff_raw=("g_eff_raw", "mean"), gain=("gain", "mean"),
            g_analytic=("g_analytic", "mean"),
            g_analytic_med=("g_analytic_med", "mean"),
            bias_frac=("bias_frac", "mean"), nrmse=("nrmse", "mean"),
            corr_sigma_g=("corr_sigma_g", "mean"),
            closes=("mse_check", "mean"), mse=("mse", "mean"),
        ).reindex(order).dropna(how="all")
        g["decomp_err"] = (g.closes - g.mse).abs() / g.mse
        print(f"\n=== R = {R}, alpha = {alpha} ({d.volume.nunique()} volumes, "
              f"{d.valid_frac.mean()*100:.1f}% of body pixels usable) ===")
        print(g.drop(columns=["closes", "mse"]).to_string(
            float_format=lambda x: f"{x:.4f}"))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
