"""Does the g-factor govern the error in a sensor-noise-dominated regime?

The central result of this study is that the analytic g-factor governs SENSE's
*noise* (r = 0.973) and not its *error* (r = 0.197), because at the SNR of a
room-temperature receiver array the residual is dominated by bias from imperfect
sensitivity maps. Both attempts to build a method on the g-map failed for that
reason.

That claim is stated absolutely and it should not be. It was measured in one
regime: high SNR, body-dominated noise with off-diagonal correlation up to 0.54.
Other receiver technologies sit elsewhere. SQUID and NV-diamond detection for
ultra-low-field MRI is **sensor-noise dominated** (the sample contributes almost
nothing at microtesla fields), the noise is close to uncorrelated across
elements, and the SNR is far lower. Nothing quantum enters the array inverse
directly -- the inverse sees only sensitivity profiles and a noise covariance --
but that combination of parameters is a genuinely different operating point, and
it is one we can set.

The prediction, which this is written to falsify: if bias dominates at high SNR
and noise dominates at low SNR, then `corr(|bias|, g)` should *rise* as SNR
falls, and the g-map should become the useful quantity it is not here.

This is a noise-regime study. It does not simulate quantum sensors and does not
claim to; the regime is the motivation, not the result.
"""
from __future__ import annotations

import os, sys, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import numpy as np
import pandas as pd

import fastmri_data as fd
from simulate import uniform_mask, ifft2c, fft2c
from recon import sense_recon, estimate_maps_acs
from run_effective_g import correlated_noise
from metrics import summarize

ROOT = "../datasets/fastmri/multicoil_balanced"


def whiten_psi(Psi, mode):
    """Noise covariance in the regime being modelled.

    `measured` keeps the array's own correlated covariance. `diagonal` keeps the
    per-element variances but removes the correlation, which is the SQUID-like
    case: independent sensors, no shared body noise. `uniform` additionally
    equalises the variances, the idealised limit.
    """
    if mode == "measured":
        return Psi
    d = np.real(np.diag(Psi))
    if mode == "diagonal":
        return np.diag(d).astype(Psi.dtype)
    return (np.eye(len(d)) * d.mean()).astype(Psi.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-vols", type=int, default=8)
    ap.add_argument("--n-slices", type=int, default=2)
    ap.add_argument("--R", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--n-rep", type=int, default=48)
    ap.add_argument("--n-acs", type=int, default=24)
    # alpha scales the added noise; large alpha = low SNR = the ULF regime
    ap.add_argument("--alpha", type=float, nargs="+",
                    default=[0.1, 1.0, 4.0, 12.0, 30.0])
    ap.add_argument("--psi", nargs="+", default=["measured", "diagonal"])
    ap.add_argument("--out", default="results/noise_regime.csv")
    a = ap.parse_args()

    vols, _ = fd.list_volumes(ROOT)
    vols = [v for v in vols if v["n_coils"] >= 12][:a.n_vols]
    print(f"{len(vols)} volumes | alpha {a.alpha} | Psi {a.psi}\n", flush=True)

    rows = []
    for vi, v in enumerate(vols):
        mid = v["n_slices"] // 2
        for sl in range(mid - a.n_slices // 2, mid - a.n_slices // 2 + a.n_slices):
            kf = fd.load_slice(v["path"], sl)
            k0 = fd.remove_readout_oversampling(kf)
            nc, ny, nx = k0.shape
            truth = np.sqrt((np.abs(ifft2c(k0)) ** 2).sum(0))
            bm = fd.body_mask(truth)
            Psi0, _ = fd.noise_covariance(k0)
            for R in a.R:
                if ny % R:
                    continue
                mask, acs = uniform_mask(ny, nx, R, a.n_acs)
                S = estimate_maps_acs(k0, acs)
                for pmode in a.psi:
                    Psi = whiten_psi(Psi0, pmode)
                    # Psi goes to the estimator and to the g-map, per regime.
                    # It used to go to neither, so coloured noise was pushed
                    # through white SENSE and scored against a white g -- the
                    # defect FIXES_2026-09-12 #6 removed from run_effective_g
                    # and left live here.
                    x0, gmap = sense_recon(k0, S, R, return_gfactor=True, Psi=Psi)
                    gb = gmap[bm]
                    ph = np.exp(-1j * np.angle(x0))
                    for alpha in a.alpha:
                        rng = np.random.default_rng(11 * vi + sl)
                        stack, stack_mag = [], []
                        for _ in range(a.n_rep):
                            ki = k0 + alpha * correlated_noise(k0.shape, Psi, rng)
                            xi = sense_recon(ki, S, R, Psi=Psi)
                            # In-phase channel, not magnitude. |x| of a noise-
                            # dominated pixel is Rician: its mean floats up and
                            # its std saturates, which books noise as bias. That
                            # made the variance share *fall* from 0.83 to 0.52
                            # between alpha 4 and 30 in v2, for an estimator
                            # whose bias cannot depend on the noise level.
                            stack.append((xi * ph).real)
                            stack_mag.append(np.abs(xi))
                        st = np.stack(stack)
                        m = st.mean(0)[bm]
                        tb = truth[bm]
                        gain = float(np.sum(tb * m) / max(np.sum(m * m), 1e-30))
                        sig = st.std(0)[bm] * gain
                        bias = m * gain - tb
                        sm = np.stack(stack_mag)
                        mm = sm.mean(0)[bm]
                        gm = float(np.sum(tb * mm) / max(np.sum(mm * mm), 1e-30))
                        var_frac_mag = float(
                            np.mean((sm.std(0)[bm] * gm) ** 2) /
                            max(np.mean((mm * gm - tb) ** 2) +
                                np.mean((sm.std(0)[bm] * gm) ** 2), 1e-30))
                        # The replica mean still carries sig^2/n of noise, which
                        # lands in `bias` and caps the variance share at n/(n+1).
                        b2 = max(float(np.mean(bias ** 2) - np.mean(sig ** 2) / a.n_rep), 0.0)
                        mse = b2 + float(np.mean(sig ** 2))
                        rows.append(dict(
                            volume=os.path.basename(v["path"]), slice=sl, R=R,
                            psi=pmode, alpha=alpha, n_coils=nc,
                            var_frac_mag=var_frac_mag,
                            corr_sigma_g=float(np.corrcoef(sig, gb)[0, 1]),
                            corr_bias_g=float(np.corrcoef(np.abs(bias), gb)[0, 1]),
                            # the quantity that decides whether a g-gated method
                            # could work: total per-pixel error, not bias alone.
                            # Measuring only the bias term was the gap in the
                            # first version of this experiment -- at low SNR the
                            # error is mostly variance, and variance *is*
                            # g-structured, so bias-vs-g answers a different
                            # question than "can g tell you where the error is".
                            corr_toterr_g=float(np.corrcoef(
                                np.sqrt(bias ** 2 + sig ** 2), gb)[0, 1]),
                            var_frac=float(np.mean(sig ** 2) / max(mse, 1e-30)),
                            nrmse=float(np.sqrt(mse) / np.sqrt(np.mean(tb ** 2))),
                            ssim=summarize(fd.apply_mask(truth, bm),
                                           fd.apply_mask(st.mean(0) * gain, bm))["ssim"]))
        print(f"  {vi+1}/{len(vols)} volumes  ({len(rows)} rows)", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv(a.out, index=False)

    print("\nDoes g start governing the ERROR once noise dominates it?")
    for R in a.R:
        d = df[df.R == R]
        if not len(d):
            continue
        print(f"\n=== R = {R} ===")
        print(f"{'Psi':10s} {'alpha':>6s} {'variance share':>15s} "
              f"{'corr(noise,g)':>14s} {'corr(|bias|,g)':>15s} "
              f"{'corr(TOTAL err,g)':>18s} {'NRMSE':>7s}")
        for pmode in a.psi:
            for alpha in a.alpha:
                s = d[(d.psi == pmode) & (d.alpha == alpha)]
                if not len(s):
                    continue
                print(f"{pmode:10s} {alpha:6.1f} {s.var_frac.mean():15.3f} "
                      f"{s.corr_sigma_g.mean():14.3f} {s.corr_bias_g.mean():15.3f} "
                      f"{s.corr_toterr_g.mean():18.3f} {s.nrmse.mean():7.3f}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
