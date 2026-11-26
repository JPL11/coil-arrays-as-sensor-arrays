"""How many noise samples does pre-whitening a receiver array actually need?

Pre-whitening is the step that makes the plain least-squares SENSE solve the
maximum-likelihood estimator, and it is the one step in this study that bought
almost nothing. The README blamed a poorly conditioned sample covariance --
the classical small-sample failure that motivates shrinkage in adaptive
beamforming. That explanation is wrong at the settings used: the default
corner region yields n = 1444 samples for nc = 20 coils, n/nc = 72, and the
sample covariance has condition number 12.0 with a Ledoit-Wolf shrinkage
intensity of 0.02 -- i.e. nothing to fix.

The interesting question is therefore not whether shrinkage helps here but
*where the sample estimator breaks and whether shrinkage moves that boundary*,
because a clinical noise pre-scan is short and the number of available samples
is a design parameter rather than a constant. Sweeping the corner fraction
takes n from 1444 down to 36 (n/nc = 1.8) on the same slices, so the
sample-starved regime is reached without changing anything else.

Three estimators, all shrinking towards the scaled identity: none (skip
pre-whitening entirely, the baseline), the sample covariance, Ledoit-Wolf, and
OAS. The reference image is built from the unwhitened k-space in every case --
whitening rescales each coil, so a reference computed after it is a different
image, the same trap documented for the whitened/unwhitened contrast in
`run_fastmri.py`.
"""
from __future__ import annotations

import os, sys, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import numpy as np
import pandas as pd

import fastmri_data as fd
from array_cov import corners, ESTIMATORS
from simulate import uniform_mask, ifft2c
from recon import sense_recon, estimate_maps_acs
from metrics import summarize

ROOT = "../datasets/fastmri/multicoil_balanced"


def whiten_with(ksp, Psi):
    nc = Psi.shape[0]
    Psi = Psi + 1e-12 * np.trace(Psi).real / nc * np.eye(nc)
    L = np.linalg.cholesky(Psi)
    return np.einsum('ij,jyx->iyx', np.linalg.inv(L), ksp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-coils", type=int, default=16)
    ap.add_argument("--n-vols", type=int, default=8)
    ap.add_argument("--n-slices", type=int, default=2)
    ap.add_argument("--R", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--fracs", type=float, nargs="+",
                    default=[0.06, 0.04, 0.03, 0.02, 0.015, 0.01])
    ap.add_argument("--n-acs", type=int, default=24)
    ap.add_argument("--out", default="results/noise_cov.csv")
    a = ap.parse_args()

    vols, _ = fd.list_volumes(ROOT)
    vols = [v for v in vols if v["n_coils"] >= a.min_coils][:a.n_vols]
    print(f"{len(vols)} volumes with >= {a.min_coils} coils "
          f"({sorted({v['n_coils'] for v in vols})})\n", flush=True)

    rows = []
    for v in vols:
        mid = v["n_slices"] // 2
        for sl in range(mid - a.n_slices // 2, mid - a.n_slices // 2 + a.n_slices):
            k = fd.remove_readout_oversampling(fd.load_slice(v["path"], sl))
            nc, ny, nx = k.shape
            ref = np.sqrt((np.abs(ifft2c(k)) ** 2).sum(0))
            bm = fd.body_mask(ref); refm = fd.apply_mask(ref, bm)

            def scored(img, R, mask, acs, S):
                x = fd.apply_mask(np.abs(img), bm)
                d = float(np.sum(x * x))
                x = (float(np.sum(refm * x)) / d if d > 0 else 1.0) * x
                return summarize(refm, x)["ssim"]

            for R in a.R:
                if ny % R:
                    continue
                mask, acs = uniform_mask(ny, nx, R, a.n_acs)
                # baseline: no pre-whitening at all
                S = estimate_maps_acs(k, acs)
                rows.append(dict(volume=os.path.basename(v["path"]), slice=sl,
                                 R=R, n_coils=nc, estimator="none", frac=np.nan,
                                 n_samples=np.nan, rho=np.nan, cond=np.nan,
                                 ssim=scored(sense_recon(k, S, R), R, mask, acs, S)))
                for frac in a.fracs:
                    X = corners(k, frac)
                    for nm, f in ESTIMATORS.items():
                        Psi, rho = f(X)
                        try:
                            kw = whiten_with(k, Psi)
                        except np.linalg.LinAlgError:
                            continue            # not positive definite
                        Sw = estimate_maps_acs(kw, acs)
                        rows.append(dict(
                            volume=os.path.basename(v["path"]), slice=sl, R=R,
                            n_coils=nc, estimator=nm, frac=frac,
                            n_samples=X.shape[1], rho=rho,
                            cond=float(np.linalg.cond(Psi)),
                            ssim=scored(sense_recon(kw, Sw, R), R, mask, acs, Sw)))
        print(f"  {os.path.basename(v['path'])[:30]:30s} done", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv(a.out, index=False)

    for R in a.R:
        d = df[df.R == R]
        if not len(d):
            continue
        base = d[d.estimator == "none"].ssim.mean()
        print(f"\n=== R = {R}   (no pre-whitening: {base:.4f}) ===")
        print(f"{'n/nc':>7s} {'n':>6s}  " +
              "  ".join(f"{e:>18s}" for e in ESTIMATORS))
        for frac in a.fracs:
            s = d[d.frac == frac]
            if not len(s):
                continue
            n = s.n_samples.iloc[0]; ratio = n / s.n_coils.iloc[0]
            cells = []
            for e in ESTIMATORS:
                t = s[s.estimator == e]
                cells.append(f"{t.ssim.mean():.4f} ({t.ssim.mean()-base:+.4f})"
                             if len(t) else " " * 18)
            print(f"{ratio:7.1f} {n:6.0f}  " + "  ".join(cells))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
