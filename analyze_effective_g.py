"""Volume-level bootstrap for the effective-g and noise-predictor results.

Every other result in this study carries a cluster bootstrap over whole volumes;
the effective-g experiment did not, and it carries the one column the literature
does not already contain (Kleineisel et al. measure empirical g for a learned
reconstructor, but never against the analytic SENSE g-factor). This puts the
same interval on it, and on the regression that asks what *does* predict a
learned reconstructor's noise once geometry has failed to.

Resampling is over volumes, not slices: three slices from one brain are not
three independent observations of a reconstructor.
"""
from __future__ import annotations

import argparse, glob
import numpy as np
import pandas as pd

ORDER = ["Zero-filled", "SENSE", "SENSE+DL", "GRAPPA", "VarNet", "U-Net"]
KEYS = ["g", "grad", "signal", "dist"]


def boot(df, col, n=5000, seed=0):
    """Mean and 95% interval, resampling whole volumes."""
    rng = np.random.default_rng(seed)
    per_vol = df.groupby("volume")[col].mean()
    vols = per_vol.index.to_numpy()
    if len(vols) < 2:
        return per_vol.mean(), np.nan, np.nan
    draws = [per_vol.reindex(rng.choice(vols, len(vols), replace=True)).mean()
             for _ in range(n)]
    return per_vol.mean(), *np.nanpercentile(draws, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/effective_g_shard_*.csv")
    ap.add_argument("--out", default="results/effective_g_full.csv")
    a = ap.parse_args()

    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(a.glob))])
    df.to_csv(a.out, index=False)
    print(f"{df.volume.nunique()} volumes, {df.slice.nunique()} slice positions, "
          f"{len(df)} rows\n")

    for R in sorted(df.R.unique()):
        d = df[df.R == R]
        print(f"=== R = {R} "
              f"(analytic g mean {d.g_analytic.mean():.2f}, "
              f"median {d.g_analytic_med.mean():.2f}) ===")
        print(f"{'method':13s} {'g_eff':>22s} {'corr(sigma, g-map)':>26s} {'NRMSE':>8s}")
        for m in ORDER:
            s = d[d.method == m]
            if not len(s):
                continue
            g = boot(s, "g_eff"); c = boot(s, "corr_sigma_g")
            print(f"{m:13s} {g[0]:8.3f} [{g[1]:.3f}, {g[2]:.3f}] "
                  f"{c[0]:9.3f} [{c[1]:+.3f}, {c[2]:+.3f}] {s.nrmse.mean():8.3f}")
        print()

    R = max(df.R.unique())
    d = df[df.R == R]
    print(f"=== What predicts the noise? standardized OLS, R = {R} ===")
    print(f"{'method':13s} " + " ".join(f"{'beta_'+k:>10s}" for k in KEYS) + f" {'R2':>7s}")
    for m in ORDER:
        s = d[d.method == m]
        if not len(s):
            continue
        betas = " ".join(f"{s['beta_'+k].mean():10.3f}" for k in KEYS)
        print(f"{m:13s} {betas} {s.r2_joint.mean():7.3f}")
    print(f"\n=== single-predictor R^2 (collinearity check), R = {R} ===")
    print(f"{'method':13s} " + " ".join(f"{k:>10s}" for k in KEYS))
    for m in ORDER:
        s = d[d.method == m]
        if not len(s):
            continue
        print(f"{m:13s} " + " ".join(f"{s['r2_'+k+'_only'].mean():10.3f}" for k in KEYS))

    print(f"\n=== intervals on the dominant coefficients, R = {R} ===")
    for m in ORDER:
        s = d[d.method == m]
        if not len(s):
            continue
        bg = boot(s, "beta_g"); bd = boot(s, "beta_grad")
        print(f"  {m:13s} beta_g {bg[0]:+.3f} [{bg[1]:+.3f}, {bg[2]:+.3f}]   "
              f"beta_grad {bd[0]:+.3f} [{bd[1]:+.3f}, {bd[2]:+.3f}]")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
