"""Coil selection: which K of N elements, chosen by the paper's own SNR law.

The aperture sweep answers "how many coils" and says nothing about "which".
That is the sensor-selection problem from array processing, and in MRI it is
distinct from coil compression: compression forms K linear combinations of N
channels and needs all N receivers present, while selection keeps K physical
elements and is what applies when channels are limited by hardware or when one
has failed.

Four selectors, each optimising a different thing:

  greedy-g        minimise mean analytic g -- the geometry penalty alone
  greedy-snr      maximise received signal energy -- the obvious heuristic
  greedy-snr-law  maximise SNR_full / g, i.e. Pruessmann's reconstructed SNR
  random          the null

and `svd-compress` reported alongside as the upper reference, since it is
allowed to mix channels and therefore answers an easier question.

The point of the comparison is that the first two each optimise one factor of
the validated law `SNR_R = SNR_full / (g sqrt(R))` and ignore the other, so each
should lose on the axis it cannot see. Nothing here uses the reference image:
every objective is built from the ACS block and the sampling pattern.
"""
from __future__ import annotations

import os, sys, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import numpy as np
import pandas as pd

import fastmri_data as fd
from simulate import uniform_mask, ifft2c
from recon import estimate_maps_acs, sense_recon
from coil_select import (greedy_g, greedy_snr, random_subset, greedy_snr_law,
                         svd_compress, gmap_for, coil_images)
from metrics import summarize

ROOT = "../datasets/fastmri/multicoil_balanced"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-coils", type=int, default=20)
    ap.add_argument("--n-slices", type=int, default=3)
    ap.add_argument("--R", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--K", type=int, nargs="+", default=[5, 6, 8, 10, 12])
    ap.add_argument("--n-acs", type=int, default=24)
    ap.add_argument("--out", default="results/coil_select.csv")
    a = ap.parse_args()

    vols, _ = fd.list_volumes(ROOT)
    vols = [v for v in vols if v["n_coils"] >= a.min_coils]
    print(f"{len(vols)} volumes with >= {a.min_coils} coils\n", flush=True)

    rows = []
    for v in vols:
        mid = v["n_slices"] // 2
        for sl in range(mid - a.n_slices // 2, mid - a.n_slices // 2 + a.n_slices):
            k = fd.remove_readout_oversampling(fd.load_slice(v["path"], sl))
            nc, ny, nx = k.shape
            ref = np.sqrt((np.abs(ifft2c(k)) ** 2).sum(0))
            bm = fd.body_mask(ref); refm = fd.apply_mask(ref, bm)
            Psi, _ = fd.noise_covariance(k)

            def scored(img):
                x = fd.apply_mask(np.abs(img), bm)
                d = float(np.sum(x * x))
                return summarize(refm, (float(np.sum(refm * x)) / d if d > 0 else 1.0) * x)

            for R in a.R:
                if ny % R:
                    continue
                mask, acs = uniform_mask(ny, nx, R, a.n_acs)
                S = estimate_maps_acs(k, acs)
                C = coil_images(k, acs)
                sel = {
                    "greedy-g":       lambda: greedy_g(S, R, K, Psi, bm),
                    "greedy-snr":     lambda: greedy_snr(S, R, K, Psi, bm),
                    "greedy-snr-law": lambda: greedy_snr_law(S, R, K, Psi, bm, C),
                    "random":         lambda: random_subset(S, R, K, Psi, bm, seed=sl),
                }
                for K in a.K:
                    if K < R:
                        continue
                    for nm, fn in sel.items():
                        t0 = time.time(); idx = fn(); dt = time.time() - t0
                        ks = k[idx]; Ss = estimate_maps_acs(ks, acs)
                        m = scored(sense_recon(ks, Ss, R))
                        rows.append(dict(volume=os.path.basename(v["path"]), slice=sl,
                                         R=R, K=K, selector=nm, sel_s=dt,
                                         g_mean=gmap_for(S[idx], R,
                                                         Psi[np.ix_(idx, idx)], bm),
                                         **m))
                    kc = svd_compress(k, K); Sc = estimate_maps_acs(kc, acs)
                    rows.append(dict(volume=os.path.basename(v["path"]), slice=sl,
                                     R=R, K=K, selector="svd-compress", sel_s=0.0,
                                     g_mean=np.nan, **scored(sense_recon(kc, Sc, R))))
        print(f"  {os.path.basename(v['path'])[:30]:30s} done", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv(a.out, index=False)

    rng = np.random.default_rng(0)
    order = ["random", "greedy-g", "greedy-snr", "greedy-snr-law", "svd-compress"]
    for R in a.R:
        print(f"\n=== R = {R}  (SSIM, 95% CI over volumes) ===")
        print(f"{'K':>3s}  " + "  ".join(f"{o:>22s}" for o in order))
        for K in a.K:
            if K < R:
                continue
            cells = []
            for o in order:
                s = df[(df.R == R) & (df.K == K) & (df.selector == o)]
                if not len(s):
                    cells.append(" " * 22); continue
                pv = s.groupby("volume").ssim.mean()
                dr = [pv.reindex(rng.choice(pv.index, len(pv), replace=True)).mean()
                      for _ in range(2000)]
                lo, hi = np.nanpercentile(dr, [2.5, 97.5])
                cells.append(f"{pv.mean():.4f} [{lo:.3f},{hi:.3f}]")
            print(f"{K:>3d}  " + "  ".join(cells))
        # paired contrast: the law against the better of the two single-factor rules
        print("  paired greedy-snr-law - greedy-snr:")
        for K in a.K:
            if K < R:
                continue
            p = df[(df.R == R) & (df.K == K)].pivot_table(
                index=["volume", "slice"], columns="selector", values="ssim")
            if "greedy-snr-law" not in p or "greedy-snr" not in p:
                continue
            d = (p["greedy-snr-law"] - p["greedy-snr"]).dropna()
            vols_ = d.index.get_level_values(0).unique()
            dr = [d.loc[rng.choice(vols_, len(vols_), replace=True)].mean()
                  for _ in range(2000)]
            lo, hi = np.nanpercentile(dr, [2.5, 97.5])
            print(f"    K={K:2d}  {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]"
                  f"{'  *' if not (lo < 0 < hi) else ''}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
