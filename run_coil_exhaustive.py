"""How far is greedy coil selection from optimal?

`run_coil_select.py` selects greedily and the README says plainly that no
approximation bound applies: submodularity is not established for either
objective, so the gap to the exhaustive optimum is unknown rather than bounded.
For the smallest subset it is not unknown for long -- C(20,5) = 15504 is
enumerable in a couple of minutes per slice -- so the bound is measured here
instead of assumed.

Both objectives are enumerated, because they answer different questions: the
best achievable mean g says how good the greedy *optimiser* is, and the best
achievable SNR-law score says how good the greedy *criterion* is. A greedy rule
that nearly attains its own optimum while still losing on SSIM is a criterion
problem, not a search problem, and the two are only separable by enumeration.
"""
from __future__ import annotations

import os, sys, time, argparse, itertools
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import numpy as np
import pandas as pd

import fastmri_data as fd
from simulate import uniform_mask, ifft2c
from recon import estimate_maps_acs, sense_recon
from coil_select import gmap_for, coil_images, greedy_g, greedy_snr_law, greedy_snr
from metrics import summarize

ROOT = "../datasets/fastmri/multicoil_balanced"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--R", type=int, default=4)
    ap.add_argument("--n-vols", type=int, default=3)
    ap.add_argument("--n-acs", type=int, default=24)
    ap.add_argument("--out", default="results/coil_exhaustive.csv")
    a = ap.parse_args()

    vols, _ = fd.list_volumes(ROOT)
    vols = [v for v in vols if v["n_coils"] >= 20][:a.n_vols]
    from math import comb
    print(f"{len(vols)} volumes | enumerating C(20,{a.K}) = {comb(20, a.K)} subsets\n",
          flush=True)

    rows = []
    for v in vols:
        k = fd.remove_readout_oversampling(fd.load_slice(v["path"], v["n_slices"] // 2))
        nc, ny, nx = k.shape
        mask, acs = uniform_mask(ny, nx, a.R, a.n_acs)
        ref = np.sqrt((np.abs(ifft2c(k)) ** 2).sum(0))
        bm = fd.body_mask(ref); refm = fd.apply_mask(ref, bm)
        Psi, _ = fd.noise_covariance(k)
        S = estimate_maps_acs(k, acs); C = coil_images(k, acs)

        def scored(idx):
            ks = k[list(idx)]; Ss = estimate_maps_acs(ks, acs)
            x = fd.apply_mask(np.abs(sense_recon(ks, Ss, a.R)), bm)
            d = float(np.sum(x * x))
            return summarize(refm, (float(np.sum(refm * x)) / d if d > 0 else 1.0) * x)["ssim"]

        t0 = time.time()
        best_g, best_g_idx = np.inf, None
        best_law, best_law_idx = -np.inf, None
        for idx in itertools.combinations(range(nc), a.K):
            sub = np.ascontiguousarray(S[list(idx)])
            P = Psi[np.ix_(idx, idx)]
            g = gmap_for(sub, a.R, P, bm)
            if not np.isfinite(g) or g <= 0:
                continue
            if g < best_g:
                best_g, best_g_idx = g, idx
            sig = float(np.mean(np.sqrt((np.abs(C[list(idx)][:, bm]) ** 2).sum(0))))
            if sig / g > best_law:
                best_law, best_law_idx = sig / g, idx
        dt = time.time() - t0

        gr_g = tuple(greedy_g(S, a.R, a.K, Psi, bm))
        gr_law = tuple(greedy_snr_law(S, a.R, a.K, Psi, bm, C))
        gr_snr = tuple(greedy_snr(S, a.R, a.K, Psi, bm))
        row = dict(volume=os.path.basename(v["path"]), enum_s=dt,
                   ssim_exhaustive_g=scored(best_g_idx),
                   ssim_exhaustive_law=scored(best_law_idx),
                   ssim_greedy_g=scored(gr_g), ssim_greedy_law=scored(gr_law),
                   ssim_greedy_snr=scored(gr_snr),
                   g_exhaustive=best_g,
                   g_greedy=gmap_for(S[list(gr_g)], a.R, Psi[np.ix_(gr_g, gr_g)], bm),
                   match_g=int(tuple(sorted(best_g_idx)) == gr_g),
                   match_law=int(tuple(sorted(best_law_idx)) == gr_law))
        rows.append(row)
        print(f"  {os.path.basename(v['path'])[:28]:28s} {dt:6.1f}s  "
              f"greedy-g g={row['g_greedy']:.3f} vs best {best_g:.3f}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"\n=== K={a.K} of 20, R={a.R}, {len(df)} volumes ===")
    print(f"  mean g   : greedy {df.g_greedy.mean():.4f}   exhaustive {df.g_exhaustive.mean():.4f}"
          f"   gap {df.g_greedy.mean()-df.g_exhaustive.mean():+.4f}")
    print(f"  SSIM     : greedy-g   {df.ssim_greedy_g.mean():.4f}   best-g   {df.ssim_exhaustive_g.mean():.4f}")
    print(f"             greedy-law {df.ssim_greedy_law.mean():.4f}   best-law {df.ssim_exhaustive_law.mean():.4f}")
    print(f"             greedy-snr {df.ssim_greedy_snr.mean():.4f}")
    print(f"  greedy subset == exhaustive optimum: g {df.match_g.sum()}/{len(df)}, "
          f"law {df.match_law.sum()}/{len(df)}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
