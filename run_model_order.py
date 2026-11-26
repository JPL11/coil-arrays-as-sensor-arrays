"""Model order selection for the ESPIRiT subspace (Wax & Kailath, 1985).

ESPIRiT decides how many calibration kernels span the signal subspace with a
hand-picked singular-value threshold `c`. `run_espirit_tune.py` sweeps 48
settings and reconstruction quality swings 0.14 SSIM across plausible ones, so
that threshold is doing real work and there is no principled reason for any
particular value -- it is tuned on the very data it is then evaluated on.

Model order selection is the array-processing answer to exactly this question:
choose the split between signal and noise eigenvalues by which one best
explains the data. MDL is consistent, AIC is not and is known to overestimate
the order; both are run so the difference is visible rather than asserted.

The comparison that matters is against `espirit tuned c=0.1` -- the *best*
threshold found by the sweep. Beating an untuned default would prove nothing;
matching a tuned one without tuning is the actual claim.

Re-run of the evaluation lost when the machine lost power mid-run; the
criterion itself lives in `recon.py` (`mdl_aic_order`) and is unchanged.
"""
from __future__ import annotations

import os, sys, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import numpy as np
import pandas as pd

import fastmri_data as fd
from recon import espirit_maps, estimate_maps_acs, sense_recon
from simulate import uniform_mask, ifft2c
from metrics import summarize

ROOT = "../datasets/fastmri/multicoil_val"

ESTIMATORS = [
    ("ratio-nohann",       dict(kind="ratio", hann=False)),
    ("espirit tuned c=.1", dict(kind="espirit", k=8, c=0.1)),
    ("espirit MDL",        dict(kind="espirit", k=6, order="mdl")),
    ("espirit AIC",        dict(kind="espirit", k=6, order="aic")),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-vols", type=int, default=4)
    ap.add_argument("--n-acs", type=int, default=48)
    ap.add_argument("--R", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--out", default="results/model_order.csv")
    a = ap.parse_args()

    vols, _ = fd.list_volumes(ROOT)
    vols = [v for v in vols if v["n_coils"] >= 8][:a.n_vols]
    print(f"{len(vols)} volumes, {a.n_acs} ACS lines\n")

    rows = []
    for vi, v in enumerate(vols):
        k = fd.remove_readout_oversampling(fd.load_slice(v["path"], v["n_slices"] // 2))
        ny, nx = k.shape[1:]
        ref = np.sqrt((np.abs(ifft2c(k)) ** 2).sum(0))
        bm = fd.body_mask(ref); refm = fd.apply_mask(ref, bm)
        _, acs = uniform_mask(ny, nx, 2, a.n_acs)

        for name, cfg in ESTIMATORS:
            t0 = time.time()
            if cfg["kind"] == "ratio":
                S, info = estimate_maps_acs(k, acs, hann=cfg["hann"]), {}
            else:
                S, info = espirit_maps(k, acs, k=cfg["k"], c=cfg.get("c", 0.02),
                                       order=cfg.get("order"), eig_thresh=0.9,
                                       return_info=True)
            dt = time.time() - t0
            for R in a.R:
                ssim = summarize(refm, fd.apply_mask(np.abs(sense_recon(k, S, R)), bm))["ssim"]
                rows.append(dict(volume=os.path.basename(v["path"]), estimator=name,
                                 R=R, ssim=ssim, map_s=dt,
                                 n_kernels=info.get("n_kernels"),
                                 n_total=info.get("n_total"),
                                 support_frac=info.get("support_frac")))
        print(f"  volume {vi+1}/{len(vols)} done", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv(a.out, index=False)

    print(f"\n{len(vols)} volumes, {a.n_acs} ACS")
    piv = df.pivot_table(index="estimator", columns="R", values="ssim", aggfunc="mean")
    extra = df.groupby("estimator").agg(map_s=("map_s", "mean"),
                                        kernels=("n_kernels", "mean"),
                                        of=("n_total", "mean"))
    print(piv.join(extra).reindex([n for n, _ in ESTIMATORS]).to_string(
        float_format=lambda x: f"{x:.4f}"))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
