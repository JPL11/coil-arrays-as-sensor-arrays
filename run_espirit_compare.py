"""Does a subspace steering-vector estimator restore SENSE on real coils?

Finding 4 of the simulated study (SENSE overtakes GRAPPA above ~24 ACS lines)
reversed on real data: GRAPPA won at every budget. The proposed explanation was
that real sensitivities are ~14x rougher than a birdcage model, so the
low-resolution *ratio* estimator cannot resolve them. ESPIRiT tests that
explanation directly -- it estimates the same steering vectors as the dominant
eigenvector of a local Gram matrix built from the ACS signal subspace, never
forming a ratio.

Note on the metric: an "R=1 map ceiling" (how well the maps reproduce the
fully-sampled RSS combine) is *not* a fair estimator comparison. The ratio
estimator reproduces RSS exactly by construction once the ACS covers all of
k-space, so that measure is self-referential. What matters is unfolding quality
at R > 1, which is what this script reports.
"""
import os, sys, time, itertools
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
import numpy as np, pandas as pd

import fastmri_data as fd
from simulate import uniform_mask, ifft2c, effective_acceleration
from recon import (sense_recon, grappa_recon, zerofill_recon,
                   estimate_maps_acs, espirit_maps, sense_combine)
from metrics import summarize

ROOT     = "../datasets/fastmri/multicoil_val"
R_VALUES = [2, 4, 5, 8]
ACS      = [24, 48]
LAM      = 3e-2                     # real-data optimum from the loading sweep
ESTIMATORS = {
    "ratio+hann":   lambda k, acs: estimate_maps_acs(k, acs, hann=True),
    "ratio-nohann": lambda k, acs: estimate_maps_acs(k, acs, hann=False),
    "espirit":      lambda k, acs: espirit_maps(k, acs, k=6, c=0.02, eig_thresh=0.5),
}


def main(out_csv="results/espirit_compare.csv", n_slices=1):
    vols, _ = fd.list_volumes(ROOT)
    print(f"{len(vols)} volumes; readout oversampling removed (exact, halves pixels)")
    rows = []
    t_all = time.time()
    for vi, v in enumerate(vols):
        mid = v["n_slices"] // 2
        for sl in range(mid, mid + n_slices):
            k = fd.remove_readout_oversampling(fd.load_slice(v["path"], sl))
            nc, ny, nx = k.shape
            ref = np.sqrt((np.abs(ifft2c(k)) ** 2).sum(0))
            bm = fd.body_mask(ref); refm = fd.apply_mask(ref, bm)
            for n_acs in ACS:
                _, acs = uniform_mask(ny, nx, 2, n_acs)
                for est_name, est in ESTIMATORS.items():
                    t0 = time.time()
                    try:
                        S = est(k, acs)
                    except ValueError:
                        continue
                    t_map = time.time() - t0
                    # unfolding reference: R=1 matched filter with these maps
                    unf = fd.apply_mask(np.abs(sense_combine(ifft2c(k), S)), bm)
                    for R in R_VALUES:
                        if ny % R:
                            continue
                        mask, _ = uniform_mask(ny, nx, R, n_acs)
                        out = {"SENSE": sense_recon(k, S, R),
                               "SENSE+DL": sense_recon(k, S, R, lam=LAM),
                               "Zero-filled": zerofill_recon(k * mask, S)}
                        try:
                            out["GRAPPA"] = grappa_recon(k, acs, R, maps=S)
                        except ValueError:
                            pass
                        for meth, img in out.items():
                            rec = fd.apply_mask(np.abs(img), bm)
                            m = summarize(refm, rec)
                            m["ssim_unfold"] = summarize(unf, rec)["ssim"]
                            m.update(method=meth, estimator=est_name, R=R, n_acs=n_acs,
                                     n_coils=nc, volume=os.path.basename(v["path"]),
                                     slice_idx=sl, acquisition=v["acquisition"],
                                     map_time_s=t_map)
                            rows.append(m)
        print(f"  [{vi+1}/{len(vols)}] {os.path.basename(v['path'])[11:]:28s} "
              f"nc={v['n_coils']:2d}  {time.time()-t_all:6.1f}s  ({len(rows)} rows)")
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    df = pd.DataFrame(rows); df.to_csv(out_csv, index=False)
    print(f"\n{len(df)} rows -> {out_csv}")
    return df


if __name__ == "__main__":
    main()
