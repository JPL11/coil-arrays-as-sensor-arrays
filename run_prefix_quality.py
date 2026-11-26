"""Can reconstruction quality be predicted from a partial acquisition?

Everything so far predicts the quality of a *completed* scan. The prospective
question is different and is the prerequisite for any streaming or on-device
abort decision: given the first X% of phase-encode lines, how well will the
**final** reconstruction turn out?

This is step 1 of the streaming-QC idea recorded in
`ARRAY_SIGNAL_PROCESSING_HANDOFF.md` section 5b, and it is deliberately run
*before* any reservoir. A reservoir only earns its place if the feature
trajectory carries information the current snapshot does not; if the static
model already predicts the final SSIM at 25% of lines, there is nothing for
recurrence to add and the idea reduces to early stopping.

**Acquisition order is an assumption, not a measurement.** `uniform_mask` has no
notion of time, so a prefix has to be defined. Two orders are run because they
bracket real practice and because the difference between them is itself the
answer to "how much does this depend on a scan parameter you control":

* **centre-out** -- low frequencies first. The ACS block arrives early, so
  sensitivity estimation is possible from a small prefix, and the prefix
  reconstruction already resembles a low-resolution version of the final image.
* **sequential** -- one end to the other. The ACS block is not complete until
  the acquisition passes the centre, so early prefixes cannot calibrate at all.

The target is the SSIM of the reconstruction from the **full** acquisition, not
from the prefix. Predicting the prefix's own quality would be a different and
much easier problem, and not the one a scanner needs answered.
"""
from __future__ import annotations

import os, sys, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import numpy as np
import pandas as pd

import fastmri_data as fd
from simulate import uniform_mask, ifft2c
from recon import sense_recon, grappa_recon, zerofill_recon, estimate_maps_acs
from self_consistency import grappa_self_consistency
from metrics import summarize
from run_noref_quality import acquisition_features

FRACS = [0.25, 0.50, 0.75, 1.00]


def acquisition_order(ny, R, n_acs, order):
    """The phase-encode lines a full R-fold scan acquires, in acquisition order."""
    mask, acs = uniform_mask(ny, 1, R, n_acs)
    lines = np.flatnonzero(mask[:, 0])
    c = ny // 2
    if order == "centre-out":
        return lines[np.argsort(np.abs(lines - c))], acs
    return np.sort(lines), acs


def prefix_mask(ny, nx, lines, acs):
    """Mask containing only the lines acquired so far."""
    m = np.zeros((ny, nx), dtype=bool)
    m[lines, :] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=[
        "../datasets/fastmri/multicoil_val",
        "../datasets/fastmri/multicoil_val_b1/multicoil_val",
        "../datasets/fastmri/multicoil_val_b2/multicoil_val"])
    ap.add_argument("--crop-pe", type=int, default=None)
    ap.add_argument("--n-slices", type=int, default=2)
    ap.add_argument("--R", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--n-acs", type=int, default=24)
    ap.add_argument("--orders", nargs="+", default=["centre-out", "sequential"])
    ap.add_argument("--max-vols", type=int, default=None)
    ap.add_argument("--out", default="results/prefix_quality.csv")
    a = ap.parse_args()

    vols = []
    for r in a.roots:
        if os.path.isdir(r):
            v, _ = fd.list_volumes(r, crop_pe=a.crop_pe)
            vols += v
    if a.max_vols:
        vols = vols[:a.max_vols]
    print(f"{len(vols)} volumes | orders {a.orders} | fractions {FRACS}\n", flush=True)

    rows, t0 = [], time.time()
    for vi, v in enumerate(vols):
        mid = v["n_slices"] // 2
        for sl in range(mid - a.n_slices // 2, mid - a.n_slices // 2 + a.n_slices):
            try:
                k = fd.remove_readout_oversampling(fd.load_slice(v["path"], sl))
            except Exception:
                continue
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
                full_mask, acs = uniform_mask(ny, nx, R, a.n_acs)
                S_full = estimate_maps_acs(k, acs)
                # the TARGET: quality of the completed acquisition
                final = {}
                for nm, fn in [("SENSE", lambda: sense_recon(k, S_full, R)),
                               ("GRAPPA", lambda: grappa_recon(k, acs, R, maps=S_full)),
                               ("Zero-filled", lambda: zerofill_recon(k * full_mask, S_full))]:
                    try:
                        final[nm] = scored(fn())["ssim"]
                    except ValueError:
                        pass
                for order in a.orders:
                    lines, acs_o = acquisition_order(ny, R, a.n_acs, order)
                    for frac in FRACS:
                        take = lines[:max(int(round(frac * len(lines))), 2)]
                        pm = prefix_mask(ny, nx, take, acs_o)
                        kp = k * pm
                        # can the prefix calibrate at all? ACS lines present so far
                        acs_have = int(pm[acs_o, 0].sum())
                        feats = dict(acs_acquired=acs_have,
                                     frac_lines=float(len(take) / len(lines)))
                        if acs_have >= 8:
                            try:
                                Sp = estimate_maps_acs(kp, acs_o)
                                _, gp = sense_recon(kp, Sp, R, return_gfactor=True)
                                feats.update(acquisition_features(
                                    kp, Sp, gp, bm, acs_o, pm, Psi))
                                feats.update(grappa_self_consistency(kp, acs_o, R))
                            except Exception:
                                pass
                        for nm, y in final.items():
                            rows.append(dict(volume=os.path.basename(v["path"]),
                                             slice=sl, R=R, order=order, frac=frac,
                                             method=nm, final_ssim=y, **feats))
        if (vi + 1) % 20 == 0:
            print(f"  {vi+1}/{len(vols)}  {time.time()-t0:6.0f}s  {len(rows)} rows",
                  flush=True)

    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"\n{len(df)} rows, {df.volume.nunique()} volumes -> {a.out}")


if __name__ == "__main__":
    main()
