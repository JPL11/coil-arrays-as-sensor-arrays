"""No-reference reconstruction quality prediction from acquisition-side features.

Can you predict how badly a reconstruction went *without* the ground truth, from
quantities available at scan time? This is array processing with no downstream
task attached: the features are the g-factor statistics, the array size, the
sampling and the calibration budget, all computable from the estimated
sensitivities and the mask.

The pilot on 20 volumes said the answer is "moderately, and only where noise
dominates" -- |corr(g_mean, SSIM)| of 0.55 for SENSE at R=4 falling to 0.06 at
R=8 -- but with 60 slices from 20 volumes nothing could be fitted: a
gradient-boosted model on all features generalised *worse* than g_mean alone,
often worse than predicting the mean. This runs the same question at scale.

Two design points that the pilot got wrong and this does not:

* **Grouping.** Every split is by volume. Slices from one brain share anatomy,
  coil loading and noise covariance; splitting on slices would report a model
  that has seen the test brain.
* **A physics baseline that has to be beaten.** `g_mean` alone is the
  incumbent. A learned model that does not beat a single analytic quantity is
  not a result, and on the pilot it did not.
"""
from __future__ import annotations

import os, sys, time, argparse, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import numpy as np
import pandas as pd

import fastmri_data as fd
from simulate import uniform_mask, ifft2c
from recon import sense_recon, grappa_recon, zerofill_recon, estimate_maps_acs
from metrics import summarize

FEATURES = ["g_mean", "g_med", "g_p90", "g_max", "g_frac_gt5", "n_coils",
            "R_eff", "n_acs", "psi_offdiag", "sens_rough", "acs_energy"]

# Image-domain no-reference features, computed from the reconstruction itself.
# This is what the existing no-reference MRI quality literature uses (BRISQUE,
# PIQE and CNN variants all read the output image), so it is the baseline the
# acquisition-side features have to be compared against rather than merely
# distinguished from. If looking at the picture predicts its error as well as
# the acquisition physics does, the acquisition side is redundant.
IMG_FEATURES = ["img_grad", "img_bg_noise", "img_contrast", "img_hf_ratio",
                "img_entropy", "img_local_var"]


def image_features(img, bm):
    """No-reference features from the reconstructed magnitude image alone."""
    from simulate import fft2c
    # np.asarray(x, dtype=float) on a complex array discards the imaginary part
    # *before* abs, so every image feature would be computed on Re(img) rather
    # than |img|. Same defect as in `hybrid_recon.match_scale`; the reconstructors
    # here all return complex images, so it silently affected every row.
    x = np.abs(np.asarray(img)).astype(float)
    x = x / max(x[bm].mean(), 1e-30)                 # gain-invariant
    gy, gx = np.gradient(x)
    grad = np.sqrt(gy ** 2 + gx ** 2)
    bg = ~bm
    k = np.abs(fft2c(x))
    ny, nx = x.shape
    Y, X = np.mgrid[0:ny, 0:nx]
    rad = np.sqrt((Y - ny / 2) ** 2 + (X - nx / 2) ** 2)
    hf = float(k[rad > 0.35 * ny / 2].sum() / max(k.sum(), 1e-30))
    h, _ = np.histogram(x[bm], bins=64, density=True)
    h = h[h > 0]
    from scipy import ndimage
    lv = ndimage.uniform_filter(x ** 2, 5) - ndimage.uniform_filter(x, 5) ** 2
    return dict(
        img_grad=float(grad[bm].mean()),
        img_bg_noise=float(x[bg].std()) if bg.any() else 0.0,
        img_contrast=float(x[bm].std()),
        img_hf_ratio=hf,
        img_entropy=float(-(h * np.log(h)).sum()),
        img_local_var=float(np.median(lv[bm])),
    )


def acquisition_features(k, S, gmap, bm, acs, mask, Psi):
    """Everything computable before the reconstruction is compared to anything.

    No term here touches the fully sampled reference: g comes from the estimated
    sensitivities and the sampling pattern, `sens_rough` from the maps
    themselves, `psi_offdiag` from the noise covariance, `acs_energy` from the
    calibration block. All are available on the scanner.
    """
    g = gmap[bm]
    corr = Psi / np.maximum(np.sqrt(np.outer(np.real(np.diag(Psi)),
                                             np.real(np.diag(Psi)))), 1e-30)
    off = np.abs(corr - np.eye(corr.shape[0]))
    gy, gx = np.gradient(np.abs(S[0]))
    rough = float(np.mean(np.sqrt(gy ** 2 + gx ** 2)[bm]) /
                  max(np.mean(np.abs(S[0])[bm]), 1e-12))
    return dict(
        g_mean=float(g.mean()), g_med=float(np.median(g)),
        g_p90=float(np.percentile(g, 90)), g_max=float(g.max()),
        g_frac_gt5=float((g > 5).mean()),
        n_coils=int(k.shape[0]),
        R_eff=float(mask.shape[0] / np.count_nonzero(mask[:, 0])),
        n_acs=int(acs.stop - acs.start),
        psi_offdiag=float(off.max()),
        sens_rough=rough,
        acs_energy=float(np.log10(np.abs(k[:, acs, :]).mean() + 1e-30)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=[
        "../datasets/fastmri/multicoil_val",
        "../datasets/fastmri/multicoil_val_b1/multicoil_val",
        "../datasets/fastmri/multicoil_val_b2/multicoil_val"])
    ap.add_argument("--n-slices", type=int, default=3)
    ap.add_argument("--R", type=int, nargs="+", default=[4, 5, 8])
    ap.add_argument("--n-acs", type=int, default=24)
    ap.add_argument("--max-vols", type=int, default=None)
    ap.add_argument("--crop-pe", type=int, default=None,
                    help="test fully-sampledness on the centre N phase-encode "
                         "lines rather than the raw array. Needed for knee, "
                         "whose ~36 all-zero lines sit at the edges of a 368/372 "
                         "line acquisition and are removed by the loader's own "
                         "centre crop before anything is reconstructed.")
    ap.add_argument("--out", default="results/noref_quality.csv")
    a = ap.parse_args()

    vols = []
    for r in a.roots:
        if os.path.isdir(r):
            v, _ = fd.list_volumes(r, crop_pe=a.crop_pe)
            vols += v
    if a.max_vols:
        vols = vols[:a.max_vols]
    print(f"{len(vols)} fully-sampled volumes from {len(a.roots)} sources\n", flush=True)

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
            for R in a.R:
                if ny % R:
                    continue
                mask, acs = uniform_mask(ny, nx, R, a.n_acs)
                S = estimate_maps_acs(k, acs)
                _, gmap = sense_recon(k, S, R, return_gfactor=True)
                feats = acquisition_features(k, S, gmap, bm, acs, mask, Psi)

                def scored(img):
                    x = fd.apply_mask(np.abs(img), bm)
                    d = float(np.sum(x * x))
                    return summarize(refm, (float(np.sum(refm * x)) / d if d > 0 else 1.0) * x)

                for name, fn in [("SENSE", lambda: sense_recon(k, S, R)),
                                 ("GRAPPA", lambda: grappa_recon(k, acs, R, maps=S)),
                                 ("Zero-filled", lambda: zerofill_recon(k * mask, S))]:
                    try:
                        rec = fn()
                    except ValueError:
                        continue
                    m = scored(rec)
                    rows.append(dict(volume=os.path.basename(v["path"]), slice=sl,
                                     method=name, R=R, **feats,
                                     **image_features(rec, bm), **m))
        if (vi + 1) % 10 == 0:
            print(f"  {vi+1}/{len(vols)} volumes  {time.time()-t0:6.0f}s  "
                  f"{len(rows)} rows", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"\n{len(df)} rows, {df.volume.nunique()} volumes -> {a.out}")


if __name__ == "__main__":
    main()
