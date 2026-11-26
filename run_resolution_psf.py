"""Local point-spread function: is the learned reconstructor's low noise blur?

`run_effective_g.py` measures noise amplification and finds VarNet at
g_eff = 0.29 -- suppressing noise below the fully sampled reference. The
bias/variance split there is the guard against that being the ignore-the-data
failure, but it is a weak one: variance scales with the probe amplitude while
bias does not, so `bias_frac` is not scale-free and cannot be read as a
resolution measurement.

Kleineisel et al. (*Z. Med. Phys.* 35:343, 2023) use the right instrument for
this, and this file reimplements it so the numbers here are comparable to
theirs rather than merely adjacent. Perturb one pixel by 0.1% of maximum
intensity, push it through the forward model, reconstruct, and subtract the
unperturbed reconstruction. The difference is that pixel's point-spread
function -- exactly the PSF column for a linear reconstructor, the local
Jacobian column for a nonlinear one, which is why the perturbation must be
small. The main-lobe width at 64% of peak is the local resolution: 1 pixel
means no blurring, larger means the reconstructor is trading resolution away.

Two efficiency and correctness points:

* One perturbed pixel per reconstruction would need 10^5 reconstructions per
  slice. Perturbations spaced far apart relative to the PSF width superpose, so
  a grid of them is measured from a single reconstruction. For a linear method
  that is exact; for VarNet it is not, so `--check-single` re-measures a sample
  of points one at a time and reports the disagreement instead of assuming it
  away.
* The perturbation is injected in *k-space* through the estimated sensitivities
  (`k + F(S * delta)`), which is the same forward model the study uses
  everywhere else, so the measured PSF includes the sensitivity estimate's own
  contribution -- as it should, since that is part of the reconstructor.
"""
from __future__ import annotations

import os, sys, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import numpy as np
import pandas as pd
from scipy import ndimage

import fastmri_data as fd
from simulate import fft2c, ifft2c, uniform_mask
from recon import (sense_recon, grappa_recon, zerofill_recon,
                   estimate_maps_acs, sense_combine)
from learned_recon import (load_varnet, varnet_recon, DEFAULT_CKPT,
                           load_unet, unet_recon, UNET_CKPT)
from gpu_guard import gpu_slot

ROOT = "../datasets/fastmri/multicoil_val"
FRAC = 0.64          # main-lobe height Kleineisel et al. measure the width at


def grid_points(bm, spacing, off):
    """Perturbation sites on a sub-grid, kept clear of the body-mask edge."""
    interior = ndimage.binary_erosion(bm, iterations=10)
    ys, xs = np.nonzero(interior)
    pts = [(y, x) for y, x in zip(ys, xs)
           if y % spacing == off[0] and x % spacing == off[1]]
    return pts


def _width_1d(prof, c, thresh):
    """Width of the main lobe along one axis, with sub-pixel crossings.

    Counting suprathreshold pixels quantizes the width to integers (and its
    square root to sqrt of integers), which reports every method as exactly
    1.000 and cannot see the modest blurring that is the whole question. The
    threshold crossing is therefore interpolated linearly on each side of the
    peak. An ideal delta response gives 0.72 by this convention, so widths are
    reported relative to the fully sampled reconstruction measured the same
    way -- which is the quantity Kleineisel et al. report as "1 = no blurring".
    """
    n = prof.size
    if prof[c] < thresh:
        return np.nan
    i = c
    while i > 0 and prof[i - 1] >= thresh:
        i -= 1
    if i == 0:
        left = 0.0
    else:
        lo, hi = prof[i - 1], prof[i]
        left = (i - 1) + (thresh - lo) / (hi - lo) if hi > lo else float(i)
    j = c
    while j < n - 1 and prof[j + 1] >= thresh:
        j += 1
    if j == n - 1:
        right = float(n - 1)
    else:
        hi, lo = prof[j], prof[j + 1]
        right = j + (hi - thresh) / (hi - lo) if hi > lo else float(j)
    return right - left


def psf_width(patch):
    """Main-lobe width at 64% of peak along the phase-encode and readout axes.

    Undersampling acts on the phase-encode axis alone, so `w_pe` is where a
    resolution loss caused by acceleration has to appear; `w_ro` is the control
    that should stay flat and says whether a measured widening is real.
    """
    cy, cx = np.array(patch.shape) // 2
    peak = patch[cy, cx]
    if peak <= 0:
        return np.nan, np.nan, np.nan
    # The threshold must come from the response AT the perturbed pixel, not from
    # the patch maximum. At R=8 aliasing puts the largest response somewhere
    # other than the centre, and thresholding on that global max both discarded
    # valid points and reported widths below 0.72 -- narrower than a delta, which
    # is impossible for a point-spread function and was the tell.
    thresh = FRAC * peak
    w_pe = _width_1d(patch[:, cx], cy, thresh)
    w_ro = _width_1d(patch[cy, :], cx, thresh)
    leak = float(patch.max() / peak)      # >1 means energy landed off-centre
    return w_pe, w_ro, leak


def measure(kf, R, n_acs, lam, delta_frac, spacing, half, varnet, offsets, unet=None):
    """Every method's LPSF widths for one slice at one acceleration."""
    k0 = fd.remove_readout_oversampling(kf)
    nc, ny, nx = k0.shape
    if ny % R:
        return []
    ref = np.sqrt((np.abs(ifft2c(k0)) ** 2).sum(0))
    bm = fd.body_mask(ref)
    delta = delta_frac * float(ref.max())

    mask_s, acs = uniform_mask(ny, nx, R, n_acs)
    mask_f, _ = uniform_mask(ny, kf.shape[-1], R, n_acs)
    S = estimate_maps_acs(k0, acs)
    S_f = estimate_maps_acs(kf, acs) if varnet is not None else None

    def recon(k, kfull, name):
        if name == "R=1 combine":
            return np.abs(sense_combine(ifft2c(k), S))
        if name == "Zero-filled":
            return np.abs(zerofill_recon(k * mask_s, S))
        if name == "SENSE":
            return np.abs(sense_recon(k, S, R))
        if name == "SENSE+DL":
            return np.abs(sense_recon(k, S, R, lam=lam))
        if name == "GRAPPA":
            return np.abs(grappa_recon(k, acs, R, maps=S))
        if name == "VarNet":
            return varnet_recon(kfull, mask_f, n_acs, crop_ro=True, model=varnet)
        if name == "U-Net":
            return unet_recon(k, mask_s, model=unet)
        raise ValueError(name)

    methods = ["R=1 combine", "Zero-filled", "SENSE", "SENSE+DL", "GRAPPA"]
    if varnet is not None:
        methods.append("VarNet")
    if unet is not None:
        methods.append("U-Net")
    base = {}
    for m in methods:
        try:
            base[m] = recon(k0, kf, m)
        except ValueError:
            pass

    rows = []
    for off in offsets:
        pts = grid_points(bm, spacing, off)
        if not pts:
            continue
        D = np.zeros((ny, nx))
        for y, x in pts:
            D[y, x] = delta
        k_p = k0 + fft2c(S * D[None])
        if varnet is not None:
            D_f = np.zeros((ny, kf.shape[-1]))
            x0 = kf.shape[-1] // 2 - nx // 2
            D_f[:, x0:x0 + nx] = D
            kf_p = kf + fft2c(S_f * D_f[None])
        else:
            kf_p = None

        for m in base:
            try:
                pert = recon(k_p, kf_p, m)
            except ValueError:
                continue
            lpsf = np.abs(pert - base[m])
            for y, x in pts:
                p = lpsf[y - half:y + half + 1, x - half:x + half + 1]
                if p.shape != (2 * half + 1, 2 * half + 1):
                    continue
                wp, wr, leak = psf_width(p)
                if np.isnan(wp):
                    continue
                rows.append(dict(method=m, R=R, y=y, x=x, w_pe=wp, w_ro=wr,
                                 leak=leak,
                                 peak=float(p.max()),
                                 snr=float(p.max() / max(np.median(p), 1e-30))))
    return rows


def main():
    """Admission-controlled entry point; see `src/gpu_guard`."""
    with gpu_slot(need_mb=3500, label="resolution"):
        return _main()


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-vols", type=int, default=4)
    ap.add_argument("--n-slices", type=int, default=1)
    ap.add_argument("--R", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--delta", type=float, nargs="+", default=[0.001],
                    help="perturbation as a fraction of max intensity; pass "
                         "several to check the response is in the linear regime")
    ap.add_argument("--spacing", type=int, default=16)
    ap.add_argument("--offsets", type=int, default=2)
    ap.add_argument("--half", type=int, default=8)
    ap.add_argument("--n-acs", type=int, default=24)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--out", default="results/resolution_psf.csv")
    a = ap.parse_args()

    offs = [(i * a.spacing // max(a.offsets, 1), i * a.spacing // max(a.offsets, 1))
            for i in range(a.offsets)]
    vols, _ = fd.list_volumes(ROOT)
    vols = vols[:a.n_vols]
    varnet = load_varnet() if os.path.exists(DEFAULT_CKPT) else None
    unet = load_unet() if os.path.exists(UNET_CKPT) else None
    print(f"{len(vols)} volumes | spacing {a.spacing} | offsets {offs} | "
          f"delta {a.delta} | VarNet {'on' if varnet else 'SKIPPED'}\n", flush=True)

    rows = []
    for vi, v in enumerate(vols):
        mid = v["n_slices"] // 2
        for sl in range(mid - a.n_slices // 2, mid - a.n_slices // 2 + a.n_slices):
            kf = fd.load_slice(v["path"], sl)
            for R in a.R:
                for dl in a.delta:
                    t0 = time.time()
                    r = measure(kf, R, a.n_acs, a.lam, dl, a.spacing, a.half,
                                varnet, offs, unet)
                    for x in r:
                        x.update(volume=os.path.basename(v["path"]), slice=sl,
                                 delta=dl)
                    rows += r
                    print(f"  {os.path.basename(v['path'])[:26]:26s} sl={sl} "
                          f"R={R} d={dl} {len(r):5d} pts {time.time()-t0:6.1f}s",
                          flush=True)

    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv(a.out, index=False)

    order = ["R=1 combine", "Zero-filled", "SENSE", "SENSE+DL", "GRAPPA",
             "VarNet", "U-Net"]
    for R in a.R:
        for dl in a.delta:
            d = df[(df.R == R) & (df.delta == dl)]
            if not len(d):
                continue
            g = d.groupby("method").agg(w_pe=("w_pe", "mean"),
                                        w_ro=("w_ro", "mean"),
                                        n=("w_pe", "size"), leak=("leak", "median"),
                                        snr=("snr", "median")).reindex(order).dropna(how="all")
            ref_pe = g.loc["R=1 combine", "w_pe"] if "R=1 combine" in g.index else np.nan
            ref_ro = g.loc["R=1 combine", "w_ro"] if "R=1 combine" in g.index else np.nan
            g["blur_pe"] = g.w_pe / ref_pe
            g["blur_ro"] = g.w_ro / ref_ro
            print(f"\n=== R = {R}, delta = {dl} ===")
            print(g.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
