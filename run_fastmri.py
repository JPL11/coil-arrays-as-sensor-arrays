"""The array study, re-run on real fastMRI multicoil brain k-space.

Same estimators as the simulation (`recon.py` is imported unchanged); only the
data and the evaluation protocol differ. Four things had to change, and each one
is a finding rather than a detail:

1. **Volume filtering.** 4 of 24 downloaded volumes carry 96 all-zero
   phase-encode lines. Retrospectively undersampling an already-gapped k-space
   measures the acquisition, not the reconstructor, so they are excluded.

2. **Body masking.** Metrics are computed inside the head. Against an unmasked
   RSS reference, a *fully sampled* matched-filter combine scores SSIM 0.48 with
   80% of its error energy in the background -- an RSS/matched-filter noise-floor
   mismatch that has nothing to do with parallel imaging.

3. **Two references.** `ssim`/`nmse` are against the fully-sampled RSS image
   (what a radiologist would see: the absolute standard). `ssim_unfold` is
   against the R=1 matched-filter image built from the *same* estimated maps,
   which isolates aliasing removal from sensitivity-map resolution. In the
   simulation the maps were nearly exact and the two coincided; on real data
   they do not, and reporting only one of them would mis-attribute the error.

4. **Noise pre-whitening.** Real coil noise is correlated (off-diagonal
   correlation up to 0.54 here), so the unweighted least-squares SENSE solve is
   not the maximum-likelihood estimator. Pre-whitening is swept as its own
   condition rather than assumed.
"""
from __future__ import annotations

import os, sys, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import numpy as np
import pandas as pd

import fastmri_data as fd
from simulate import ifft2c, uniform_mask, effective_acceleration
from recon import (sense_recon, grappa_recon, zerofill_recon,
                   estimate_maps_acs, sense_combine)
from learned_recon import (load_varnet, varnet_recon, DEFAULT_CKPT,
                           load_unet, unet_recon, UNET_CKPT)
from gpu_guard import gpu_slot
from metrics import summarize

ROOT     = "../datasets/fastmri/multicoil_val"
N_SLICES = 3          # central slices per volume
# 320 phase-encode lines: R must divide it exactly for the SENSE aliasing model
DEFAULTS = dict(R=4, n_coils=None, drop=0, map_err=0.0, n_acs=24, lam=1e-2,
                hann=True, whiten=False)


def perturb_maps(S, err, rng):
    """Smooth multiplicative sensitivity error -- the MRI form of steering-vector
    mismatch. Same construction as the simulation, so the axis is comparable."""
    if err <= 0:
        return S
    ny, nx = S.shape[-2:]
    gy = np.linspace(-1, 1, ny); gx = np.linspace(-1, 1, nx)
    X, Y = np.meshgrid(gx, -gy)
    out = np.empty_like(S)
    for c in range(S.shape[0]):
        a = rng.normal(size=6)
        sm = a[0]*X + a[1]*Y + a[2]*X*Y + a[3]*X**2 + a[4]*Y**2 + a[5]
        sm /= max(np.abs(sm).max(), 1e-12)
        out[c] = S[c] * np.exp(1j * err * np.pi * sm) * (1 + err * sm)
    r = np.sqrt((np.abs(out) ** 2).sum(0))
    return out / np.maximum(r, 1e-8)


def one_slice(k_full, seed, R, n_coils, drop, map_err, n_acs, lam, hann, whiten,
              varnet=None, unet=None):
    """All methods on one real slice under one condition.

    `k_full` still carries fastMRI's 2x readout oversampling. The classical
    estimators see it removed (exact, and it halves the pixel count); the
    learned reconstructor sees it intact, because that is the k-space support
    it was trained on and `validate_varnet.py` measures the difference as
    0.010 SSIM in its favour. Coil selection and whitening are applied to both
    forms from the same draws and the same Psi, so the two never diverge into
    different arrays.
    """
    rng = np.random.default_rng(seed)
    kf = k_full
    nc_avail = kf.shape[0]

    if n_coils is not None and n_coils < nc_avail:      # aperture (array size)
        kf = kf[np.sort(rng.choice(nc_avail, n_coils, replace=False))]
    if drop:                                            # element loss
        keep = np.sort(rng.choice(kf.shape[0], max(kf.shape[0] - drop, 1), replace=False))
        kf = kf[keep]
    k = fd.remove_readout_oversampling(kf)
    # The reference is built from the retained coils *before* whitening.
    # Whitening rescales each coil, so an RSS computed after it is a different
    # image -- evaluating the whitened and unwhitened conditions against their
    # own references compared two different targets and made that contrast
    # meaningless (its bootstrap CI was correspondingly uninformative).
    ref = np.sqrt((np.abs(ifft2c(k)) ** 2).sum(0))      # fully-sampled RSS
    if whiten:
        Psi, _ = fd.noise_covariance(k)                 # the same Psi for both
        k  = fd.prewhiten(k,  Psi)                      # forms, or they are no
        kf = fd.prewhiten(kf, Psi)                      # longer the same slice

    nc, ny, nx = k.shape
    if ny % R:
        return []                                       # SENSE needs R | ny
    mask, acs = uniform_mask(ny, nx, R, n_acs)

    bm   = fd.body_mask(ref)
    refm = fd.apply_mask(ref, bm)

    def scaled(img):
        """Least-squares optimal global gain, then mask.

        Whitening and coil selection change the overall gain of the
        reconstruction without changing its structure, so a fixed reference
        would otherwise score them on units rather than image quality. One
        scalar per image is fitted; nothing spatially varying is corrected.
        """
        x = fd.apply_mask(np.abs(img), bm)
        denom = float(np.sum(x * x))
        a = float(np.sum(refm * x)) / denom if denom > 0 else 1.0
        return a * x

    S = perturb_maps(estimate_maps_acs(k, acs, hann=hann), map_err, rng)
    # R=1 matched filter with the SAME maps: isolates unfolding from map quality
    unfold_ref = scaled(sense_combine(ifft2c(k), S))

    out, times = {}, {}
    def timed(name, fn):
        t0 = time.perf_counter()
        try:
            out[name] = fn()
        except ValueError:
            return                                      # e.g. ACS too small for GRAPPA
        times[name] = (time.perf_counter() - t0) * 1e3

    timed("Zero-filled", lambda: zerofill_recon(k * mask, S))
    timed("SENSE",       lambda: sense_recon(k, S, R))
    timed("SENSE+DL",    lambda: sense_recon(k, S, R, lam=lam))
    timed("GRAPPA",      lambda: grappa_recon(k, acs, R, maps=S))
    if varnet is not None:
        # The network estimates its own sensitivities from the ACS block, so it
        # takes no `maps` argument and the map_err condition cannot reach it --
        # which is the comparison, not an omission.
        mask_f, _ = uniform_mask(ny, kf.shape[-1], R, n_acs)
        timed("VarNet", lambda: varnet_recon(kf, mask_f, n_acs, crop_ro=True,
                                             model=varnet))
    if unet is not None:
        # Image-domain only: it sees the zero-filled RSS and nothing else -- no
        # k-space, no mask, no sensitivities, no data consistency. Whatever it
        # shares with VarNet is a property of learned reconstruction; whatever
        # it does not is a property of the physics the unrolled network keeps.
        timed("U-Net", lambda: unet_recon(k, mask, model=unet))

    _, gmap = sense_recon(k, S, R, return_gfactor=True)
    gbody = gmap[bm]

    rows = []
    for name, img in out.items():
        rec = scaled(img)
        m = summarize(refm, rec)
        m["ssim_unfold"] = summarize(unfold_ref, rec)["ssim"]
        m.update(method=name, R=R, R_eff=effective_acceleration(mask), n_coils=nc,
                 dropped=drop, map_err=map_err, n_acs=n_acs, lam=lam, hann=hann,
                 whiten=whiten, runtime_ms=times[name],
                 g_mean=float(gbody.mean()), g_max=float(gbody.max()))
        rows.append(m)
    return rows


SWEEPS = {
    "acceleration": ("R",       [2, 4, 5, 8, 10]),   # 320 PE lines: R must divide 320
    "aperture":     ("n_coils", [4, 8, 12, 16, 20]),
    "dropout":      ("drop",    [0, 1, 2, 4, 8]),
    "map_error":    ("map_err", [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]),
    "acs":          ("n_acs",   [12, 24, 32, 48, 64, 96, 128]),
    "loading":      ("lam",     [0.0, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1]),
    "taper":        ("hann",    [True, False]),
    "prewhiten":    ("whiten",  [False, True]),
}


def main(out_csv="results/fastmri_experiments.csv", max_vols=None, use_varnet=True,
         use_unet=True, root=ROOT):
    vols, rejected = fd.list_volumes(root)
    if max_vols:
        vols = vols[:max_vols]
    print(f"{len(vols)} fully-sampled volumes ({len(rejected)} rejected)")
    print(f"coil counts present: {sorted({v['n_coils'] for v in vols})}\n")

    varnet = None
    if use_varnet:
        if os.path.exists(DEFAULT_CKPT):
            varnet = load_varnet()
            print(f"VarNet: pretrained fastMRI brain leaderboard weights on "
                  f"{next(varnet.parameters()).device}")
        else:
            print(f"VarNet: SKIPPED, no weights at {DEFAULT_CKPT}")
    unet = None
    if use_unet:
        if os.path.exists(UNET_CKPT):
            unet = load_unet()
            print("U-Net: pretrained fastMRI brain baseline (image domain, "
                  "no data consistency)")
        else:
            print(f"U-Net: SKIPPED, no weights at {UNET_CKPT}")

    # cache the slices once -- loading dominates otherwise. Cached with the
    # readout oversampling still on: removing it is a cheap 1-D FFT that
    # `one_slice` does per condition, and caching both forms doubles the
    # resident k-space for no gain.
    slices = []
    for v in vols:
        mid = v["n_slices"] // 2
        for sl in range(mid - N_SLICES // 2, mid - N_SLICES // 2 + N_SLICES):
            slices.append((v, sl, fd.load_slice(v["path"], sl)))
    print(f"cached {len(slices)} slices\n")

    rows = []
    for sweep, (key, values) in SWEEPS.items():
        t0 = time.time()
        for val in values:
            cfg = dict(DEFAULTS); cfg[key] = val
            for i, (v, sl, k) in enumerate(slices):
                if cfg["n_coils"] is not None and k.shape[0] < cfg["n_coils"]:
                    continue                            # volume lacks that many coils
                for r in one_slice(k, seed=1000 + i, varnet=varnet, unet=unet, **cfg):
                    r.update(sweep=sweep, level=val, volume=os.path.basename(v["path"]),
                             slice_idx=sl, acquisition=v["acquisition"])
                    rows.append(r)
        print(f"  {sweep:13} {len(values)} levels   {time.time()-t0:6.1f}s   ({len(rows)} rows)")

    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\n{len(df)} rows -> {out_csv}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-vols", type=int, default=None)
    ap.add_argument("--out", default="results/fastmri_experiments.csv")
    ap.add_argument("--no-varnet", action="store_true",
                    help="skip the unrolled learned reconstructor")
    ap.add_argument("--no-unet", action="store_true",
                    help="skip the image-domain learned reconstructor")
    ap.add_argument("--root", default=ROOT,
                    help="volume directory; point at multicoil_balanced for the "
                         "contrast-balanced set")
    a = ap.parse_args()
    need = 3000 + (2500 if not a.no_unet else 0)
    with gpu_slot(need_mb=need, label="fastmri"):
        main(a.out, a.max_vols, use_varnet=not a.no_varnet,
             use_unet=not a.no_unet, root=a.root)
