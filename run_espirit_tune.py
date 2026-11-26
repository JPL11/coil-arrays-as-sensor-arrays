"""Is ESPIRiT actually worse, or just mis-parameterized?

The first comparison used hand-picked settings (k=6, c=0.02, eigenvalue crop
0.5) and SENSE got *worse*. Iyer et al. 2018 (arXiv:1811.05672) report that
ESPIRiT has three coupled calibration parameters -- kernel size, subspace size,
and eigenvalue crop -- that "poor selection" degrades badly, and they place the
realistic crop range at 0.7-0.95. Our 0.5 was outside it. They introduce
SURE-based automatic selection because manual choice is unreliable; this script
is the brute-force stand-in for that, to establish whether *any* setting makes a
subspace estimator competitive before concluding anything about the method.

The Gram matrix depends only on (k, c), so every eigenvalue crop is evaluated
from one Gram -- the sweep costs one calibration per (k, c), not per triple.
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

import fastmri_data as fd
from simulate import uniform_mask, ifft2c
from recon import estimate_maps_acs, sense_recon, sense_combine
from metrics import summarize

KS      = [4, 6, 8]
CS      = [0.005, 0.02, 0.05, 0.1]
CROPS   = [0.5, 0.7, 0.85, 0.95]
R_LIST  = [4, 8]
N_ACS   = 48
N_VOLS  = 6


def gram(ksp, acs, k, c):
    nc, ny, nx = ksp.shape
    W = sliding_window_view(ksp[:, acs, :], (k, k), axis=(1, 2))
    A = np.ascontiguousarray(np.moveaxis(W, 0, 2)).reshape(-1, nc * k * k)
    w, V = np.linalg.eigh(A.conj().T @ A)
    sv = np.sqrt(np.maximum(w, 0))
    V = V[:, sv > c * sv.max()]; nk = V.shape[1]
    ker = V.T.reshape(nk, nc, k, k).conj()
    G = np.zeros((ny, nx, nc, nc), complex); buf = np.zeros((nc, ny, nx), complex)
    y0, x0 = ny // 2 - k // 2, nx // 2 - k // 2
    for j in range(nk):
        buf[:] = 0; buf[:, y0:y0 + k, x0:x0 + k] = ker[j]
        m = ifft2c(buf) * np.sqrt(ny * nx) / k
        G += np.einsum('cyx,dyx->yxcd', m, m.conj(), optimize=True)
    return G, nk


def main(out="results/espirit_tune.csv"):
    vols, _ = fd.list_volumes("../datasets/fastmri/multicoil_val")
    vols = [v for v in vols if v["n_coils"] >= 8][:N_VOLS]
    print(f"{len(vols)} volumes, coils {[v['n_coils'] for v in vols]}")
    rows = []
    for vi, v in enumerate(vols):
        k0 = fd.remove_readout_oversampling(fd.load_slice(v["path"], v["n_slices"] // 2))
        nc, ny, nx = k0.shape
        ref = np.sqrt((np.abs(ifft2c(k0)) ** 2).sum(0))
        bm = fd.body_mask(ref); refm = fd.apply_mask(ref, bm)
        _, acs = uniform_mask(ny, nx, 2, N_ACS)
        S_ratio = estimate_maps_acs(k0, acs, hann=False)
        def ev(S, R):
            return summarize(refm, fd.apply_mask(np.abs(sense_recon(k0, S, R)), bm))["ssim"]
        for R in R_LIST:
            rows.append(dict(volume=v["path"], est="ratio-nohann", k=np.nan, c=np.nan,
                             crop=np.nan, R=R, ssim=ev(S_ratio, R), n_kernels=np.nan))
        for kk in KS:
            for c in CS:
                t0 = time.time(); G, nk = gram(k0, acs, kk, c)
                evals, evecs = np.linalg.eigh(G); lam = evals[..., -1]
                base = np.moveaxis(evecs[..., -1], -1, 0)
                ph = base[0] / np.maximum(np.abs(base[0]), 1e-12)
                base = base * np.conj(ph)[None]
                # alignment against the ratio maps -- a sanity check independent
                # of reconstruction quality
                m = bm & (lam > 0.7)
                a = base[:, m]; b = S_ratio[:, m]
                a = a / np.maximum(np.linalg.norm(a, axis=0), 1e-12)
                b = b / np.maximum(np.linalg.norm(b, axis=0), 1e-12)
                al = float(np.abs(np.sum(np.conj(a) * b, axis=0)).mean()) if m.any() else np.nan
                for crop in CROPS:
                    S = base.copy(); S[:, lam < crop] = 0
                    for R in R_LIST:
                        rows.append(dict(volume=v["path"], est="espirit", k=kk, c=c,
                                         crop=crop, R=R, ssim=ev(S, R), n_kernels=nk,
                                         align_vs_ratio=al, support=float((lam >= crop).mean()),
                                         calib_s=time.time() - t0))
        print(f"  [{vi+1}/{len(vols)}] nc={nc} done ({len(rows)} rows)")
        pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
