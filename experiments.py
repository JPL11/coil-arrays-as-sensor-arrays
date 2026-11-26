"""Full experimental program for the parallel-imaging array study.

Every condition is repeated over N_SEEDS independent noise / dropout / map-error
realizations and reported as mean +/- std. Single-seed numbers are not reported.

Sweeps
------
  acceleration : R                      (array load)
  aperture     : number of coils        (array size)
  dropout      : dead coils             (element loss)
  map_error    : sensitivity error      (steering-vector mismatch)
  snr          : k-space noise level    (measurement noise)
  acs          : ACS lines              (GRAPPA calibration budget)
  loading      : SENSE diagonal loading (robust-beamforming remedy)
"""
from __future__ import annotations

import os, sys, time, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd

from simulate import (shepp_logan, birdcage_maps, multicoil_kspace,
                      uniform_mask, effective_acceleration)
from recon import (sense_recon, grappa_recon, zerofill_recon, estimate_maps_acs)
from metrics import summarize

N        = 192
LESION   = (65, 58)
N_SEEDS  = 10
DEFAULTS = dict(R=4, nc=8, drop=0, map_err=0.0, noise=2e-3, n_acs=24, lam=1e-2)

_PHANTOM = shepp_logan(N)
_MAPS    = {}


def maps_for(nc):
    if nc not in _MAPS:
        _MAPS[nc] = birdcage_maps(nc, N)
    return _MAPS[nc]


def perturb_maps(S, err, rng):
    """Smooth multiplicative magnitude+phase error on the sensitivity maps.

    Models calibration error / patient motion between the calibration scan and
    the accelerated acquisition -- the MRI form of steering-vector mismatch.
    """
    if err <= 0:
        return S
    g = np.linspace(-1, 1, N); X, Y = np.meshgrid(g, -g)
    out = np.empty_like(S)
    for c in range(S.shape[0]):
        a = rng.normal(size=6)
        sm = a[0]*X + a[1]*Y + a[2]*X*Y + a[3]*X**2 + a[4]*Y**2 + a[5]
        sm /= max(np.abs(sm).max(), 1e-12)
        out[c] = S[c] * np.exp(1j * err * np.pi * sm) * (1 + err * sm)
    rss = np.sqrt((np.abs(out) ** 2).sum(0))
    return out / np.maximum(rss, 1e-8)


def one_run(seed, R, nc, drop, map_err, noise, n_acs, lam):
    """One realization. Returns a list of per-method metric dicts."""
    rng = np.random.default_rng(seed)
    x = _PHANTOM
    S = maps_for(nc)
    ksp = multicoil_kspace(x, S, noise_std=noise, seed=seed)
    mask, acs = uniform_mask(N, N, R, n_acs)

    keep = np.arange(nc)
    if drop:
        keep = np.sort(rng.choice(nc, nc - drop, replace=False))
    ksp_k = ksp[keep]

    Shat = perturb_maps(estimate_maps_acs(ksp_k, acs), map_err, rng)

    out, times = {}, {}
    t0 = time.perf_counter(); out["Zero-filled"] = zerofill_recon(ksp_k * mask, Shat); times["Zero-filled"] = time.perf_counter()-t0
    t0 = time.perf_counter(); out["SENSE"]       = sense_recon(ksp_k, Shat, R);        times["SENSE"] = time.perf_counter()-t0
    t0 = time.perf_counter(); out["SENSE+DL"]    = sense_recon(ksp_k, Shat, R, lam=lam);times["SENSE+DL"] = time.perf_counter()-t0
    try:
        t0 = time.perf_counter(); out["GRAPPA"]  = grappa_recon(ksp_k, acs, R, maps=Shat); times["GRAPPA"] = time.perf_counter()-t0
    except ValueError:
        pass  # ACS too small to calibrate any kernel -- recorded as missing

    _, gmap = sense_recon(ksp_k, Shat, R, return_gfactor=True)
    body = np.abs(x) > 0.05 * np.abs(x).max()

    rows = []
    for name, rec in out.items():
        m = summarize(x, rec, lesion_center=LESION)
        m.update(method=name, seed=seed, R=R, R_eff=effective_acceleration(mask),
                 n_coils=len(keep), dropped=drop, map_err=map_err, noise=noise,
                 n_acs=n_acs, lam=lam, runtime_ms=times[name]*1e3,
                 g_mean=float(gmap[body].mean()), g_max=float(gmap[body].max()))
        rows.append(m)
    return rows


SWEEPS = {
    "acceleration": ("R",       [2, 3, 4, 5, 6, 8]),
    "aperture":     ("nc",      [4, 6, 8, 12, 16]),
    "dropout":      ("drop",    [0, 1, 2, 3, 4]),
    "map_error":    ("map_err", [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]),
    "snr":          ("noise",   [5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2]),
    "acs":          ("n_acs",   [12, 16, 24, 32, 48]),
    "loading":      ("lam",     [0.0, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1]),
}


def main(out_csv="results/all_experiments.csv"):
    rows = []
    for sweep, (key, values) in SWEEPS.items():
        t0 = time.time()
        for v in values:
            cfg = dict(DEFAULTS); cfg[key] = v
            for s in range(N_SEEDS):
                for r in one_run(seed=1000 + s, **cfg):
                    r["sweep"] = sweep; r["level"] = v
                    rows.append(r)
        print(f"  {sweep:13} {len(values)} levels x {N_SEEDS} seeds   {time.time()-t0:5.1f}s")
    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\n{len(df)} rows -> {out_csv}")
    return df


if __name__ == "__main__":
    print(f"Running {len(SWEEPS)} sweeps x {N_SEEDS} seeds ...")
    main()
