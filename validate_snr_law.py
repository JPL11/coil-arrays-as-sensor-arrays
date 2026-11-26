"""Replication check against the original papers' own quantitative claims.

Everything in the README's original "Verification status" table tests *internal
consistency* -- a forward-model round trip, exact recovery with exact maps, the
aliasing index mapping. None of it tests a claim the papers actually make, so
none of it could have caught a faithful-looking implementation that gets the
noise behaviour wrong.

The central quantitative result of Pruessmann et al. 1999 is

    SNR_R = SNR_full / (g * sqrt(R)),    g = sqrt[(S^H Psi^-1 S)^-1_pp (S^H Psi^-1 S)_pp]

i.e. the analytically computed g-factor must predict the noise amplification
actually produced by the reconstruction. That is checked here by the
pseudo-replica method (Robson et al. 2008): many independent noise realizations
are pushed through the same reconstructor, the per-pixel standard deviation is
measured, and the measured amplification is compared against g * sqrt(R).

A second check covers Griswold et al. 2002: with noiseless data and an ACS block
large enough to calibrate the kernel, GRAPPA must recover the fully sampled
image, because the kernel that reproduces the ACS lines exactly also fills the
missing ones exactly.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
import numpy as np

from simulate import (shepp_logan, birdcage_maps, multicoil_kspace, uniform_mask,
                      fft2c, ifft2c)
from recon import sense_recon, grappa_recon, estimate_maps_acs, rss

N_REP = 96


def correlated_noise(shape, Psi, rng):
    """White complex noise coloured by the Cholesky factor of Psi."""
    nc = Psi.shape[0]
    w = (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2)
    L = np.linalg.cholesky(Psi + 1e-12 * np.eye(nc))
    return np.einsum('ij,j...->i...', L, w)


def snr_law(N=128, nc=8, R=4, sigma=2e-3, Psi=None, seed=0, label=""):
    x = shepp_logan(N); S = birdcage_maps(nc, N)
    k0 = multicoil_kspace(x, S, noise_std=0.0, seed=0)
    if Psi is None:
        Psi = np.eye(nc)
    rng = np.random.default_rng(seed)

    full, red = [], []
    for i in range(N_REP):
        k = k0 + sigma * correlated_noise((nc, N, N), Psi, rng)
        full.append(sense_recon(k, S, 1, Psi=Psi))
        red.append(sense_recon(k, S, R, Psi=Psi))
    full = np.stack(full); red = np.stack(red)

    # per-pixel noise: std of the complex estimate over replicas
    n_full = np.std(full, axis=0).real if False else np.sqrt(
        0.5 * (np.var(full.real, axis=0) + np.var(full.imag, axis=0)) * 2)
    n_red = np.sqrt(0.5 * (np.var(red.real, axis=0) + np.var(red.imag, axis=0)) * 2)

    _, g = sense_recon(k0, S, R, return_gfactor=True, Psi=Psi)
    body = np.abs(x) > 0.1 * np.abs(x).max()
    measured = n_red[body] / np.maximum(n_full[body], 1e-30) / np.sqrt(R)
    predicted = g[body]
    ratio = measured / np.maximum(predicted, 1e-12)
    print(f"  {label:28s} R={R}  g_pred={predicted.mean():.4f}  "
          f"g_meas={measured.mean():.4f}  ratio={ratio.mean():.4f} "
          f"+/-{ratio.std():.4f}  (median {np.median(ratio):.4f})")
    return float(np.median(ratio))


def grappa_exact(N=128, nc=8, R=2, n_acs=48):
    """Griswold 2002: noiseless, well-calibrated GRAPPA must be near-exact."""
    x = shepp_logan(N); S = birdcage_maps(nc, N)
    k = multicoil_kspace(x, S, noise_std=0.0, seed=0)
    _, acs = uniform_mask(N, N, R, n_acs)
    img = grappa_recon(k, acs, R, lam=1e-12)
    ref = rss(ifft2c(k))
    err = float(np.linalg.norm(np.abs(img) - ref) / np.linalg.norm(ref))
    print(f"  GRAPPA noiseless, R={R}, {n_acs} ACS: relative error = {err:.3e}")
    return err


if __name__ == "__main__":
    print("Pruessmann 1999, SNR_R = SNR_full / (g sqrt(R)) -- pseudo-replica check")
    print(f"  ({N_REP} noise replicas; ratio should be 1.0 if g predicts the noise)")
    for R in (2, 4):
        snr_law(R=R, Psi=None, label="white noise (Psi = I)")
    # correlated coil noise: the case where Psi actually matters
    rng = np.random.default_rng(3)
    Q = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    Psi = Q @ Q.conj().T / 8 + 0.5 * np.eye(8)
    Psi = Psi / (np.trace(Psi).real / 8)
    d = np.sqrt(np.real(np.diag(Psi)))
    print(f"\n  correlated Psi: max |off-diagonal correlation| = "
          f"{np.abs(Psi/np.outer(d,d) - np.eye(8)).max():.3f}")
    for R in (2, 4):
        snr_law(R=R, Psi=Psi, label="correlated noise, Psi used")
    print("\n  control: correlated noise but Psi ignored in the solve (Psi=I)")
    for R in (4,):
        # reconstruct with the wrong (white) model but generate correlated noise
        N, nc = 128, 8
        x = shepp_logan(N); S = birdcage_maps(nc, N)
        k0 = multicoil_kspace(x, S, noise_std=0.0, seed=0)
        rng2 = np.random.default_rng(1)
        full, red = [], []
        for _ in range(N_REP):
            k = k0 + 2e-3 * correlated_noise((nc, N, N), Psi, rng2)
            full.append(sense_recon(k, S, 1)); red.append(sense_recon(k, S, R))
        full = np.stack(full); red = np.stack(red)
        nf = np.sqrt(np.var(full.real, 0) + np.var(full.imag, 0))
        nr = np.sqrt(np.var(red.real, 0) + np.var(red.imag, 0))
        _, g = sense_recon(k0, S, R, return_gfactor=True)      # white-noise g
        body = np.abs(x) > 0.1 * np.abs(x).max()
        meas = nr[body] / np.maximum(nf[body], 1e-30) / np.sqrt(R)
        print(f"  {'white g vs correlated noise':28s} R={R}  "
              f"g_pred={g[body].mean():.4f}  g_meas={meas.mean():.4f}  "
              f"ratio={np.median(meas/np.maximum(g[body],1e-12)):.4f}")
    print("\nGriswold 2002, exact reconstruction in the noiseless limit")
    for R in (2, 3, 4):
        grappa_exact(R=R)
