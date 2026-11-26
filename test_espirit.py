"""Correctness guards for the ESPIRiT implementation.

Both checks are cheap and catch the failure that actually happened: a
convention error that leaves every Gram eigenvalue at 1 -- so the method looks
converged -- while returning maps orthogonal to the true sensitivities.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from simulate import shepp_logan, birdcage_maps, multicoil_kspace, uniform_mask, ifft2c
from recon import espirit_maps


def test_parseval(N=128, K=6, nc=8):
    """Retaining every singular vector must make the per-pixel Gram exactly I."""
    x, S = shepp_logan(N), birdcage_maps(nc, N)
    k = multicoil_kspace(x, S, noise_std=0.0, seed=0)
    _, acs = uniform_mask(N, N, 4, 32)
    W = sliding_window_view(k[:, acs, :], (K, K), axis=(1, 2))
    A = np.ascontiguousarray(np.moveaxis(W, 0, 2)).reshape(-1, nc * K * K)
    _, V = np.linalg.eigh(A.conj().T @ A)
    ker = V.T.reshape(-1, nc, K, K).conj()
    G = np.zeros((N, N, nc, nc), complex); buf = np.zeros((nc, N, N), complex)
    y0 = N // 2 - K // 2
    for j in range(ker.shape[0]):
        buf[:] = 0; buf[:, y0:y0 + K, y0:y0 + K] = ker[j]
        m = ifft2c(buf) * np.sqrt(N * N) / K
        G += np.einsum('cyx,dyx->yxcd', m, m.conj(), optimize=True)
    err = np.abs(G - np.eye(nc)[None, None]).max()
    assert err < 1e-10, f"Parseval failed: max|G - I| = {err:.3e}"
    return err


def test_alignment(N=128, nc=8, tol=0.98):
    """Recovered maps must align with the known birdcage sensitivities."""
    x, S_true = shepp_logan(N), birdcage_maps(nc, N)
    k = multicoil_kspace(x, S_true, noise_std=0.0, seed=0)
    _, acs = uniform_mask(N, N, 4, 32)
    body = np.abs(x) > 0.05 * np.abs(x).max()
    S = espirit_maps(k, acs, k=6, c=0.01, eig_thresh=0.0)
    a, b = S[:, body], S_true[:, body]
    a = a / np.maximum(np.linalg.norm(a, axis=0), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=0), 1e-12)
    al = float(np.abs(np.sum(np.conj(a) * b, axis=0)).mean())
    assert al > tol, f"alignment {al:.4f} below {tol} (chance = {1/np.sqrt(nc):.3f})"
    return al


if __name__ == "__main__":
    e = test_parseval();  print(f"  Parseval  max|G - I| = {e:.3e}   PASS")
    a = test_alignment(); print(f"  Alignment vs birdcage = {a:.4f}     PASS (chance 0.354)")
