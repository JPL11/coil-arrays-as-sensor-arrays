"""Shrinkage estimators for the coil noise covariance.

`prewhiten` applies Psi^-1/2 so that the plain least-squares SENSE solve becomes
the maximum-likelihood estimator. That argument assumes Psi is *known*; in
practice it is estimated from a finite number of noise samples, and a sample
covariance estimated from n snapshots of an nc-element array is badly
conditioned when n/nc is small -- the classical failure that motivates
diagonal loading in adaptive beamforming and shrinkage in covariance
estimation. The README named this as the most likely reason the measured
pre-whitening gain is small. This file tests that explanation rather than
repeating it.

Two standard estimators, both shrinking the sample covariance S towards the
scaled identity F = (tr S / nc) I:

    Psi = rho F + (1 - rho) S

* **Ledoit-Wolf** picks rho to minimise expected squared Frobenius error with
  no distributional assumption.
* **OAS** (Chen et al. 2010) assumes Gaussian samples and is the better choice
  exactly in the small-n regime where shrinkage matters at all.

Both are written for complex Hermitian covariances, which is what a receiver
array has and what the real-valued textbook formulas do not cover: the sample
term uses x x^H rather than x x^T throughout, and every trace is taken of the
real part.
"""
from __future__ import annotations

import numpy as np


def sample_cov(X):
    """S = (1/n) sum x_i x_i^H for X of shape (nc, n), already zero-mean."""
    n = X.shape[1]
    return (X @ X.conj().T) / n


def _target(S):
    nc = S.shape[0]
    mu = float(np.real(np.trace(S)) / nc)
    return mu * np.eye(nc), mu


def ledoit_wolf(X):
    """Ledoit-Wolf shrinkage towards the scaled identity. Returns (Psi, rho)."""
    nc, n = X.shape
    S = sample_cov(X)
    F, _ = _target(S)
    d2 = float(np.real(np.sum(np.abs(S - F) ** 2)))
    # mean squared deviation of the per-sample outer products from S
    b2 = 0.0
    for i in range(n):
        x = X[:, i:i + 1]
        b2 += float(np.real(np.sum(np.abs(x @ x.conj().T - S) ** 2)))
    b2 /= n ** 2
    b2 = min(b2, d2)
    rho = 0.0 if d2 <= 0 else b2 / d2
    rho = float(np.clip(rho, 0.0, 1.0))
    return rho * F + (1 - rho) * S, rho


def oas(X):
    """Oracle Approximating Shrinkage (Chen et al., IEEE TSP 58:5016, 2010)."""
    nc, n = X.shape
    S = sample_cov(X)
    F, mu = _target(S)
    tr_S2 = float(np.real(np.trace(S @ S)))
    tr2_S = float(np.real(np.trace(S)) ** 2)
    num = (1 - 2.0 / nc) * tr_S2 + tr2_S
    den = (n + 1 - 2.0 / nc) * (tr_S2 - tr2_S / nc)
    rho = 1.0 if den <= 0 else float(np.clip(num / den, 0.0, 1.0))
    return rho * F + (1 - rho) * S, rho


def corners(ksp, frac=0.06):
    """Signal-free k-space corners as an (nc, n) zero-mean sample matrix.

    Same region `fastmri_data.noise_covariance` uses, exposed separately so the
    number of samples can be varied -- which is the quantity shrinkage is
    supposed to be sensitive to, and therefore the axis the experiment sweeps.
    """
    nc, ny, nx = ksp.shape
    hy, hx = max(int(ny * frac), 2), max(int(nx * frac), 2)
    X = np.concatenate([
        ksp[:, :hy, :hx].reshape(nc, -1), ksp[:, :hy, -hx:].reshape(nc, -1),
        ksp[:, -hy:, :hx].reshape(nc, -1), ksp[:, -hy:, -hx:].reshape(nc, -1)], axis=1)
    return X - X.mean(axis=1, keepdims=True)


ESTIMATORS = {"sample": lambda X: (sample_cov(X), 0.0),
              "ledoit-wolf": ledoit_wolf,
              "oas": oas}
