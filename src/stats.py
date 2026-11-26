"""One bootstrap implementation, used by every reported contrast.

Each experiment in this study grew its own resampling code, which is how the
same nominal procedure ends up with different seeds, different replicate counts
and -- in one case during development -- a different resampling unit. This module
is the single implementation. It resamples **whole cases or whole volumes**,
never slices, because several slices from one brain are not several independent
observations.

It also does the thing a long study of this shape needs and rarely does: it
separates the contrasts the conclusions rest on from the ones found by looking.
Dozens of intervals have been computed here. Reporting them all at a nominal 95%
and then narrating whichever excluded zero would be the standard way to be wrong.
`PRIMARY` names the contrasts fixed in advance of the final runs; everything else
is exploratory and is labelled as such. Holm-Bonferroni is applied within the
primary family only, which is the family the claims are drawn from.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_BOOT = 5000
SEED = 0


def _draws(per_unit, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = per_unit.index.to_numpy()
    if len(idx) < 2:
        return np.array([per_unit.mean()])
    return np.array([per_unit.reindex(rng.choice(idx, len(idx), replace=True)).mean()
                     for _ in range(n)])


def ci(per_unit, n=N_BOOT, seed=SEED):
    """Point estimate and 95% cluster-bootstrap interval over the index units."""
    d = _draws(per_unit, n, seed)
    return float(per_unit.mean()), *np.nanpercentile(d, [2.5, 97.5])


def boot_p(per_unit, n=N_BOOT, seed=SEED):
    """Two-sided bootstrap p-value for a mean difference against zero.

    The proportion of resampled means on the far side of zero, doubled. Floored
    at 1/n because a bootstrap cannot resolve a p-value below its own
    resolution, and quoting `p < 0.0002` from 5000 draws would be reading
    precision the procedure does not have.
    """
    d = _draws(per_unit, n, seed)
    p = 2.0 * min((d <= 0).mean(), (d >= 0).mean())
    return float(max(p, 1.0 / n))


def paired(df, a, b, unit="case", value="dice_wt", by=None):
    """Per-unit paired difference a - b, optionally within a subset."""
    d = df if by is None else df[by(df)]
    p = d.pivot_table(index=unit, columns="method", values=value)
    if a not in p or b not in p:
        return None
    return (p[a] - p[b]).dropna()


def holm(pvals, alpha=0.05):
    """Holm-Bonferroni step-down. Returns adjusted p-values, order preserved."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def report(rows, alpha=0.05):
    """Assemble a contrast table with Holm correction inside the primary family."""
    df = pd.DataFrame(rows)
    df["adj_p"] = np.nan
    prim = df.family == "primary"
    if prim.any():
        df.loc[prim, "adj_p"] = holm(df.loc[prim, "p"].values, alpha)
    # three families, three meanings. A replication contrast is not exploratory:
    # it is a pre-specified claim re-tested on different data, so it is reported
    # uncorrected against the primary family it is confirming, and labelled as
    # confirming or failing to confirm rather than as a discovery.
    verdict = []
    for _, r in df.iterrows():
        if r.family == "primary":
            verdict.append("significant" if r.adj_p < alpha else "not significant")
        elif r.family == "replication":
            verdict.append("replicates" if r.p < alpha else "does not replicate")
        else:
            verdict.append("exploratory" if r.p < alpha else "null")
    df["verdict"] = verdict
    return df
