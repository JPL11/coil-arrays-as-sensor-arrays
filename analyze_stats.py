"""Cluster bootstrap confidence intervals for the headline claims.

Slices are not independent: three come from each volume, and each BraTS case
contributes many conditions. A naive per-row CI would understate the spread by
treating correlated measurements as independent samples. Every interval here
resamples whole *clusters* (volumes for the fastMRI results, cases for the
tumor results) with replacement, which is the right unit of independence.

Contrasts are also computed *paired* -- the same slice under both conditions --
so anatomy variation cancels rather than dominating, which is exactly what made
the simulation's "SENSE variance grows 40x" claim look like it failed.
"""
from __future__ import annotations
import numpy as np, pandas as pd

N_BOOT = 5000
RNG = np.random.default_rng(0)


def cluster_bootstrap(df, cluster, value, n_boot=N_BOOT, stat=np.mean):
    """CI for a statistic, resampling whole clusters with replacement."""
    groups = [g[value].to_numpy() for _, g in df.groupby(cluster)]
    if not groups:
        return np.nan, np.nan, np.nan
    obs = stat(np.concatenate(groups))
    idx = np.arange(len(groups))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = RNG.choice(idx, size=len(idx), replace=True)
        boots[b] = stat(np.concatenate([groups[i] for i in pick]))
    return obs, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def paired(df, key, cond_a, cond_b, unit, value="ssim", cluster="volume"):
    """Paired difference (b - a) on matched units, with a cluster bootstrap CI."""
    a = df[df[key] == cond_a].set_index(unit)[value]
    b = df[df[key] == cond_b].set_index(unit)[value]
    common = a.index.intersection(b.index)
    d = (b.loc[common] - a.loc[common]).reset_index()
    d.columns = list(d.columns[:-1]) + ["delta"]
    d[cluster] = [i[0] if isinstance(i, tuple) else i for i in common]
    return cluster_bootstrap(d, cluster, "delta"), len(common)


def fmt(o, lo, hi):
    star = "" if (lo <= 0 <= hi) else "  *"
    return f"{o:+.4f}  [{lo:+.4f}, {hi:+.4f}]{star}"


def main():
    df = pd.read_csv("results/fastmri_experiments.csv")
    df["unit"] = list(zip(df.volume, df.slice_idx))
    print("=" * 76)
    print("CLUSTER BOOTSTRAP (resampling volumes), 95% CI.  * = excludes zero")
    print("=" * 76)

    me = df[df.sweep == "map_error"].copy(); me["level"] = me["level"].astype(float)
    print("\n[1] Steering-vector mismatch 0 -> 0.4, paired per slice")
    for meth in ["SENSE", "SENSE+DL", "GRAPPA"]:
        d = me[me.method == meth]
        (o, lo, hi), n = paired(d, "level", 0.0, 0.4, "unit")
        print(f"   {meth:10s} delta SSIM = {fmt(o,lo,hi)}   (n={n} slices)")

    pw = df[df.sweep == "prewhiten"]
    print("\n[2] Noise pre-whitening (off -> on), paired per slice")
    for meth in ["SENSE", "SENSE+DL", "GRAPPA", "Zero-filled"]:
        d = pw[pw.method == meth]
        (o, lo, hi), n = paired(d, "whiten", False, True, "unit")
        print(f"   {meth:12s} delta SSIM = {fmt(o,lo,hi)}   (n={n})")

    acs = df[df.sweep == "acs"].copy(); acs["level"] = acs["level"].astype(float)
    print("\n[3] GRAPPA - SENSE at each calibration budget (paired per slice)")
    for lvl in sorted(acs.level.unique()):
        d = acs[acs.level == lvl]
        (o, lo, hi), n = paired(d, "method", "SENSE", "GRAPPA", "unit")
        print(f"   n_acs={int(lvl):3d}   GRAPPA - SENSE = {fmt(o,lo,hi)}")

    tp = df[df.sweep == "taper"]
    print("\n[4] Hann taper (on -> off), paired per slice")
    for meth in ["SENSE", "SENSE+DL", "GRAPPA", "Zero-filled"]:
        d = tp[tp.method == meth]
        (o, lo, hi), n = paired(d, "hann", True, False, "unit")
        print(f"   {meth:12s} delta SSIM = {fmt(o,lo,hi)}")

    # --- tumor Dice ---------------------------------------------------------
    import glob
    parts = [pd.read_csv(f) for f in sorted(glob.glob("results/brats_tumor_dice*.csv"))]
    tb = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["case", "R", "method"])
    print("\n" + "=" * 76)
    print(f"TUMOR DICE -- cluster bootstrap over {tb.case.nunique()} BraTS cases")
    print("=" * 76)
    for R in sorted(tb.R.unique()):
        for meth in sorted(tb[tb.R == R].method.unique()):
            d = tb[(tb.R == R) & (tb.method == meth)]
            o, lo, hi = cluster_bootstrap(d, "case", "dice_wt")
            o2, lo2, hi2 = cluster_bootstrap(d, "case", "dice_tc")
            print(f"  R={R} {meth:26s} WT {o:.3f} [{lo:.3f},{hi:.3f}]   "
                  f"TC {o2:.3f} [{lo2:.3f},{hi2:.3f}]")
    print("\nPaired vs simulated fully-sampled (R=1), same cases:")
    base = tb[tb.R == 1].set_index("case")["dice_wt"]
    for R in sorted(tb[tb.R > 1].R.unique()):
        for meth in sorted(tb[tb.R == R].method.unique()):
            d = tb[(tb.R == R) & (tb.method == meth)].set_index("case")["dice_wt"]
            common = base.index.intersection(d.index)
            diff = pd.DataFrame({"delta": (d.loc[common] - base.loc[common]).to_numpy(),
                                 "case": common})
            o, lo, hi = cluster_bootstrap(diff, "case", "delta")
            print(f"  R={R} {meth:26s} delta WT = {fmt(o,lo,hi)}")


if __name__ == "__main__":
    main()
