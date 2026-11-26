"""Every headline contrast, recomputed through one bootstrap, with Holm control.

Reproduces the numbers quoted throughout the README from the stored CSVs using
`src/stats.py` alone, so that no reported interval depends on the resampling
code that happened to be written alongside its experiment. The primary family is
declared here rather than chosen after seeing the results.
"""
from __future__ import annotations

import os, sys, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import numpy as np
import pandas as pd
import stats as st

GLI = "results/brats_volume32.csv"
GLI_FULL = "results/brats_tumor_dice32_full.csv"
AFR = "results/brats_africa32.csv"


def load_nnunet():
    return pd.concat([pd.read_csv(f) for f in
                      sorted(glob.glob("results/brats_nnunet_shard_*.csv"))])


def load_r1():
    return pd.concat([pd.read_csv(f) for f in
                      sorted(glob.glob("results/brats_r1_shard_*.csv"))])


def contrast(name, family, diff, note=""):
    if diff is None or not len(diff):
        return None
    m, lo, hi = st.ci(diff)
    return dict(contrast=name, family=family, n=len(diff), delta=m,
                lo=lo, hi=hi, p=st.boot_p(diff), note=note)


def main():
    rows = []
    gli = pd.read_csv(GLI_FULL)
    afr = pd.read_csv(AFR)
    nn = load_nnunet()
    r1 = load_r1()

    R8 = lambda d: d.R == 8

    # ---- primary family: the contrasts the paper's claims rest on -----------
    for a, b, lab in [("GRAPPA", "Zero-filled", "classical PI vs no reconstruction"),
                      ("SENSE", "Zero-filled", "SENSE vs no reconstruction"),
                      ("VarNet", "Zero-filled", "unrolled vs no reconstruction"),
                      ("U-Net", "Zero-filled", "image-domain vs no reconstruction")]:
        rows.append(contrast(f"{a} - {b}, Dice WT, R=8 (GLI)", "primary",
                             st.paired(gli, a, b, by=R8), lab))
    # the R=1 control: prior isolated from acceleration
    p = r1[r1.R == 1].pivot_table(index="case", columns="method", values="dice_wt")
    for m in ["VarNet", "U-Net"]:
        if m in p and "Fully sampled" in p:
            rows.append(contrast(f"{m} - fully sampled, Dice WT, R=1", "primary",
                                 (p[m] - p["Fully sampled"]).dropna(),
                                 "prior applied to fully sampled data"))

    # ---- replication family: same claims, second reader and second cohort ---
    for a, b in [("GRAPPA", "Zero-filled"), ("SENSE", "Zero-filled"),
                 ("VarNet", "Zero-filled"), ("U-Net", "Zero-filled")]:
        rows.append(contrast(f"{a} - {b}, Dice WT, R=8 (nnU-Net reader)",
                             "replication", st.paired(nn, a, b, by=R8)))
        rows.append(contrast(f"{a} - {b}, Dice WT, R=8 (BraTS-Africa)",
                             "replication", st.paired(afr, a, b, by=R8)))

    # ---- exploratory -------------------------------------------------------
    for a, b in [("VarNet", "GRAPPA"), ("U-Net", "GRAPPA"), ("SENSE+DL", "SENSE")]:
        rows.append(contrast(f"{a} - {b}, Dice WT, R=8 (GLI)", "exploratory",
                             st.paired(gli, a, b, by=R8)))

    out = st.report([r for r in rows if r])
    pd.set_option("display.width", 150)
    for fam in ["primary", "replication", "exploratory"]:
        d = out[out.family == fam]
        if not len(d):
            continue
        print(f"\n=== {fam.upper()} "
              f"{'(Holm-corrected within family)' if fam == 'primary' else ''} ===")
        print(f"{'contrast':52s} {'n':>3s} {'delta':>8s} {'95% CI':>20s} "
              f"{'p':>8s} {'adj p':>8s}  verdict")
        for _, r in d.iterrows():
            ap = f"{r.adj_p:8.4f}" if not np.isnan(r.adj_p) else " " * 8
            print(f"{r.contrast:52s} {r.n:3d} {r.delta:+8.4f} "
                  f"[{r.lo:+8.4f},{r.hi:+8.4f}] {r.p:8.4f} {ap}  {r.verdict}")
    out.to_csv("results/stats_report.csv", index=False)
    print(f"\nwrote results/stats_report.csv "
          f"({(out.family=='primary').sum()} primary, "
          f"{(out.family=='replication').sum()} replication, "
          f"{(out.family=='exploratory').sum()} exploratory)")


if __name__ == "__main__":
    main()
