"""Figures for the real-data half of the study.

Palette is the validated 4-slot categorical set (checked with the dataviz
validator: lightness band, chroma floor, CVD separation, normal-vision floor
all PASS). Contrast against the surface is below 3:1 for two slots, so every
series is directly labelled as well as legended -- identity is never carried by
colour alone.
"""
from __future__ import annotations
import os, glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = {"Zero-filled": "#2a78d6", "SENSE": "#eb6834",
          "SENSE+DL": "#1baf7a", "GRAPPA": "#eda100", "VarNet": "#8d54d1"}
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"
SURFACE = "#fcfcfb"
RNG = np.random.default_rng(0)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": INK3, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#e6e5e0", "grid.linewidth": 0.8,
})


def boot_ci(groups, n=2000):
    """Cluster bootstrap mean + 95% CI; `groups` is a list of per-cluster arrays."""
    if not groups:
        return np.nan, np.nan, np.nan
    obs = float(np.mean(np.concatenate(groups)))
    idx = np.arange(len(groups))
    b = np.array([np.mean(np.concatenate([groups[i] for i in RNG.choice(idx, len(idx), True)]))
                  for _ in range(n)])
    return obs, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def label_right(ax, x, y, text, color, dy=0):
    ax.annotate(text, xy=(x, y), xytext=(6, dy), textcoords="offset points",
                color=INK2, fontsize=8, va="center", ha="left")


def stagger(vals, min_gap):
    """Nudge near-coincident label positions apart, preserving their order."""
    order = np.argsort(vals); out = list(vals)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < min_gap:
            out[b] = out[a] + min_gap
    return out


# ---------------------------------------------------------------- figure 1
def fig_tumor_dice(out="figures/tumor_dice_vs_R.png"):
    parts = [pd.read_csv(f) for f in sorted(glob.glob("results/brats_tumor_dice*.csv"))]
    df = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["case", "R", "method"])
    Rs = [2, 4, 8]; xs = np.arange(len(Rs))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharex=True)
    for ax, metric, title in zip(axes, ["dice_wt", "dice_tc"],
                                 ["Whole tumor", "Tumor core"]):
        base = df[df.R == 1]
        m, lo, hi = boot_ci([g[metric].to_numpy() for _, g in base.groupby("case")])
        # drawn across the data range only -- a full-width axhline strikes
        # through the direct labels in the right margin
        ax.plot(xs, [m] * len(xs), color=INK3, lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.annotate("fully sampled", xy=(xs[0], m), xytext=(0, 6),
                    textcoords="offset points", color=INK3, fontsize=8, ha="left")
        ends = {}
        for meth, col in SERIES.items():
            ys, los, his = [], [], []
            for R in Rs:
                d = df[(df.R == R) & (df.method == meth)]
                a, b, c = boot_ci([g[metric].to_numpy() for _, g in d.groupby("case")])
                ys.append(a); los.append(b); his.append(c)
            ax.fill_between(xs, los, his, color=col, alpha=0.13, lw=0, zorder=2)
            ax.plot(xs, ys, color=col, lw=2, marker="o", ms=5.5, zorder=3,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=meth)
            ends[meth] = ys[-1]
        names = list(ends); placed = stagger([ends[n] for n in names], 0.055)
        for n, yy in zip(names, placed):
            label_right(ax, xs[-1], ends[n], n, SERIES[n], dy=(yy - ends[n]) * 320)
        ax.set_xticks(xs); ax.set_xticklabels([f"R={r}" for r in Rs])
        ax.set_xlim(-0.25, len(Rs) - 1 + 1.35)
        ax.set_ylim(0, 1.0); ax.set_ylabel("Dice" if metric == "dice_wt" else "")
        ax.set_title(title, color=INK, fontsize=10, loc="left", pad=8)
        ax.grid(axis="y", zorder=0); ax.set_axisbelow(True)
    fig.suptitle("Tumor delineation survives to R=4, then SENSE fails outright",
                 x=0.02, ha="left", fontsize=11.5, color=INK)
    fig.text(0.02, 0.005, f"{df.case.nunique()} BraTS-GLI cases · bands are 95% cluster "
             f"bootstrap over cases · array borrowed from real fastMRI acquisition",
             fontsize=7.5, color=INK3, ha="left")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK2)
    fig.tight_layout(rect=[0, 0.03, 1, 0.93]); fig.savefig(out, dpi=200)
    plt.close(fig); print("wrote", out)


# ---------------------------------------------------------------- figure 2
def fig_mismatch(out="figures/mismatch_sim_vs_real.png"):
    real = pd.read_csv("results/fastmri_experiments.csv")
    real = real[real.sweep == "map_error"].copy(); real["level"] = real["level"].astype(float)
    sim = pd.read_csv("results/all_experiments.csv")
    sim = sim[sim.sweep == "map_error"].copy(); sim["level"] = sim["level"].astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=False)
    for ax, df, title, clus in [(axes[0], sim, "Simulation (birdcage array)", "seed"),
                                (axes[1], real, "Real fastMRI array", "volume")]:
        lv = sorted(df.level.unique())
        for meth, col in SERIES.items():
            if meth not in set(df.method): continue
            ys = [df[(df.level == l) & (df.method == meth)]["ssim"].mean() for l in lv]
            ax.plot(lv, ys, color=col, lw=2, marker="o", ms=5.5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=meth)
            label_right(ax, lv[-1], ys[-1], meth, col)
        ax.set_xlabel("sensitivity-map error (steering-vector mismatch)")
        ax.set_xlim(min(lv) - 0.02, max(lv) + 0.13)
        ax.set_title(title, color=INK, fontsize=10, loc="left", pad=8)
        ax.grid(axis="y"); ax.set_axisbelow(True)
    axes[0].set_ylabel("SSIM")
    fig.suptitle("GRAPPA's immunity to steering-vector mismatch survives on real coils",
                 x=0.02, ha="left", fontsize=11.5, color=INK)
    fig.text(0.02, 0.005, "SENSE loses 0.24 SSIM in simulation and 0.064 on real data -- "
             "it starts already degraded by real map-estimation error. GRAPPA is flat in both.",
             fontsize=7.5, color=INK3, ha="left")
    fig.tight_layout(rect=[0, 0.03, 1, 0.93]); fig.savefig(out, dpi=200)
    plt.close(fig); print("wrote", out)


# ---------------------------------------------------------------- figure 3
def fig_panel(out="figures/recon_panel_real.png"):
    import sys; sys.path.insert(0, "src")
    import fastmri_data as fd
    from simulate import uniform_mask, ifft2c
    from recon import estimate_maps_acs, sense_recon, grappa_recon, zerofill_recon
    from learned_recon import load_varnet, varnet_recon, DEFAULT_CKPT
    p = "../datasets/fastmri/multicoil_val/file_brain_AXT1PRE_200_6002079.h5"
    k_full = fd.load_slice(p, 8)                  # readout oversampling intact:
    k = fd.remove_readout_oversampling(k_full)    # what the network was trained on
    varnet = load_varnet() if os.path.exists(DEFAULT_CKPT) else None
    ny, nx = k.shape[1:]
    ref = np.sqrt((np.abs(ifft2c(k)) ** 2).sum(0)); bm = fd.body_mask(ref)
    Rs = [2, 4, 8]
    ncol = 6 if varnet is not None else 5
    fig, axes = plt.subplots(len(Rs), ncol, figsize=(2.2 * ncol, 2.3 * len(Rs)))
    vmax = np.percentile(ref[bm], 99.5)
    for i, R in enumerate(Rs):
        mask, acs = uniform_mask(ny, nx, R, 24); S = estimate_maps_acs(k, acs)
        imgs = [("fully sampled", ref),
                ("Zero-filled", np.abs(zerofill_recon(k * mask, S))),
                ("SENSE", np.abs(sense_recon(k, S, R))),
                ("SENSE+DL", np.abs(sense_recon(k, S, R, lam=3e-2))),
                ("GRAPPA", np.abs(grappa_recon(k, acs, R, maps=S)))]
        if varnet is not None:
            mask_f, _ = uniform_mask(ny, k_full.shape[-1], R, 24)
            imgs.append(("VarNet", varnet_recon(k_full, mask_f, 24,
                                                crop_ro=True, model=varnet)))
        for j, (nm, im) in enumerate(imgs):
            ax = axes[i, j]; ax.imshow(fd.apply_mask(im, bm), cmap="gray", vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
            if i == 0: ax.set_title(nm, fontsize=9, color=INK, pad=6)
            if j == 0: ax.set_ylabel(f"R = {R}", fontsize=9.5, color=INK2)
    fig.suptitle("Real fastMRI brain, 16-coil array, 24 ACS lines",
                 x=0.02, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(out, dpi=200)
    plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    fig_tumor_dice(); fig_mismatch(); fig_panel()
