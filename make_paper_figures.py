"""Figures 1-3 for paper_miccai.tex, vector PDF, from results/v3 and fig3_data.npz."""
import sys, glob, os; sys.path.insert(0, "src")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from stats import ci
from simulate import uniform_mask

plt.rcParams.update({"pdf.fonttype": 42, "font.family": "sans-serif", "font.size": 7.5,
                     "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 6.8,
                     "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
                     "axes.spines.top": False, "axes.spines.right": False})
C = {"GRAPPA": "#D1495B", "SENSE": "#7A5195", "SENSE+DL": "#2A7F86", "ZF": "#666666"}
READERS = [("swin", "SwinUNETR"), ("nnunet", "nnU-Net"), ("spikebridge", "spiking")]
OUT = "figures/paper"

# ---------------------------------------------------------------- Fig 1 --
def fig1():
    ny, R, nacs = 184, 8, 64
    mask_s, acs = uniform_mask(ny, 1, R, nacs)
    uni = np.zeros(ny, bool); uni[::R] = True
    shared = uni.copy(); shared[acs] = True
    fig, axs = plt.subplots(3, 1, figsize=(4.7, 1.2), sharex=True,
                            gridspec_kw=dict(hspace=0.55))
    rows = [("acquired: uniform lines + survey (87 lines)", uni, acs), ("uniform: every arm reconstructs", uni, None),
            ("shared-ACS: every arm reconstructs", shared, None)]
    for ax, (title, m, a) in zip(axs, rows):
        ax.vlines(np.nonzero(m)[0], 0, 1, color="#16262E", lw=0.9)
        if a is not None:
            ax.add_patch(Rectangle((a.start, 0), a.stop - a.start, 1, color="#E4B04A", alpha=0.55, lw=0))
            ax.text(a.start + (a.stop - a.start) / 2, 0.5, "64-line ACS survey: calibration only", ha="center", va="center", fontsize=6.0, color="#5a4000",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
        n = int(m.sum()); tot = n + (nacs if a is not None or title.startswith("uniform") else 0)
        lab = f"{n} + {nacs} lines" if a is not None else f"{n} lines"
        ax.text(ny + 3, 0.5, f"{lab}\n$R_\\mathrm{{rec}}$ {ny/n:.2f}, $R_\\mathrm{{tot}}$ {ny/tot:.2f}", va="center", fontsize=6.2)
        ax.set_title(title, loc="left", fontsize=7.2, pad=2)
        ax.set_yticks([]); ax.set_xlim(0, ny + 48); ax.spines["left"].set_visible(False)
    axs[-1].set_xlabel("phase-encode line (nominal $R=8$, 184-line grid)")
    fig.savefig(f"{OUT}/fig1_linesets.pdf", bbox_inches="tight")

# ---------------------------------------------------------------- Fig 2 --
def sweep(files, methods):
    d = pd.concat([pd.read_csv(f) for f in files]); d = d[d.R > 1]
    rows = []
    for r, _ in READERS:
        for reff, g in d[d.reader == r].groupby("R_eff"):
            piv = g.pivot_table(index="patient", columns="method", values="dice_wt")
            for m in methods:
                if m not in piv: continue
                a, lo, hi = ci((piv[m] - piv["Zero-filled"]).dropna())
                rows.append(dict(reader=r, R_eff=reff, method=m, d=a, lo=lo, hi=hi, zf=piv["Zero-filled"].mean()))
    return pd.DataFrame(rows)

def fig2():
    fixed = sweep(sorted(glob.glob("results/v3/reff*.csv")), ["GRAPPA", "SENSE", "SENSE+DL"])
    ach = sweep(["results/v3/acs16.csv", "results/v3/acs24.csv", "results/v3/acs32.csv", "results/v3/sharedacs.csv"], ["GRAPPA", "SENSE+DL"])
    ach = ach[~((ach.R_eff < 2.0))]                       # sharedacs R=2,4 rows not part of the sweep
    fig, axs = plt.subplots(2, 3, figsize=(4.9, 1.40), sharex=True, sharey="row",
                            gridspec_kw=dict(hspace=0.25, wspace=0.12))
    for j, (r, name) in enumerate(READERS):
        for i, (df, title) in enumerate([(fixed, "calibration fixed at 64 lines"), (ach, "ACS block acquired and calibrating")]):
            ax = axs[i, j]; s = df[df.reader == r]
            ax.axhline(0, color="#999", lw=0.7)
            for m in ["GRAPPA", "SENSE", "SENSE+DL"]:
                q = s[s.method == m].sort_values("R_eff")
                if not len(q): continue
                ax.plot(q.R_eff, q.d, "-o", color=C[m], ms=2.6, lw=1.2, label=m, zorder=3)
                ax.fill_between(q.R_eff, q.lo, q.hi, color=C[m], alpha=0.18, lw=0)
            ax.set_xscale("log"); ax.set_xticks([2, 3, 4, 6, 8]); ax.set_xticklabels(["2", "3", "4", "6", "8"])
            ax.minorticks_off(); ax.grid(alpha=0.25, lw=0.5)
            if i == 0: ax.set_title(name)
            if j == 0: ax.text(-0.02, 1.02, ["(a)", "(b)"][i], transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, weight="bold")
        axs[1, j].set_xlabel("$R_\\mathrm{rec}$")
    # Legend goes in the upper left of panel (a). Measured on the data the figure
    # actually plots (all reff*.csv concatenated, so plain SENSE is included):
    # for R_rec <= 4.97 the highest interval edge is +0.196, so x <= 5,
    # y in [+0.23, +0.68] is empty in every column. An earlier placement used
    # -0.192 as the floor, which came from a per-file loop that silently skipped
    # plain SENSE (its file carries no Zero-filled column); the true floor is
    # -0.423 and the bottom band is not empty.
    # One row inside panel (a): measured on the data the figure plots, for
    # R_rec <= 6.13 the highest interval edge is +0.359, so a single-row legend
    # sitting above +0.45 and left of 6.3 clears everything. (A three-row legend
    # does not fit: the free band is 0.45 units and it needs 0.56.)
    axs[0, 0].legend(loc="upper left", bbox_to_anchor=(2.05, 0.675),
                     bbox_transform=axs[0, 0].transData, ncol=3, frameon=False,
                     handlelength=0.9, borderpad=0.0, columnspacing=0.6,
                     handletextpad=0.25, fontsize=4.9)
    axs[0, 0].set_ylim(-0.45, 0.68); axs[1, 0].set_ylim(-0.5, 0.15)
    fig.supylabel("$\\Delta$ Dice vs ZF", fontsize=7, x=0.035)
    fig.savefig(f"{OUT}/fig2_crossover.pdf", bbox_inches="tight")

# ---------------------------------------------------------------- Fig 3 --
def fig3():
    z = np.load(f"{OUT}/fig3_data.npz", allow_pickle=True)
    seg = z["seg"] > 0
    cols = [("uniform", "Zero-filled"), ("uniform", "GRAPPA"), ("uniform", "SENSE"),
            ("shared-acs", "Zero-filled"), ("shared-acs", "GRAPPA"), ("shared-acs", "SENSE")]
    fig, axs = plt.subplots(1, 7, figsize=(4.9, 0.95), gridspec_kw=dict(wspace=0.04))
    def show(ax, img, pred, title):
        ys, xs = np.nonzero(z["t1c_full"] > 0)
        crop = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
        im = np.abs(img)[crop]; ax.imshow(im.T, cmap="gray", vmin=0, vmax=np.percentile(im[im > 0], 99.5), origin="lower")
        ax.contour(seg[crop].T, levels=[0.5], colors="#2ecc71", linewidths=0.55)
        ax.contour((pred[crop] > 0).T, levels=[0.5], colors="#ff3b30", linewidths=0.55)
        ax.set_title(title, fontsize=6.4, pad=2); ax.axis("off")
    show(axs[0], z["t1c_full"], z["pred_full"], "fully sampled")
    for ax, (samp, m) in zip(axs[1:], cols):
        show(ax, z[f"{samp}|{m}|img"], z[f"{samp}|{m}|pred"], f"{'uniform' if samp=='uniform' else 'shared-ACS'}\n{m.replace('Zero-filled','zero-filled')}")
    fig.text(0.34, 0.02, "$R_\\mathrm{rec}=8.0$ (23 lines)", ha="center", fontsize=6.6)
    fig.text(0.77, 0.02, "$R_\\mathrm{rec}=2.33$ (79 lines)", ha="center", fontsize=6.6)
    fig.savefig(f"{OUT}/fig3_examples.pdf", bbox_inches="tight")

if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3"]
    if "1" in which: fig1()
    if "2" in which: fig2()
    if "3" in which: fig3()
    print("done", which)
