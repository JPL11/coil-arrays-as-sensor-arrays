"""Confound-free aperture sweep.

The aperture sweep in `run_fastmri.py` compares array sizes across whatever
volumes happen to have that many coils, so each level is measured on a
different set of patients -- which is why it reported 20 coils as *worse* than
16. This restricts every level to the same volumes (those with >= 16 coils) and
draws coil subsets from them, so array size is the only thing that varies.

Same class of confound as the ACS/R_eff one already documented in the README.

Readout oversampling: this script passes `fd.load_slice` straight to
`one_slice`, which is now correct -- `one_slice` removes the 2x readout
oversampling itself -- but was not before the learned reconstructor was added,
when `one_slice` expected k-space with the oversampling already removed and
`run_fastmri.main` did it on the way in. Every number this script produced
before that refactor was therefore computed on the 640-wide uncropped FOV,
including the empty oversampled region, which depresses every metric. The
numbers below supersede those.
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
import numpy as np, pandas as pd
import fastmri_data as fd
from run_fastmri import one_slice, DEFAULTS, ROOT, N_SLICES
from learned_recon import (load_varnet, DEFAULT_CKPT, load_unet, UNET_CKPT)
from gpu_guard import gpu_slot

_slot = gpu_slot(need_mb=5500, label='aperture')
_slot.__enter__()   # module-level script: held until the process exits

MIN_COILS = 16
LEVELS    = [4, 8, 12, 16]

vols, _ = fd.list_volumes(ROOT)
vols = [v for v in vols if v["n_coils"] >= MIN_COILS]
print(f"{len(vols)} volumes with >= {MIN_COILS} coils: "
      f"{sorted({v['n_coils'] for v in vols})}")

varnet = load_varnet() if os.path.exists(DEFAULT_CKPT) else None
unet = load_unet() if os.path.exists(UNET_CKPT) else None
print(f"learned models: VarNet {'on' if varnet else 'off'}, "
      f"U-Net {'on' if unet else 'off'}")

slices = []
for v in vols:
    mid = v["n_slices"] // 2
    for sl in range(mid - N_SLICES // 2, mid - N_SLICES // 2 + N_SLICES):
        slices.append((v, sl, fd.load_slice(v["path"], sl)))
print(f"cached {len(slices)} slices (same slices at every level)\n")

rows = []
for nc in LEVELS:
    t0 = time.time()
    for i, (v, sl, k) in enumerate(slices):
        cfg = dict(DEFAULTS); cfg["n_coils"] = nc
        for r in one_slice(k, seed=1000 + i, varnet=varnet, unet=unet, **cfg):
            r.update(sweep="aperture_fixed", level=nc,
                     volume=os.path.basename(v["path"]), slice_idx=sl,
                     acquisition=v["acquisition"])
            rows.append(r)
    print(f"  n_coils={nc:2d}   {time.time()-t0:6.1f}s")

df = pd.DataFrame(rows)
df.to_csv("results/fastmri_aperture_fixed.csv", index=False)
print(f"\n{len(df)} rows -> results/fastmri_aperture_fixed.csv\n")
m = df.pivot_table(index="level", columns="method", values="ssim", aggfunc="mean")
s = df.pivot_table(index="level", columns="method", values="ssim", aggfunc="std")
print("SSIM vs fully-sampled RSS (same 27 slices at every level):")
for lvl in m.index:
    print(f"  {lvl:>3} coils | " + "  ".join(
        f"{c}={m.loc[lvl,c]:.3f}+/-{s.loc[lvl,c]:.3f}" for c in m.columns))
