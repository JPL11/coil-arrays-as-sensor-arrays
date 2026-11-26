# Array signal processing: state, results, dead ends, and what is left

> **Power-outage recovery:** see `RECOVERY_HANDOFF_2026-10-10.md` for the
> verified paper state, resumed ACS-only compression comparison, and qualifications
> to the minimum-redundancy k-space analogy. The old compression figures below
> used a full-acquisition SVD basis; the recovery run verifies ACS-only compression
> at R=4, K=5 (+0.03328 SSIM versus the old selector, seven volumes / 21 slices).

Handoff for the array-processing half of the parallel-MRI study. Written to be
picked up cold. The downstream/tumour half is documented in `README.md`; this
file covers only the sensor-array work and the one result that connects them.

**Framing.** The receiver-coil array is a sensor array, the coil-sensitivity map
is its steering vector, and SENSE is the array inverse. Everything below follows
from taking that seriously and then checking whether it holds.

> **2026-09-18 — see `HANDOFF_2026-09-18.md` first.** The 2026-08-30 audit
> below was correct and was not acted on: the same `sense_recon` masking defect
> it names was left live in the tumour pipeline and produced the paper's entire
> headline. A review panel caught it on 2026-09-12; everything downstream has
> been re-run and the conclusion inverted. Do not reuse tumour-half numbers
> predating that file.

> **2026-08-30 correction / lean-method audit.** Sections 1 and 2.2--2.4 below
> predate a fairness audit and must not be used as method claims without being
> regenerated. `sense_recon` reconstructs strictly uniform every-Rth lines,
> while zero-filling and the learned methods received the uniform+ACS mask and
> GRAPPA restored the true ACS block. At nominal R=8 with 24 ACS lines this was
> true R=8 SENSE (40/320 lines) versus R_eff=5.25 for the masked methods
> (61/320 lines). Coil selection also used `Psi` in its objective but omitted it
> from the reconstruction; SVD compression neither transformed nor propagated
> `Psi`. These are material differences, not bookkeeping.
>
> `run_coil_select_bias.py` is the corrected, contained test: ACS is a separate
> calibration survey, every reconstruction consumes the same uniform samples,
> generalized least-squares SENSE receives the appropriate covariance, and SVD
> compression propagates `Psi_c = W Psi W^H`. On the original 7-volume / 21-slice
> 20-channel cohort at R=4:
>
> | paired SSIM vs old SNR-law selector | K=5 | K=6 |
> |---|---:|---:|
> | exact GLS noise-risk selection | **+0.0063 [+0.0008, +0.0126]** | +0.0033 [-0.0026, +0.0110] |
> | held-out ACS calibration risk | +0.0015 [-0.0044, +0.0089] | -0.0002 [-0.0074, +0.0072] |
> | covariance-correct SVD compression | **+0.0305 [+0.0220, +0.0402]** | **+0.0217 [+0.0149, +0.0303]** |
>
> The exact criterion is
> `mean(log(diag((S^H Psi^-1 S)^-1)))`: unlike g alone, it contains receive
> strength and unfolding geometry in one quantity. It is the only lean candidate
> that survived, and only at K=5 where the subset is barely overdetermined. The
> proposed held-out-ACS bias term did not help and should be dropped. At R=8,
> K=8 (7 middle slices), noise-risk selection and compression both tie the old
> selector (respectively -0.0000 [-0.0009, +0.0010] and -0.0003
> [-0.0022, +0.0027]); fair uniform zero-filling wins by +0.1375
> [+0.1115, +0.1616]. Thus there is a useful corrected selector result, but not
> yet a broad new reconstruction method, and the compression-crossover claim in
> section 2.4 does not survive this corrected test.

---

## 1. The central result, and it is a negative one

**The g-factor governs SENSE's noise. It does not govern SENSE's failure.**

| R = 8, SENSE | correlation with the analytic g-map |
|---|---|
| noise (per-pixel sd over 64 noise replicas) | **+0.973 [+0.964, +0.982]** |
| error (bias against the fully sampled reference) | **+0.197** |

Standardized regression on the noise gives `beta_g = +0.963 [+0.948, +0.977]`
with `R^2 = 0.947`, so geometry explains 95% of the *noise* variance and about
4% of the *error* variance. At R=8 the error is dominated by bias -- imperfect
sensitivity maps and residual aliasing -- and that term is close to spatially
uniform.

The direct demonstration, which is the thing to re-run first if you doubt this:

| mean absolute error, R=8 | g < 2 | 2 <= g < 5 | g >= 5 |
|---|---|---|---|
| SENSE | 1.94e-4 | 1.66e-4 | 1.57e-4 |
| Zero-filled | 2.69e-5 | 2.40e-5 | 2.93e-5 |

Zero-filling beats SENSE **by 72-86% even where g < 2**, at R = 5, 8 and 10, and
SENSE's error barely varies with g at all. Whatever destroys SENSE at high
acceleration is not the conditioning of the array.

**Consequence: no g-gated method can work.** Two were built and both failed.
Anything that uses the g-map as a spatial gate, weight or trigger is keyed on
the noise term, and the noise term is not what is going wrong. Do not rebuild
this without first refuting the table above.

---

## 2. Positive results

### 2.1 How many noise samples pre-whitening needs (`run_noise_cov.py`)

Pre-whitening makes the plain least-squares SENSE solve maximum-likelihood, and
it buys almost nothing here (+0.007 SSIM at R=4). The earlier explanation -- a
poorly conditioned sample covariance -- is **wrong**: the default corner region
gives n = 1444 for nc = 20, n/nc = 72, condition number 12.0, Ledoit-Wolf
shrinkage intensity 0.02.

Sweeping the sample count is where the result is (SSIM change vs no whitening, R=4):

| n/nc | n | sample | Ledoit-Wolf | OAS |
|---|---|---|---|---|
| 72.2 | 1444 | +0.0067 | +0.0068 | +0.0068 |
| 7.2 | 144 | +0.0035 | +0.0051 | +0.0052 |
| 3.2 | 64 | **-0.0086** | +0.0011 | +0.0016 |
| 1.8 | 36 | **-0.0233** | +0.0006 | +0.0013 |

**Below n/nc ~ 4 the sample covariance makes pre-whitening actively harmful and
shrinkage removes the failure.** Ruled out: signal contamination of the corners
(energy per sample rises only 7.5% from n=16 to n=4096).

**Still open:** why correlated coil noise (off-diagonal up to 0.54) costs so
little in reconstruction quality at any n.

### 2.2 Coil selection by the paper's own SNR law (`run_coil_select.py`)

The aperture sweep answers *how many* coils; this answers *which*. Distinct from
coil compression, which mixes all N channels and needs them all present.

Selecting on g alone is wrong, and the experiment shows it twice. `greedy-g`
achieves the lowest mean g at every K (2.07 vs 2.16 and 2.28 at K=5) and still
loses on SSIM, because g is only the noise penalty in
`SNR_R = SNR_full / (g sqrt(R))` and selecting on it picks well-conditioned but
weakly-receiving elements. Using the whole law as the objective
(`greedy_snr_law`) fixes it, but only where conditioning binds:

| greedy-snr-law - greedy-snr (paired) | R=4 |
|---|---|
| K=5 | **+0.0096 [+0.0023, +0.0163]** |
| K=6 | **+0.0100 [+0.0039, +0.0165]** |
| K=8 | -0.0009 [-0.0039, +0.0020] |

Nothing once K is comfortably above R, and nothing at all at R=8 where the whole
selector spread is 0.007 SSIM against a 0.55 baseline -- acceleration dominates
that regime and no selection rule rescues it.

### 2.3 Criterion beats search (`run_coil_exhaustive.py`)

C(20,5) = 15504 enumerated, ~5 min/slice, 4 volumes:

| | mean g | SSIM |
|---|---|---|
| greedy-g | 2.045 | 0.5794 |
| **exhaustive best-g** | **1.842** | **0.5282** |
| greedy-snr-law | -- | 0.5877 |
| exhaustive best-SNR-law | -- | 0.5895 |

**Optimising g properly is 0.051 SSIM worse than optimising it badly.** Greedy is
11% above the true g minimum and never finds it (0/4), and greedy's weakness was
protecting it. For the right criterion greedy is within 0.0018 of its own
optimum. The criterion carries the difficulty; the search carries almost none.

### 2.4 Coil compression stops helping at high acceleration

SVD compression answers an easier question than selection and should win
outright. It does not (paired, vs `greedy-snr-law`):

| | R=2 | R=4 | R=5 | R=8 | R=10 |
|---|---|---|---|---|---|
| K=8 | **+0.0109** | **+0.0117** | +0.0043 | **-0.0031** | -- |
| K=10 | **+0.0092** | **+0.0109** | **+0.0059** | -0.0014 | **-0.0040** |

Monotone decay with a sign change; crossover between R=5 and R=8 for K=8, later
for K=10 -- more retained components, less diversity given away. Compression
trades spatial diversity for SNR and unfolding needs the diversity as the fold
count grows. **The standard advice to compress before reconstructing is
acceleration-dependent.**

### 2.5 Aperture separates classical from learned

Same 27 slices at every coil count (`run_aperture_fixed.py`):

| SSIM | 4 | 8 | 12 | 16 | delta | monotonic |
|---|---|---|---|---|---|---|
| SENSE | 0.862 | 0.903 | 0.911 | 0.925 | **+0.063** | yes |
| GRAPPA | 0.896 | 0.924 | 0.927 | 0.938 | +0.042 | yes |
| VarNet | 0.979 | 0.977 | 0.976 | 0.981 | **+0.002** | no |
| U-Net | 0.889 | 0.877 | 0.870 | 0.870 | -0.019 | no |

Every method that solves the array inverse improves strictly monotonically with
aperture; neither learned method does. On this axis VarNet behaves like
zero-filling, not like a parallel imaging method.

---

## 3. Things that do not work (do not rebuild without new reason)

| Attempt | Result | Why |
|---|---|---|
| Diagonal loading against mismatch | -0.0481 vs -0.0476 unregularized | fixes conditioning, not mismatch |
| Worst-case robust SENSE (Chandrasekaran / El Ghaoui-Lebret) | indistinguishable from loading | bounded uncertainty is the wrong model |
| MDL/AIC subspace order for ESPIRiT | `support_frac = 1.000`, worse than a tuned threshold | Wax-Kailath assumes equal noise eigenvalues; the calibration spectrum decays continuously, so there is no order to select |
| ESPIRiT vs the plain ratio estimator | -0.109 SSIM at R=4, 48 ACS | real steering vectors are 14x rougher than a birdcage model |
| g-gated SENSE / VarNet hybrid | never beats VarNet at any threshold | VarNet wins even where g is smallest |
| g-gated SENSE / zero-filled hybrid | premise refuted outright | zero-filling wins even where g < 2 |

The last two share one cause, in section 1.

---

## 4. Replication status (this is the strongest part)

* **Pruessmann SNR law**, 96-replica pseudo-replica: measured/predicted g ratios
  **1.0000, 0.9997, 0.9996**. Caught that white-noise SENSE under-predicts the
  noise cost by 23% on a correlated array; `Psi` now runs through both the solve
  and the g-factor.
* **Griswold convergence**: error *rose* with kernel size, which exposed a real
  defect -- the apply step never filled the k-space border. Fixed: noiseless
  error 5.5e-2 -> 9e-3, now decreasing in both kernel size and ACS. Every GRAPPA
  number was regenerated; the defect had been penalising GRAPPA.
* Both papers' central figures reproduced (`make_paper_figures.py`).

---

## 5. Open leads, in the order I would take them

1. **Sampling-pattern design.** The natural array-geometry extension and
   completely untouched. Uniform undersampling is forced by Cartesian SENSE;
   CAIPIRINHA-style shifts and non-uniform patterns change *which* pixels alias
   together, which is a steering-vector separation problem. Needs CG-SENSE for
   non-uniform patterns -- a real build, not a patch.
2. **A second array geometry.** Everything here is one 20-channel head array.
   The compression crossover and the coil-selection result are claims *about
   array geometry* and should be shown on another. fastMRI knee is the obvious
   choice; signed URLs were issued 2026-08-28 and expire in 90 days, and
   `datasets/fastmri/fetch_subset.sh` takes the URL as `$1`.
3. **Exhaustive selection past K=5.** C(20,6) = 38760 is still tractable and
   would say whether "criterion beats search" holds as K grows.
4. **Why pre-whitening buys so little.** Section 2.1's open question.
5. **Non-Cartesian / sparse-array analogues -- worked out 2026-10-06, and it
   splits in two: negative for the coils, live for the lines.**

   **Do not build the coil version.** Minimum-redundancy, nested and coprime
   designs pay off through the difference co-array, and that object exists only
   when the array manifold is shift-invariant: far-field narrowband makes each
   element's response a pure phase ramp in position, so a covariance entry
   depends on sensor *separation* alone and lags become the currency. Receive
   sensitivities are near-field -- smooth complex maps with magnitude falloff
   and no phase ramp in position -- so no covariance entry is a function of a
   lag and there is no co-array to thin. The snapshot objection recorded here
   before (one snapshot per aliased pixel set; the R aliasing partners are the
   deterministic unknowns, not uncorrelated emitters) is also true, but it is
   the weaker of the two: unlimited snapshots would not create the structure.
   Sections 2.3 and 2.5 agree from the other side -- what governs the classical
   arms is the conditioning of `S^H Psi^-1 S`, not the extent of an aperture --
   so thinning coils to spread aperture optimises a quantity that does not
   control the answer.

   **The import that does work is on the lines, and it is item 1.** k-space
   samples are Fourier coefficients, so a lag structure genuinely exists there,
   and GRAPPA is already a lag-domain estimator: `grappa_recon`
   (`src/recon.py:228`) takes sources at offsets `(arange(kh) - (kh//2-1)) * R`,
   predicts targets at `dy = 1..R-1`, and shrinks `kh` until the ACS block can
   calibrate it (>= 8 rows, span `3R`). The fittable kernel geometries are
   therefore exactly the lag coverage of the acquired set, i.e. its difference
   set. A uniform every-Rth set is maximally *redundant* in that sense: every
   available lag is a multiple of R, and the contiguous ACS block is the only
   source of short lags. That is the one-sentence account of the achievable-ACS
   result in `HANDOFF_2026-09-18.md` (GRAPPA harmful at every budget below 64
   lines in every reader, non-monotone through kernel-height auto-shrink) --
   the shrink *is* the kernel losing lags. A minimum-redundancy *line* set,
   short lags covered contiguously plus long lags for aperture without spending
   64 contiguous lines, is a real design question and the first principled thing
   to try under item 1.

   Scope, before anyone starts: it needs GRAPPA kernel fitting on an arbitrary
   lag set, which is a build and not a patch; it re-excludes VarNet and the
   U-Net, which need a low-frequency anchor; and CAIPIRINHA plus the
   learned-mask literature already occupy "design a better pattern", so the
   co-array framing is the novelty rather than the goal. Out of scope for
   `paper_miccai.tex` by `PAPER_PLAN_MICCAI.md`.

---

## 5b. Streaming QC: step 1 run, and it closes the reservoir question

The idea was to predict reconstruction quality *during* a scan, for abort or
extend decisions, with a reservoir over the k-space stream. The warning recorded
here was that the baseline is not "nothing" but the static predictor applied to a
prefix. That test has now been run (`run_prefix_quality.py`, 60 volumes, two
acquisition orders, prefixes at 25/50/75/100%), predicting the SSIM of the
**completed** reconstruction from a partial acquisition.

| predicting final SSIM, R=8 | 25% | 50% | 75% | 100% |
|---|---|---|---|---|
| SENSE, centre-out | 0.895 | 0.913 | 0.913 | 0.913 |
| SENSE, sequential | n/a | 0.738 | 0.913 | 0.913 |
| GRAPPA, centre-out | 0.939 | 0.900 | 0.900 | 0.900 |
| Zero-filled, sequential | n/a | 0.466 | 0.894 | 0.894 |

**Early prediction works, and the reason is that the features are calibration-
derived.** Mean ACS lines acquired: centre-out reaches 19.5 of 24 by 25% and 24
by 50%; sequential has **zero** at 25% and 11.5 at 50%. Once the ACS block is
complete the features stop changing -- `g_mean`, `sens_rough`, `pc_resid` and the
rest are bit-identical at 50%, 75% and 100%, because every one of them is
computed from the estimated sensitivities and the calibration block, not from the
peripheral lines.

So the actionable statement is about scan ordering rather than about modelling:
**with a centric acquisition the abort decision is available after roughly a
quarter of the scan, at no cost in accuracy; with sequential ordering no
prediction is possible until 75%.**

**And this closes the reservoir question negatively.** A reservoir integrates a
sequence. Here the sequence is constant after the first quarter, so there is no
trajectory to integrate and recurrence has nothing to contribute. The streaming
idea reduces to "wait for the ACS block, then apply the static predictor", which
is a model that already exists.

> **Parked as a companion paper (2026-10-11): `COMPANION_ACQUISITION_PAPER.md`.**
> The ordering result is the keeper, and the one experiment owed is to redo the
> prefix test with a **task** endpoint rather than SSIM -- `paper_miccai.tex`
> proves SSIM does not order task damage, so predicting final SSIM argues
> against its own sibling.

**The honest boundary on that conclusion.** This is a property of *these*
features, not of k-space. They were built to be calibration-derived, so it is
close to a tautology that they freeze when calibration completes. A feature set
that read the peripheral lines -- outer-k-space energy, a running reconstruction,
line-to-line consistency in the high frequencies -- would genuinely evolve, and a
reservoir over *those* is not ruled out by this experiment. It is simply not
motivated by anything measured so far, and designing such features is the work
that would have to come first.

## 6. Reproducing

```bash
cd parallel_mri
# correctness first
python validate_snr_law.py && python test_espirit.py
# the array-processing results, in the order of section 2
python run_noise_cov.py            --out results/noise_cov.csv
python run_coil_select.py          --out results/coil_select.csv
python run_coil_select.py --R 2 4 5 8 10 --K 8 10 --out results/coil_select_Rsweep.csv
python run_coil_exhaustive.py --K 5 --R 4 --n-vols 4
python run_aperture_fixed.py
python run_model_order.py
# the negative result in section 1
python run_effective_g.py --n-vols 20 --n-slices 3 --R 2 4 8 --n-rep 64 --alpha 0.1
python analyze_effective_g.py
```

**Environment:** a conda Python env (`neurofusion` on the lab desktop). sklearn,
matplotlib and monai live there, not in base.

**Gotchas that cost time:**

* `one_slice` in `run_fastmri.py` takes k-space with the **readout oversampling
  still present** and removes it internally. It did not always; scripts written
  against the old contract (`run_aperture_fixed.py` was one) silently computed
  metrics over a 640-wide FOV that is half empty, depressing every number.
* `estimate_maps_acs` zeroes the sensitivities outside the object, so any
  regulariser scaled by `trace(G)` is zero exactly where G is singular. Use an
  absolute floor from the slice's global scale.
* Greedy selection has no well-defined objective below `|subset| = R`. Score
  partial subsets at `R_eff = |subset|`, not with a placeholder.
* GPU: two concurrent processes holding the 496 M-parameter U-Net plus a
  segmentation network is the practical limit on 16 GB. Four OOMs.
* Several scripts take a `--root`; `../datasets/fastmri/multicoil_balanced` is
  the contrast-balanced 21-volume set, `multicoil_val` the original 20.

## 7. Data and files

| Path | What |
|---|---|
| `src/recon.py` | SENSE (+Psi, loading, worst-case robust), GRAPPA, ESPIRiT, `mdl_aic_order` |
| `src/array_cov.py` | Ledoit-Wolf and OAS shrinkage for complex Hermitian covariance |
| `src/coil_select.py` | `greedy_g`, `greedy_snr`, `greedy_snr_law`, `svd_compress`, `gmap_for` |
| `src/hybrid_recon.py` | the g-gated hybrid that does not work; kept as the record |
| `src/stats.py` | the one bootstrap; resamples volumes/cases, never slices |
| `results/effective_g_unet.csv` | the noise/error correlations of section 1 |
| `results/noise_cov.csv`, `coil_select*.csv`, `coil_exhaustive.csv` | section 2 |
| `results/fastmri_aperture_fixed.csv` | section 2.5 |
