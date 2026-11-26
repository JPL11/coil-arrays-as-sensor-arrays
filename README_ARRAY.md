# Receiver-Coil Arrays as Sensor Arrays: SENSE and GRAPPA as Array Processing

SENSE and GRAPPA are implemented **from the original
papers**, not called from a library, and studied as *array-processing* methods:
the receiver-coil array is a sensor array and the coil-sensitivity map is its
steering vector.

Required reading (both journal papers):
1. Pruessmann et al., "SENSE: Sensitivity Encoding for Fast MRI," *Magn. Reson. Med.* 42(5):952-962, 1999.
2. Griswold et al., "Generalized Autocalibrating Partially Parallel Acquisitions (GRAPPA)," *Magn. Reson. Med.* 47(6):1202-1210, 2002.

This file covers the array half: implementation, replication, what breaks each
method, the design results the framing makes available, and a no-reference
quality predictor. **The downstream clinical study -- whether these failures reach
tumour delineation, the learned reconstructors, and the on-device QC agent --
is a separate study and is not part of this repository.** State, dead ends and open leads for this half
are in [`ARRAY_SIGNAL_PROCESSING_HANDOFF.md`](ARRAY_SIGNAL_PROCESSING_HANDOFF.md).

### The array half in five results

1. **Replication succeeds and found a real defect.** Pruessmann's SNR law is
   confirmed to 0.03%; Griswold's convergence claim initially *failed* and
   exposed an unfilled k-space border in this implementation of GRAPPA.
2. **SENSE and GRAPPA fail in complementary ways**, and the framing predicts
   which: mismatch destroys SENSE and leaves GRAPPA flat; calibration starvation
   reverses it; aperture separates them again.
3. **Three canonical repairs do not work** -- diagonal loading, worst-case robust
   beamforming, MDL/AIC model order selection -- each for a measured reason.
4. **The g-factor governs SENSE's noise (r = 0.973) and not its error
   (r = 0.197)**, and localises total error exactly in proportion to how much of
   the error is variance. Two methods built on the g-map failed for this reason.
5. **Reconstruction quality is predictable from the acquisition alone**, without
   a reference, and -- with a centric ordering -- from the first quarter of the scan.


## What is and is not claimed

The two required papers introduced SENSE and GRAPPA. This study implements them
from those papers and studies where they fail; it does not propose a competing
reconstructor. The one *reconstruction* method the findings here motivated -- a hybrid gated on
the analytic g-map -- **failed**, twice, and the reason was then measured. What
the failure produced instead is a no-reference quality predictor (see "A method:
predicting reconstruction quality without a reference"), which estimates how far
a reconstruction can be trusted without ever seeing a reference -- a different
kind of method, and not one that competes with a reconstructor. Nothing below should be read as "better than Pruessmann" or "better
than Griswold", and the QC agent in particular is not a reconstruction method:
it runs on the segmenter's output, downstream of whatever produced the image.

Three claims, in the order they are earned:

1. **Replication.** SENSE and GRAPPA are reproduced from the original papers and
   their central quantitative claims are confirmed -- the SNR law to within
   0.03%. Doing so exposed a real defect in this implementation of GRAPPA that
   only the papers' own convergence claim could have caught, and a 23%
   under-prediction of noise cost from assuming white coil noise.
2. **The failure reaches the clinical task, and image quality does not predict
   it.** At R=8 GRAPPA is significantly worse than not reconstructing at all
   (-0.102 Dice WT [-0.149, -0.065]) and SENSE inflates the median tumour volume
   by 200% in 97% of cases. VarNet buys 0.124 SSIM over zero-filling and at most
   0.005 Dice. Neither original paper could have made this statement; the
   downstream task did not exist for them, and the contribution is going past
   them rather than competing with them.
3. **The array-processing account is half right, and the half that fails is the
   half everyone would use.** The g-factor governs SENSE's *noise* (r = 0.973)
   and not its *error* (r = 0.197). Two methods built on the assumption that it
   governs both did not work.

The QC agent's baseline is confidence-based quality control, not SENSE or
GRAPPA. Comparing a failure detector against a reconstructor would be a category
error regardless of how the numbers came out.

## Verification status

Two different things are checked, and the distinction matters. **Internal
consistency** confirms the code does what the code intends. **Replication**
confirms the implementation reproduces a quantitative claim the original papers
actually make -- which internal consistency cannot do, and which is what caught
the GRAPPA edge-fill defect below.

### Internal consistency

| Check | Result |
|---|---|
| Forward model round-trip | 5.5e-16 relative error |
| SENSE, exact maps, R=2 | 6.3e-16 relative error |
| SENSE, exact maps, R=4 | 2.1e-14 relative error |
| Aliasing index mapping | derived empirically from a point source, not assumed |
| fastMRI loader vs dataset's own `reconstruction_rss` | 2.7e-08, all 20 volumes (complex64 storage limit) |
| Readout-oversampling removal | 1.4e-16; SENSE output unchanged |
| Vectorized GRAPPA vs original loop | bit-exact (max diff 0) |
| ESPIRiT Parseval (all singular vectors -> Gram = I) | 2.9e-15 |
| ESPIRiT maps vs known birdcage sensitivities | alignment 1.0000 (chance 0.354) |
| `sense_robust(eps=0)` vs `sense_recon` | exact |
| VarNet checkpoint load (`strict=True`, arch read from weights) | 324/324 tensors |
| bias^2 + variance = MSE over 64 replicas, every method | closes to 1e-16 |
| measured vs analytic g at R=2, 20 volumes (magnitude pseudo-replica) | 1.083 vs 1.100 |

### Replication (`validate_snr_law.py`)

**Pruessmann 1999, SNR_R = SNR_full / (g sqrt(R)).** The paper's central
quantitative claim: the analytically computed g-factor must predict the noise
the reconstruction actually produces. Checked by pseudo-replica Monte Carlo (96
independent noise realizations through the same reconstructor, per-pixel
standard deviation measured and compared against g*sqrt(R)).

| Condition | g predicted | g measured | ratio |
|---|---|---|---|
| white noise, R=2 | 1.0195 | 1.0208 | **1.0000** |
| white noise, R=4 | 3.6835 | 3.6927 | **0.9997** |
| correlated Psi (max off-diag 0.51), R=4 | 4.5388 | 4.5495 | **0.9996** |

Correlated coil noise raises the true penalty from g=3.68 to **4.54** at R=4.
The white-noise version of `sense_recon` -- the only one implemented at first --
would have under-predicted the noise cost of acceleration by 23% on an array
like the real one measured here, so `Psi` is now carried through both the solve
and the g-factor as in the paper.

**Griswold 2002, convergence.** GRAPPA is a finite-support least-squares
predictor, so it is *not* exact even noiselessly and the paper claims no such
thing; what must hold is that the residual shrinks as the kernel and the
calibration region grow. It did the opposite -- error *rose* from 2.3e-2 to
5.5e-2 as the kernel grew -- which exposed a real defect: the apply step never
filled the first and last `hw` readout columns, nor the phase-encode lines whose
kernel footprint fell outside the array, so a border of k-space was left at zero
and the border got wider with the kernel. With k-space padded before the kernel
is applied, noiseless error drops to ~9e-3 and now decreases with both kernel
size and ACS budget. Every GRAPPA number in this study was regenerated after the
fix; because the defect penalized GRAPPA, correcting it strengthens rather than
reverses the SENSE/GRAPPA comparisons below.

## Recreated figures from the required papers

`make_paper_figures.py` reproduces the central figures of both papers using this
implementation. They reproduce each figure's *structure and claim*, not the
original hardware -- Pruessmann used a 5-element cardiac array and Griswold a
phantom on a Siemens system, neither of which is available here -- so each panel
states what it actually shows.

| Figure | Paper | What it shows |
|---|---|---|
| `paper_P0_aliasing.png` | Pruessmann Figs. 1-2 | why undersampling aliases and how sensitivities undo it: three pixels FOV/3 apart fold onto one, each coil weights them differently, so 8 coils give 8 equations in 3 unknowns |
| `paper_P1_gfactor_maps.png` | Pruessmann Fig. 5 | g-factor maps at R = 2, 3, 4, 6. Mean g runs 1.02 -> 1.50 -> 3.60 -> 6.62, max 1.1 -> 18.2, with the characteristic banding at multiples of ny/R |
| `paper_P2_sense_vs_R.png` | Pruessmann Fig. 6/7 | SENSE reconstructions and 5x difference images vs R. Artifact power 1.1e-4 -> 3.7e-4 -> 1.6e-2 -> 1.4e-1; the difference images show the same banding the g-maps predict |
| `paper_P3_gfactor_validation.png` | (the claim behind Fig. 5) | measured vs predicted g, per pixel, from 64 noise replicas -- the scatter lies on the identity line from g = 1 to g = 9 |
| `paper_G1_kernel.png` | Griswold Fig. 1 | kernel geometry: source block of kh acquired lines x kw readout points, from every coil, predicting one missing line |
| `paper_G2_blocks_acs.png` | Griswold Fig. 5/6 | artifact power vs number of blocks and ACS lines at R = 4, with difference images |

Two things worth reading off them rather than asserting:

* **P1 and P2 agree.** The spatial pattern of the difference images in P2 is the
  banding of the g-maps in P1 -- the analytic geometry factor is not a summary
  statistic, it localizes where the reconstruction will be noisy.
* **G2 is non-monotonic in kernel size at small ACS.** More blocks is better
  only once the calibration region can support them: at 16-24 ACS lines two
  blocks beats four, and the ordering reverses above ~40 lines. Griswold's
  recommendation to use more blocks carries a calibration-budget precondition
  that is easy to miss.

The originals use a rainbow colour scale for the g-maps; these use `magma`,
which is perceptually uniform, because a rainbow scale misrepresents magnitude.

## Findings

**SENSE and GRAPPA fail in complementary ways, and the reason is structural.**
SENSE inverts an explicit array model, so it inherits that model's fragility.
GRAPPA never forms sensitivities at all, so a whole class of error cannot reach
it — but it needs enough calibration data to exist.

| Condition (SSIM) | SENSE | SENSE+DL | GRAPPA |
|---|---|---|---|
| Benign (R=4, 8 coils) | **0.925** | 0.910 | 0.916 |
| 40% sensitivity-map error | 0.682 | 0.710 | **0.919** |
| 4 coils (R = n_coils) | 0.648 | **0.821** | 0.803 |
| 4 of 8 coils dead | 0.663 | 0.748 | **0.752** |
| R = 8 | 0.470 | **0.698** | 0.676 |
| 12 calibration lines | 0.698 | 0.758 | **0.877** |
| 48 calibration lines | **0.975** | 0.958 | 0.950 |

1. **GRAPPA is structurally immune to steering-vector mismatch.** Across a 0→40%
   sensitivity-map error it moves 0.916→0.919 (flat), while SENSE loses 0.24
   SSIM. SENSE's *variance* also grows 40× (±0.001→±0.043): under mismatch it
   becomes not merely worse but unpredictable.
2. **Diagonal loading fixes conditioning, not mismatch.** The classic robust-
   beamforming remedy recovers +0.17 SSIM on a 4-coil array and +0.23 at R=8,
   but only +0.03 under map error. Ill-conditioning and model mismatch are
   different failures and need different remedies.
3. **Loading is not free.** On a healthy array the optimum is λ≈1e-3
   (0.930 vs 0.925 unloaded); λ=0.3 costs 0.16 SSIM. Robustness trades against
   benign-case accuracy.
4. **Calibration budget crosses over at ~24 lines.** Below it GRAPPA wins by a
   wide margin (0.877 vs 0.698 at 12 lines); above it SENSE does (0.975 vs
   0.950 at 48). At 8 lines GRAPPA cannot calibrate *any* kernel and fails
   outright — a hard floor rather than graceful degradation.
5. **g-factor predicts the SENSE collapse.** Max g in the body runs
   1.5 → 10 → 39 → 2.0e4 for R = 2 → 4 → 5 → 8, and SENSE's NMSE tracks it.

### Methodological note
The calibration-budget sweep holds the **sampling pattern fixed** (R=4, 48 ACS
lines acquired, R_eff = 2.29 for every row) and varies only how many central
lines the methods may calibrate on. A naive ACS sweep confounds calibration
budget with effective acceleration, because adding ACS lines also samples more
k-space; `results/all_experiments.csv` (sweep `acs`) contains that confounded
version for comparison, and `results/calibration_budget.csv` the corrected one.

## Real-data validation (fastMRI multicoil brain)

20 fully-sampled volumes / 60 slices from `brain_multicoil_val`, 4-20 receiver
coils, Siemens 1.5T/3T. `recon.py` is imported unchanged; only the data and the
evaluation protocol differ. All intervals are 95% cluster bootstrap (5000
resamples of whole *volumes*), contrasts paired on the same slice so anatomy
variation cancels. See `analyze_stats.py`.

### Which simulated findings survive

| # | Claim | Verdict | Real-data evidence |
|---|---|---|---|
| 1 | GRAPPA immune to steering-vector mismatch | **survives, restated** | over 0->40% map error, paired: GRAPPA -0.0008 [-0.0013, -0.0005], SENSE -0.0454 [-0.0538, -0.0371]. GRAPPA is not *exactly* flat -- with a tight body mask its interval excludes zero -- but it degrades **57x less** |
| 1b | SENSE *variance* grows 40x under mismatch | **fails** | the simulation had zero anatomy variance, so mismatch was its only noise source; on real data slice-to-slice variation dominates and the ratio is ~1 |
| 2 | Loading fixes conditioning, not mismatch | **survives** | under map error SENSE -0.0454 vs SENSE+DL -0.0498: loading buys nothing against mismatch, and neither does worst-case robust SENSE (below) |
| 3 | Loading is not free | **reverses** | optimum lam=0.1 gains **+0.038** over lam=0 (0.9121 vs 0.8741; simulation: +0.005), and lam=0.3 costs only 0.013. Real data is genuinely ill-conditioned, so loading is worth more and costs less |
| 4 | Calibration crossover at ~24 ACS lines | **fails** | GRAPPA wins at every budget 12->128, all intervals excluding zero. The gap is U-shaped -- +0.096 at 12 lines, a minimum of +0.016 at 48, then *widening* to +0.036 at 128 -- so SENSE does not merely fail to overtake, it falls further behind as calibration grows |
| 5 | g-factor predicts the SENSE collapse | **survives** | g_mean 1.10 -> 4.09 -> 11.9 -> 24.8 for R=2,4,5,8; SENSE SSIM 0.975 -> 0.874 -> 0.781 -> 0.660. Independently validated against measured noise (see Replication) |

### Acceleration (SSIM, body-masked, vs fully-sampled RSS)

| R | Zero-filled | SENSE | SENSE+DL | GRAPPA | VarNet | U-Net |
|---|---|---|---|---|---|---|
| 2 | 0.937 | 0.975 | 0.975 | 0.975 | **0.989** | 0.919 |
| 4 | 0.874 | 0.874 | 0.902 | 0.904 | **0.982** | 0.873 |
| 5 | 0.862 | 0.781 | 0.858 | 0.849 | **0.979** | 0.862 |
| 8 | 0.844 | 0.660 | 0.771 | 0.770 | **0.968** | 0.845 |
| 10 | 0.833 | 0.632 | 0.742 | 0.717 | **0.959** | 0.834 |

The two learned columns are not interchangeable. **U-Net tracks zero-filling to
within 0.002 SSIM at every acceleration from R=4 up** -- it is an image-domain
post-processor applied to the zero-filled image and it does not, under a
body-masked metric, improve on its own input inside the head. VarNet, which
enforces data consistency in k-space, is 0.12 above both at R=8. Adding the
second model left every other number bit-identical (10695 rows, max difference
0.00e+00).

The classical columns are unchanged from the pre-VarNet run -- 8556 rows
matched, max difference 2.3e-14 -- which is the regression check on the
refactor that threaded the full-readout k-space through `one_slice`. VarNet
loses **0.031** SSIM from R=2 to R=10; SENSE loses 0.343. Read the learned
column with the contrast caveat below: 48 of these 60 slices are AXT2.

### What only real data shows

1. **Real steering vectors are 14x rougher than a birdcage model.** Relative
   spatial gradient of |S_0|: 0.189 measured vs 0.0137 simulated. This single
   fact explains finding 4's reversal: the low-resolution ratio estimator cannot
   resolve them, and SENSE never recovers what GRAPPA never needed.
2. **Coil noise is correlated** (off-diagonal correlation up to 0.54). Carrying
   Psi through the solve and the g-factor is what the paper specifies, and it
   matters: at R=4 the true noise penalty is g=4.54 rather than the 3.68 a
   white-noise model predicts. As a *preprocessing* step, pre-whitening is a
   small but real gain for SENSE (+0.0046 [+0.0012, +0.0100]) and marginally
   negative for GRAPPA (-0.0014 [-0.0021, -0.0007]).
3. **The Hann taper on ACS map estimation helps SENSE and is neutral for
   GRAPPA**: removing it costs SENSE -0.0359 [-0.0550, -0.0172] and moves GRAPPA
   +0.0030 [-0.0002, +0.0091]. It was an unconditional default in the simulation.
4. **Zero-filling "wins" on SSIM at R >= 5** (0.844 vs GRAPPA 0.770 at R=8). This
   is a metric artifact -- zero-filling returns a low-resolution image dominated
   by the ACS block, and SSIM rewards blur over aliasing. It is the clearest
   argument for the downstream task metric below, where it also wins, but for a
   reason that is not an artifact.

### Three remedies that do not work

**Worst-case robust SENSE** (`sense_robust`, from Chandrasekaran et al. 1998 /
El Ghaoui & Lebret 1997) solves the mismatch-targeted problem
`min_rho max_{||dS|| <= eps||S||} ||(S+dS) rho - a||`, making the loading
per-pixel and data-adaptive instead of one global lambda. Over a 0->40% map
error at R=4 it delivers **-0.0476 against diagonal loading's -0.0481** --
indistinguishable -- while over-regularizing (eps=0.3) is worse than doing
nothing. Its only real benefit is variance reduction (+/-0.034 vs +/-0.045).

**ESPIRiT** (`espirit_maps`, Uecker et al. 2014), the subspace steering-vector
estimator, loses to the plain low-resolution ratio at every setting tested:
-0.109 [-0.138, -0.079] SSIM at R=4, 48 ACS. Parameterization is not the
explanation -- a 48-point sweep over kernel size, subspace threshold and
eigenvalue crop improves it by +0.036 (R=4) and +0.089 (R=8) but does not close
the gap, and the optimum is a plateau in kernel size (k=8, 10, 12 all give
~0.810 against the ratio's 0.871), not a grid edge. Nor is it a metric artifact:
the unfolding-only score gives the same answer as the absolute one (-0.112 vs
-0.109), and at good settings the ESPIRiT maps align 0.98 with the ratio maps,
so they are sensible sensitivities. It also costs 32 s/slice against 0.05 s.

**Model order selection** (`mdl_aic_order`, Wax & Kailath 1985; run by
`run_model_order.py` on 4 volumes at 48 ACS) is the obvious objection to the
ESPIRiT result: the subspace size came from a hand-picked singular-value
threshold, the 48-point sweep swings 0.14 SSIM across plausible settings, and
that threshold is therefore tuned on the very data it is scored on. MDL and AIC
answer the question properly -- choose the signal/noise split by which one best
explains the eigenspectrum -- and neither helps. Against the *best* threshold
the sweep found (c=0.1): MDL 0.813 / 0.539 and AIC 0.840 / 0.560 at R = 4 / 8,
against the threshold's 0.806 / 0.544. All three still lose to the ratio
estimator's 0.871 / 0.663, and they cost 173 s and 337 s per slice against
0.08 s.

AIC scoring highest is not a better estimate, and the diagnostic that says so is
in the same table. Both criteria report `support_frac = 1.000`: the per-pixel
eigenvalue clears the 0.9 crop **everywhere**, background included. Keeping 205
of 612 kernels (MDL) or 521 of 612 (AIC) drives the per-pixel Gram towards the
identity -- the Parseval case `test_espirit.py` checks -- where every eigenvalue
is 1 and the object support ESPIRiT exists to recover has dissolved. The tuned
threshold keeps 4% of its kernels and reports 0.389, a plausible head fraction.
The failure is structural rather than a bad implementation: Wax-Kailath assumes
the noise eigenvalues are equal in expectation, and a calibration matrix built
from overlapping patches of smooth sensitivities has no such split -- its
spectrum decays continuously, so there is no order there to select.

Together these matter more than any one alone. Finding 1 invites the objection
"SENSE just needs a better estimator or a better regularizer." Three canonical
answers from the array-processing literature -- worst-case robust beamforming, a
subspace steering-vector estimator, and model order selection for its one free
parameter -- were implemented and tested, and none recovers more than ~0.005
SSIM. The problem is not ill-conditioning, not bounded uncertainty, not
steering-vector resolution, and not a tuned threshold: it is that SENSE inverts
an explicit array model at all. The learned reconstructor below is the control
that shows this is a property of *explicit model inversion* rather than of the
measurement.



### Which coils? Sensor selection using the paper's own SNR law

The aperture sweep answers *how many* coils and says nothing about *which*.
That second question is the sensor-selection problem from array processing, and
in MRI it is not the same question as coil compression: compression forms K
linear combinations of N channels and needs all N receivers present, while
selection keeps K physical elements, which is what applies when channels are
limited by hardware or when one has failed. 7 volumes with 20 coils, 3 slices
each, intervals paired over the same slices and bootstrapped over volumes.

Three objectives, each optimising a different part of the validated law
`SNR_R = SNR_full / (g sqrt(R))`:

| SSIM, R=4 | random | greedy-g | greedy-snr | greedy-snr-law | svd-compress |
|---|---|---|---|---|---|
| K=5 | 0.706 | 0.748 | 0.755 | **0.764** | 0.791 |
| K=6 | 0.711 | 0.767 | 0.772 | **0.782** | 0.797 |
| K=8 | 0.767 | 0.787 | 0.801 | 0.800 | 0.811 |
| K=10 | 0.781 | 0.803 | 0.813 | 0.810 | 0.821 |
| K=12 | 0.804 | 0.815 | 0.821 | 0.818 | 0.827 |

**Minimising the g-factor is the wrong objective, and the experiment shows it
twice.** `greedy-g` achieves the lowest mean analytic g at every K -- 2.07
against 2.16 and 2.28 for the other rules at K=5, so the optimiser works -- and
still loses on image quality. The reason is the law itself: g is only the noise
*penalty*, and selecting on it alone discards the `SNR_full` factor, which picks
well-conditioned but weakly-receiving elements. `greedy-g` does beat the random
null everywhere (+0.042 [+0.031, +0.056] at K=5), so geometry carries real
information; it is simply not sufficient.

Using the whole law as the objective closes that gap, but only where
conditioning binds:

| greedy-snr-law - greedy-snr (paired) | R=4 | R=8 |
|---|---|---|
| K=5 | **+0.0096 [+0.0023, +0.0163]** | -- |
| K=6 | **+0.0100 [+0.0039, +0.0165]** | -- |
| K=8 | -0.0009 [-0.0039, +0.0020] | +0.0008 [-0.0023, +0.0038] |
| K=10 | -0.0023 [-0.0058, +0.0003] | -0.0009 [-0.0042, +0.0025] |
| K=12 | -0.0027 [-0.0095, +0.0012] | -0.0031 [-0.0101, +0.0013] |

The gain is real but narrow: about +0.010 SSIM when the array is thinned to
K <= 1.5R at R=4, and nothing once K is comfortably above R, where raw
sensitivity magnitude is all that is left to choose on. At R=8 it never appears
-- not even at K=8, where K/R = 1 and conditioning should bind hardest --
because the whole selector spread there is 0.007 SSIM against a 0.55 baseline.
Acceleration, not coil choice, is what limits that regime, and no selection
rule rescues it.

**Coil compression stops helping at high acceleration, and the crossover can be
located.** SVD compression is allowed to mix all 20 channels, so it answers a
strictly easier question than selection and should win outright. It does not.
Sweeping acceleration at fixed K (paired, bootstrapped over volumes):

| svd-compress - greedy-snr-law | R=2 | R=4 | R=5 | R=8 | R=10 |
|---|---|---|---|---|---|
| K=8 | **+0.0109** | **+0.0117** | +0.0043 | **-0.0031** | -- |
| K=10 | **+0.0092** | **+0.0109** | **+0.0059** | -0.0014 | **-0.0040** |

*(bold = 95% interval excludes zero.)*

The advantage decays monotonically with acceleration and changes sign: for K=8
it is worth +0.012 SSIM at R=4, is indistinguishable from zero at R=5, and is
significantly **negative** by R=8. For K=10 the crossover sits later, between
R=8 and R=10 -- more retained components means less diversity given away, which
is the direction the mechanism predicts.

Compression trades spatial diversity for SNR, and unfolding an aliased pixel
needs the diversity more than the SNR as the fold count grows. The practical
reading is that the standard advice to compress before reconstructing is
acceleration-dependent, and that at high R it is better to keep fewer *physical*
elements chosen well than more virtual ones mixed from all of them.


**Is greedy the problem, or is the objective?** `greedy-g` attains the lowest
mean g of any rule tested and still loses on image quality, which admits two
readings: the search is too weak, or the criterion is wrong. C(20,5) = 15504 is
enumerable in ~5 minutes per slice, so both objectives were evaluated
exhaustively on 4 volumes at K=5, R=4:

| | mean g achieved | SSIM |
|---|---|---|
| greedy-g | 2.045 | 0.5794 |
| **exhaustive best-g** | **1.842** | **0.5282** |
| greedy-snr | -- | 0.5790 |
| greedy-snr-law | -- | 0.5877 |
| **exhaustive best-SNR-law** | -- | **0.5895** |

**Searching harder for a lower g makes the reconstruction worse.** The greedy
rule is genuinely a poor optimiser of g -- it is 11% above the true minimum
(2.045 vs 1.842) and never once finds the optimal subset (0/4) -- and the subset
that actually minimises g scores **0.051 SSIM below** the one greedy settles
for. The failure is therefore in the criterion, not the search, and greedy's
weakness was accidentally protecting it: an imperfect minimiser of g stays
closer to subsets that still receive signal.

For the right criterion the picture inverts. Greedy on the SNR law lands within
**0.0018 SSIM** of its own exhaustive optimum (0.5877 vs 0.5895) and finds the
optimal subset in 1 of 4 cases. So the criterion carries essentially all of the
difficulty here and the search carries almost none -- which is the useful
practical statement, since exhaustive enumeration is only tractable at the very
smallest K and greedy is what would be used in practice.


### A method: predicting reconstruction quality without a reference

Every earlier attempt to *use* the g-factor failed, and the reason was that it
predicts SENSE's noise and not its error. That conclusion needs one refinement,
and the refinement is the method.

Two different questions were being conflated:

* **Where** is the error, within an image? g does not know: per pixel,
  `corr(|bias|, g) = 0.197`. This is why both g-gated hybrids failed -- a spatial
  gate needs the spatial answer.
* **How bad** will this reconstruction be, overall? That is a different
  question, it is the one a scanner actually needs answered, and g statistics
  answer it well.

`run_noref_quality.py` predicts per-slice SSIM from acquisition-side features
alone -- g distribution, array size, calibration budget, noise-correlation
off-diagonal, sensitivity roughness, ACS energy. **No feature touches the
reference**; every one is computable before there is anything to compare against.
161 volumes, 4347 rows, 5-fold cross-validation grouped by volume.

| cross-validated r | g_mean alone | all g features | **all features** | best single feature |
|---|---|---|---|---|
| SENSE, R=4 | 0.783 | 0.882 | **0.944** | g_frac_gt5 (0.785) |
| SENSE, R=8 | **0.244** | 0.683 | **0.821** | acs_energy (0.443) |
| GRAPPA, R=4 | 0.709 | 0.862 | **0.944** | g_frac_gt5 (0.730) |
| GRAPPA, R=8 | **0.226** | 0.708 | **0.830** | sens_rough (0.461) |
| Zero-filled, R=4 | 0.099 | 0.652 | **0.885** | sens_rough (0.623) |
| Zero-filled, R=8 | 0.021 | 0.711 | **0.807** | sens_rough (0.588) |

**At R=8, where `g_mean` collapses to 0.244, the model reaches 0.821** -- the
regime where the prediction is worth having is the regime the analytic quantity
abandons, and it is recovered.

Two things carry it, and both are interpretable.

**The g *distribution*, not its mean.** `g_frac_gt5` -- the fraction of the brain
above g = 5 -- is the strongest single feature at R = 4 and 5, and moving from
`g_mean` to the full g summary takes SENSE at R=8 from 0.244 to 0.683. The
earlier "g does not predict error" result was in part a failure of the *mean*:
severity lives in the tail.

**Sensitivity roughness reaches what g cannot.** `sens_rough` is the best single
feature for GRAPPA at R=8 and is nearly the entire model for zero-filling
(0.837 alone against 0.885 for all features), where `g_mean` gives 0.02.
Zero-filling inverts no array model, so it has no g-dependence at all -- but its
error still depends on how much structure the sensitivities carry, which is the
bias term. This is the same 14x-rougher-than-birdcage measurement from earlier
in this file, now doing predictive work.

Checked rather than assumed: permuting the target across volumes drops every
score to between -0.11 and +0.07, so nothing is leaking through the grouping;
removing `n_coils` changes nothing (0.944 -> 0.946), so the model is not merely
learning which array was used; and gradient boosting beats ridge everywhere, so
the relationship is genuinely nonlinear.

**Generalisation, measured rather than assumed.** Pooled across accelerations
with R as a feature -- the deployable configuration, since R is always known --
the model reaches r = 0.964 (SENSE), 0.952 (GRAPPA) and 0.885 (zero-filled) at
a mean absolute error of about 0.02 SSIM. It extrapolates well to an *adjacent*
acceleration (R=4 -> R=5: 0.942) and poorly across the noise-to-bias transition
(R=4 -> R=8: 0.449; GRAPPA R=8 -> R=4: -0.059), so training data is needed at the
acceleration where it will be deployed. That limit is real and is not hidden.

**What this is and is not.** It is a no-reference quality predictor -- an
acquisition-side estimate of how much a reconstruction can be trusted, available
before any reference exists. It is not a reconstructor and does not compete with
SENSE or GRAPPA. It is the piece that makes the negative results productive: the
g-map cannot tell you where to intervene, but the acquisition can tell you
whether to trust the result.



#### Predictive coding: the residual predicts the failure of the model that produced it

GRAPPA is a linear predictor of k-space, so it is a predictive-coding model, and
predictive coding's native quantity -- the prediction error -- is measurable here
with no reference at all. `src/self_consistency.py` fits the GRAPPA kernel on
half the ACS block and scores it on the half held out. Acquired lines predict
acquired lines; nothing is compared against a fully sampled image.

Adding those residuals to the acquisition features (cross-validated by volume,
161 volumes):

| cross-validated r | acquisition | pc only | **acquisition + pc** |
|---|---|---|---|
| SENSE, R=8 | 0.821 | 0.438 | **0.849** |
| GRAPPA, R=8 | 0.830 | 0.518 | **0.919** |
| Zero-filled, R=8 | 0.807 | **-0.008** | 0.834 |

**The residual predicts the failure of the method whose prediction it measures.**
It adds most to GRAPPA (0.830 -> 0.919), which is the method whose own kernel
the residual is computed from -- a self-diagnostic. It adds moderately to SENSE
(0.821 -> 0.849), which depends on the same calibration block through its
sensitivity estimate. And it carries **nothing at all** for zero-filling
(pc-only r = -0.008), which fits no kernel, uses no calibration and therefore
has no self-consistency to violate.

That null is the useful part. A feature that improved every method equally would
be suspected of proxying for image quality in general; one that is worth 0.09 to
GRAPPA, 0.03 to SENSE and exactly nothing to zero-filling is measuring what it
claims to measure.

Two further predictive-coding signals were tried. **Inter-coil predictability** --
predicting each coil's k-space from the others, which is predictive coding across
the array rather than across k-space -- reaches r = -0.63 against SENSE SSIM at
R=8 where `g_mean` reaches -0.41, and its correlation with `g_mean` is only
**+0.107**. Two quantities that both describe array diversity, arrived at by
completely different routes, turn out to be nearly orthogonal. A **multi-scale**
residual ratio (kernel 2x3 against 4x7) was the weakest of the three at -0.31 to
-0.39 and is not carried forward; the single-scale residual does the work.

The residuals do **not** improve cross-method transfer (0.944 -> 0.952 for
SENSE -> GRAPPA, 0.940 -> 0.940 reversed), which is consistent with them being
method-coupled by construction.

#### Against the image-domain baseline

The existing no-reference quality literature for MRI reads the *output image* --
BRISQUE, PIQE and the CNN variants all work from the reconstruction's
appearance. That is the baseline this has to be measured against rather than
merely distinguished from, so six image-domain features (gradient energy,
background noise, contrast, high-frequency ratio, entropy, local variance) are
computed from each reconstruction and given the same treatment.

**Within a known reconstructor the two are equivalent:**

| cross-validated r | image-only | acquisition-only | both |
|---|---|---|---|
| SENSE, R=4 | 0.942 | 0.944 | **0.968** |
| SENSE, R=8 | 0.827 | 0.821 | **0.875** |
| GRAPPA, R=8 | 0.880 | 0.830 | **0.890** |
| Zero-filled, R=8 | 0.816 | 0.807 | **0.879** |

Neither wins, and combining them helps everywhere -- so they are complementary
rather than redundant, and an honest paper cannot claim the acquisition side is
simply better at this task.

**Across reconstructors they are not equivalent at all:**

| train -> test, R=8 | image-only | acquisition-only |
|---|---|---|
| SENSE -> GRAPPA | **-0.220** | **+0.944** |
| GRAPPA -> SENSE | +0.572 | **+0.940** |
| SENSE -> Zero-filled | -0.015 | **+0.709** |

**Image-domain features do not transfer between reconstructors; one transfer is
outright negative.** Acquisition-side features transfer almost perfectly. The
reason is structural rather than incidental: image features describe how the
*output* looks, and SENSE's noise looks nothing like GRAPPA's blur, so a model
fitted on one is mis-specified for the other. Acquisition features describe the
*scan*, which is the same object whatever is later run on it.

The practical claim, at the width the data supports: acquisition-side features
predict reconstruction quality **consistently** across reconstructors and arrays
(0.86-0.98 everywhere tested), while image-based features do so
**inconsistently** (-0.17 to +0.77, varying with array size for reasons this
study does not resolve -- see "When does image-based transfer work?"). For a QA
method that has to be trusted without a reference, predictable behaviour is the
property that matters. This is weaker than the claim the mixed-brain result
alone suggested, and it is the one that survives stratification.



#### Predicting the finished scan from a partial one

Everything above scores a *completed* acquisition. The prospective question is
different and is what an on-device abort decision would need: given the first
X% of phase-encode lines, how good will the **finished** reconstruction be?
`run_prefix_quality.py` answers it on 60 volumes, at prefixes of 25/50/75/100%,
under two acquisition orders. The target is the SSIM of the full reconstruction,
not of the prefix -- predicting how good a half-acquired image currently looks is
a much easier and useless question.

| predicting final SSIM, R=8 | 25% | 50% | 75% | 100% |
|---|---|---|---|---|
| SENSE, centre-out | **0.895** | 0.913 | 0.913 | 0.913 |
| SENSE, sequential | **n/a** | 0.738 | 0.913 | 0.913 |
| GRAPPA, centre-out | **0.939** | 0.900 | 0.900 | 0.900 |
| GRAPPA, sequential | n/a | 0.783 | 0.900 | 0.900 |
| Zero-filled, centre-out | 0.896 | 0.894 | 0.894 | 0.894 |
| Zero-filled, sequential | **n/a** | 0.466 | 0.894 | 0.894 |

**Prediction is as good at a quarter of the scan as at the end, provided the
acquisition is centric.** Mean calibration lines acquired (of 24): centre-out
reaches 19.5 by 25% and all 24 by 50%; sequential has **zero** at 25% and 11.5 at
50%, so no prediction exists there at all and the 50% figure is made on a
half-calibrated block.

The mechanism is plain in the features rather than inferred: once the ACS block
is complete they **stop changing**. `g_mean`, `sens_rough`, `pc_resid` and the
rest are bit-identical at 50%, 75% and 100%, because every one is computed from
the estimated sensitivities and the calibration block and none of them reads the
peripheral lines. Quality is therefore decided, as far as these features can see,
by the time the centre of k-space is in.

The useful statement is about scan design rather than modelling: **a centric
ordering makes an abort decision available after roughly a quarter of the scan at
no cost in accuracy, and a sequential ordering makes it unavailable until 75%.**

It also settles, negatively, whether a recurrent model over the k-space stream
would help. A reservoir integrates a sequence; here the sequence is constant
after the first quarter, so there is no trajectory to integrate. The streaming
idea reduces to "wait for the calibration block, then apply the static
predictor". That conclusion is bounded, though, and the bound matters: these
features were *built* to be calibration-derived, so their freezing is close to
tautological. Features that read the peripheral lines -- outer-k-space energy, a
running reconstruction, high-frequency line-to-line consistency -- would evolve,
and nothing here rules out recurrence over those. Designing them is the work that
would have to come first.

#### Knee: what reproduces on a second anatomy and array

Every number above is from one array family. The method is a claim about
acquisition physics, so it has to be shown on another. 69 fastMRI knee volumes,
15 coils uniformly (against brain's 4-20), two contrasts in near-balance.

Getting them in at all took a fix worth recording: **all 69 were initially
rejected** as not fully sampled. Knee carries ~36 all-zero phase-encode lines at
the *edges* of a 368/372-line acquisition, and `load_slice` centre-crops to 320
before anything is reconstructed -- so the filter was testing data the pipeline
never touches. `is_fully_sampled(crop_pe=320)` tests the region actually used.
It is off by default and the brain pools are byte-identical either way
(20/49/94), so nothing above moved.

| | KNEE g_mean | KNEE acq | BRAIN g_mean | BRAIN acq |
|---|---|---|---|---|
| SENSE, R=4 | 0.606 | 0.893 | 0.783 | 0.946 |
| SENSE, R=8 | 0.477 | 0.658 | **0.244** | 0.813 |
| GRAPPA, R=8 | 0.579 | 0.734 | **0.226** | 0.836 |
| Zero-filled, R=8 | 0.553 | 0.822 | **0.021** | 0.820 |

**What reproduces.** The method works (acq 0.66-0.90), and the mechanism holds:
`sens_rough` beats `g_mean` at every acceleration and every method on knee --
0.822 against 0.606, 0.645 against 0.477, 0.831 against 0.683 -- exactly as on
brain. Sensitivity roughness carrying the bias term the geometry factor misses
is not an artefact of one array.

**What does not.** `g_mean` does *not* collapse at R=8 on knee (0.477-0.579)
the way it does on brain (0.021-0.244). The dramatic collapse that motivated
this whole line is stronger on brain than it is in general, and should be
reported as such.

**And the differentiator is narrower than the brain data implied.** Transfer at
R=8:

| train -> test | KNEE acq | KNEE img | BRAIN acq | BRAIN img |
|---|---|---|---|---|
| SENSE -> GRAPPA | **+0.984** | +0.772 | **+0.944** | **-0.220** |
| GRAPPA -> SENSE | **+0.982** | +0.591 | **+0.940** | +0.572 |

Acquisition features transfer better than image features on both anatomies, so
the claim survives. But image features do not *fail* on knee -- they reach 0.59
to 0.77 where on brain one transfer was outright negative. The likely reason is
homogeneity: knee is 15 coils and two contrasts throughout, so SENSE and GRAPPA
outputs resemble each other far more than they do across brain's 4-to-20-coil,
four-contrast population. The honest statement is that the advantage is a matter
of degree and grows with how heterogeneous the acquisition population is -- not
that image-based assessment cannot transfer at all.

Within a single reconstructor, image features are now *better* than acquisition
features on knee (0.943 against 0.893 at R=4), where on brain the two were
equal. The case for the acquisition side rests on transfer, and only on transfer.


#
### When does the g-factor localise error? Exactly as far as the error is noise

The result above -- g governs SENSE's noise (r = 0.973) and not its error
(r = 0.197) -- was measured at one operating point: the SNR and correlated body
noise of a room-temperature receiver array. Stating it absolutely was too strong.
`run_noise_regime.py` sweeps the noise scale over 300x and the covariance from
measured-correlated to diagonal (the sensor-noise-dominated case that SQUID and
NV-diamond detection for ultra-low-field MRI would occupy -- the regime is the
motivation; nothing quantum enters the array inverse, which sees only
sensitivities and a covariance).

| R=4 | variance share of error | corr(noise, g) | corr(&#124;bias&#124;, g) | **corr(total error, g)** |
|---|---|---|---|---|
| alpha = 0.1 (clinical SNR) | 0.009 | 0.893 | 0.246 | **0.255** |
| alpha = 1.0 | 0.461 | 0.892 | 0.245 | **0.540** |
| alpha = 4.0 | **0.830** | 0.855 | 0.220 | **0.815** |
| alpha = 12 | 0.647 | 0.839 | 0.171 | **0.645** |
| alpha = 30 | 0.519 | 0.875 | 0.069 | **0.483** |

**The g-factor localises total error exactly in proportion to how much of the
error is variance.** The last two columns track each other almost one to one --
0.830 against 0.815, 0.647 against 0.645, 0.519 against 0.483 -- which is what a
variance-weighted mixture of the noise and bias correlations should do, and is
not something either column was fitted to.

That turns two failures into a bounded positive claim. Both g-gated methods
failed because at clinical SNR **variance is under 1% of the error budget**, so
the g-map is predicting a term that barely exists. Push the noise up until
variance dominates and the g-map becomes genuinely predictive (0.815). The
condition for a g-gated method to work is therefore explicit and checkable
before building one: *the acquisition must be variance-dominated*, which a
clinical array is not.

Two further readings. **The covariance structure is nearly irrelevant** --
measured-correlated and diagonal give the same answer at every noise level
(0.815 vs 0.852 at alpha=4), so the correlation this array carries (off-diagonal
to 0.54) does not change what g explains. Given how much of this study is spent
on pre-whitening and shrinkage, that is worth stating plainly.

And the sweep is **not monotonic**: the variance share peaks at alpha=4 and falls
again by alpha=30 (0.830 -> 0.519), because magnitude images acquire a Rician
bias at very low SNR and the bias term grows back. The informative row is the
peak, not the endpoint -- there is no arbitrarily-noise-dominated regime to reach
in a magnitude reconstruction, which itself bounds how far the g-map can ever be
pushed.

At R=8 the same pattern holds at a lower ceiling (variance share peaks at 0.438,
corr(total, g) at 0.392), because the bias term is larger there to begin with.

### When does image-based transfer work? No clean answer

The knee result -- image features transferring at +0.77 where on brain they
managed -0.22 -- looked like an effect of population homogeneity, and brain
restricted to 20-coil volumes seemed to confirm it (+0.86). That explanation is
**wrong**, and splitting brain by coil count with anatomy held constant shows
why:

| brain, coils | volumes | SENSE -> GRAPPA, image | acquisition |
|---|---|---|---|
| 4 | 57 | **-0.115** | 0.864 |
| 12 | 11 | **-0.173** | 0.860 |
| 16 | 49 | +0.313 | 0.908 |
| 20 | 30 | **+0.766** | 0.980 |
| mixed (all) | 161 | -0.184 | 0.944 |

Homogeneous 4-coil and 12-coil subsets transfer no better than the mixed
population, so homogeneity is not the driver. And mixing coil counts does not
break transfer either: 16-and-20 together reaches +0.718, *better* than 16 alone
at +0.313, which is the opposite of what a homogeneity account predicts.

The pattern that remains is that image-based transfer improves with **array
size** -- and a plausible reading is that at 20 coils and R=8 both SENSE and
GRAPPA still work, so their outputs resemble each other, while at 4 coils R
approaches the coil count and SENSE is destroyed where GRAPPA is not, leaving
two images with nothing in common. But knee at 15 coils reaches +0.772 where
brain at 16 reaches +0.313, which that account does not explain.

**So no mechanism is claimed.** What is established is narrower and should be
stated as such: *acquisition-side features transfer across reconstructors
consistently (0.86-0.98 in every subset tested), and image-based features
transfer inconsistently, ranging from -0.17 to +0.77 depending on the array in
ways this study cannot explain.* Consistency is a real advantage for a QA method
and it is what the data supports. The stronger claim -- that image-based
assessment cannot transfer -- held only on the mixed brain population and does
not survive stratification.

This is the second explanation offered for this effect and the second one the
data removed. It is recorded rather than deleted because the pattern is real,
unexplained, and would otherwise be re-proposed.

### How many noise samples does pre-whitening need?

Pre-whitening is the step that makes the plain least-squares SENSE solve the
maximum-likelihood estimator, and it is the one step here that bought almost
nothing (+0.007 SSIM at R=4). The README previously blamed a poorly conditioned
sample covariance -- the classical small-sample failure that motivates shrinkage
in adaptive beamforming. **That explanation was wrong.** The default corner
region yields n = 1444 samples for nc = 20 coils, n/nc = 72, condition number
12.0, and a Ledoit-Wolf shrinkage intensity of 0.02: there is nothing to fix.

The useful question is where the sample estimator *does* break, since a clinical
noise pre-scan is short and the sample count is a design parameter. Sweeping the
corner fraction takes n from 1444 to 36 on the same 16 slices:

| n/nc | n | sample | Ledoit-Wolf | OAS |
|---|---|---|---|---|
| 72.2 | 1444 | +0.0067 | +0.0068 | +0.0068 |
| 16.2 | 324 | +0.0072 | +0.0076 | +0.0076 |
| 7.2 | 144 | +0.0035 | +0.0051 | +0.0052 |
| 3.2 | 64 | **-0.0086** | +0.0011 | +0.0016 |
| 1.8 | 36 | **-0.0233** | +0.0006 | +0.0013 |

*(SSIM change against not pre-whitening at all, R = 4.)*

**Below n/nc ~ 4 the sample covariance makes pre-whitening actively harmful,
and shrinkage removes that failure entirely.** At n/nc = 1.8 the sample
estimator costs 0.023 SSIM -- three times the benefit pre-whitening ever
delivers -- while OAS still gains 0.001. The same pattern holds at R=8, where
shrinkage turns -0.002 into +0.005. Shrinkage never *helps* in the
sample-rich regime; what it buys is that pre-whitening stops being a gamble
when noise samples are scarce, which is the regime a short pre-scan actually
occupies.

Two things this rules out. Signal contamination of the corner region is not
what limits the large-n end: energy per sample rises only 7.5% from n = 16 to
n = 4096, so the corners are signal-free throughout and the flatness above
n/nc ~ 16 is genuine convergence, not bias creeping in. And the small size of
the pre-whitening gain itself is not a conditioning artefact -- it survives at
every n -- so the honest conclusion is that this array's noise correlation
(off-diagonal up to 0.54) costs less in reconstruction quality than the
correlation magnitude suggests.

### Protocol changes the real data forced

* **4 of 24 volumes are not fully sampled** (96 all-zero phase-encode lines,
  elliptical/partial-Fourier). Undersampling an already-gapped k-space measures
  the acquisition, not the reconstructor; they are excluded.
* **Metrics are body-masked**, and the mask must be tight. Against an unmasked
  RSS reference a *fully sampled* matched-filter combine scores SSIM 0.48 with
  80% of its error energy outside the head. An early mask thresholded and then
  filled holes, which welded 154 isolated noise specks into one blob covering
  0.83 of the frame -- about twice the head -- so metrics were still averaging
  over background.
* **The reference is built before whitening.** Whitening rescales each coil, so
  an RSS computed after it is a different image; evaluating the whitened and
  unwhitened conditions against their own references compared two different
  targets and made that contrast meaningless.
* **Two references are reported.** `ssim` is against the fully-sampled RSS
  image; `ssim_unfold` is against the R=1 matched filter built from the *same*
  estimated maps, separating aliasing removal from map resolution.
* **The aperture sweep was confounded** and is reported separately
  (`run_aperture_fixed.py`). Comparing array sizes across whatever volumes have
  that many coils measures a different patient set at every level -- it reported
  20 coils as *worse* than 16. Restricted to the same 27 slices, SENSE runs
  0.932 -> 0.947 -> 0.946 -> 0.957 for 4 -> 8 -> 12 -> 16 coils.

## Layout

| Path | Role |
|---|---|
| `src/simulate.py` | forward model, phantom, coil maps, sampling masks |
| `src/recon.py` | zero-filled, SENSE (+ Psi, diagonal loading, worst-case robust), GRAPPA, ESPIRiT |
| `src/metrics.py` | NMSE, PSNR, SSIM, lesion-ROI NMSE |
| `src/fastmri_data.py` | fastMRI loader, noise covariance, pre-whitening, body mask |
| `src/brats_kspace.py` | BraTS -> multicoil k-space using a borrowed real array |
| `src/learned_recon.py` | pretrained fastMRI E2E-VarNet, adapted to this study's conventions |
| `experiments.py` | 7 seeded simulated sweeps |
| `run_fastmri.py` | the same sweeps on real fastMRI k-space |
| `run_aperture_fixed.py` | confound-free aperture sweep |
| `run_espirit_compare.py`, `run_espirit_tune.py` | subspace vs ratio steering-vector estimation |
| `run_brats_tumor.py` | acceleration -> reconstruction -> segmentation -> Dice |
| `run_model_order.py` | MDL/AIC subspace order vs a tuned ESPIRiT threshold |
| `src/array_cov.py`, `run_noise_cov.py` | shrinkage noise covariance vs sample covariance |
| `src/coil_select.py`, `run_coil_select.py` | which K of N coils to keep (sensor selection) |
| `run_effective_g.py` | pseudo-replica g_eff, bias/variance split, noise-predictor regression |
| `analyze_effective_g.py` | volume-level bootstrap for the above |
| `src/stats.py`, `run_stats_report.py` | one bootstrap for every reported contrast, with Holm control |
| `run_noref_quality.py` | no-reference reconstruction quality prediction from acquisition-side features |
| `run_noise_regime.py` | does g localise error once noise dominates it? (sweep over SNR and covariance) |
| `run_prefix_quality.py` | predicting the finished scan's quality from a partial acquisition |
| `run_qc_agent_recon.py` | does the MedAgent QC signature catch reconstruction-induced failure? |
| `src/gpu_guard.py` | two-level GPU admission control (resident models, then peak) |
| `ARRAY_SIGNAL_PROCESSING_HANDOFF.md` | the array half: results, dead ends and the leads left, self-contained |
| `run_lesion_error.py` | where the reconstruction error lands, by anatomical region |
| `run_resolution_psf.py` | local point-spread-function width: the resolution side of the trade |
| `validate_snr_law.py` | **replication** against the papers' own quantitative claims |
| `validate_varnet.py` | gate for the learned reconstructor (adapter + readout handling) |
| `test_espirit.py` | Parseval and known-map guards for ESPIRiT |
| `analyze_stats.py` | cluster bootstrap confidence intervals |
| `make_figures_real.py` | figures |
| `make_miccai_figures.py` | the four downstream-task panels (F1-F4) |

```bash
python validate_snr_law.py && python test_espirit.py     # correctness first
python run_fastmri.py && python run_brats_tumor.py
python analyze_stats.py && python make_figures_real.py
```

*The downstream scripts (`run_brats_tumor.py`, `run_lesion_error.py`,
`run_qc_agent_recon.py`, `run_qc_full29.py`) are documented in `README.md`.*

## Current limitations

* The two cohorts differ in tumour size by 3.5x, so cohort and lesion size are
  confounded in the between-cohort comparison; only the within-cohort
  size correlations separate them, and those rest on 32 cases each.
* The tumor half transplants a real array onto BraTS anatomy: those coils never
  scanned those patients, and BraTS ships magnitude images, so phase is
  synthetic (verified not to matter -- identical results at 0 and 0.6*pi).
* fastMRI brain is general clinical imaging with no tumor labels, and BraTS has
  no k-space; no single public dataset supports both halves at once.
* 20 volumes / 60 slices and 32 BraTS cases. The volume-level bootstrap is
  honest about what that buys: small effects such as pre-whitening are resolved
  only barely, and the aperture sweep is limited to the 9 volumes with >= 16
  coils. The BraTS half is 32 of 1350 available subjects; the move from 12 in
  directory order to 32 at random changed SENSE's R=8 Dice from 0.104 to 0.234
  without changing any conclusion, which is roughly the size of sampling effect
  to expect at this n.
* ESPIRiT is tuned by grid search. Model-order selection (MDL/AIC) was
  implemented to remove that tuning and does not work, for the reason the
  bullet used to anticipate: the spectrum has no clean signal/null gap, so
  there is no order to select (see "Three remedies that do not work"). The
  SURE-based criterion of Iyer et al. 2018 (arXiv:1811.05672) is the remaining
  untried option and does not assume a spectral gap.
* The small pre-whitening gain is **not** explained by a poorly conditioned
  sample covariance, as this file previously speculated: at the settings used
  n/nc = 72 and the shrinkage intensity is 0.02 (see "How many noise samples
  does pre-whitening need?"). Why correlated coil noise costs so little here
  remains open.
* Resolution is measured by local PSF (see "The resolution side of the trade"),
  but VarNet's width depends on the probe amplitude by 0.889 at R=8, so its
  resolution is bracketed rather than determined. Separating the network's
  genuine local Jacobian from its response to the probe's own noise would need
  either many probes averaged at fixed small amplitude or an explicit Jacobian-
  vector product through the network, neither of which is done here.
* Two off-the-shelf models is not a survey. They were chosen to bracket the
  design space (unrolled with data consistency vs image-domain post-processing)
  and the properties they share are reported as properties of learned
  reconstruction, but two points cannot rule out that both inherit something
  from the fastMRI training distribution they have in common -- which also
  overlaps the evaluation set (see "What this comparison is not"). Neither is
  retrained, so nothing here separates the paradigm from these particular
  weights.
* The U-Net's benefit is protocol-dependent and it is reported under the
  stricter protocol. Under fastMRI's own unmasked evaluation against
  `reconstruction_rss` it beats zero-filling clearly (0.546 vs 0.526 at R=4,
  0.568 vs 0.512 at R=8); under the body-masked, gain-fitted metric used
  throughout this study the gap nearly closes. Its visible gain is largely in
  the background, which the body mask removes by design. An affine rather than
  gain-only fit does not change this (0.8353 -> 0.8335), so it is not an
  additive offset.
* `g_eff` is measured on magnitude images and saturates where the analytic g is
  large, so the R=8 measured/analytic gap is a property of the metric as much as
  of the reconstructors. A complex-domain version would need a reconstructor
  that emits phase, which VarNet as released does not.
* Coil selection is greedy and no approximation bound applies -- submodularity is
  not established for either objective -- but the gap is now measured rather than
  unknown at K=5, R=4: greedy is 11% above the true g minimum and within 0.0018
  SSIM of the SNR-law optimum. Whether that holds at larger K is untested, since
  enumeration stops being tractable past about K=6.
* Sampling is strictly uniform, because Cartesian SENSE requires it. Non-uniform
  and sparse-array designs (the CAIPIRINHA family, and the coprime/nested-array
  literature) are the natural array-geometry extension and are untouched here.

*Limitations specific to the tumour study are in `README.md`.*
