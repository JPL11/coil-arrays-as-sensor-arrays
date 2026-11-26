# Receiver-Coil Arrays as Sensor Arrays


SENSE (Pruessmann et al., *MRM* 1999) and GRAPPA (Griswold et al., *MRM* 2002) implemented from the
original papers, without a reconstruction library, and studied as sensor-array processing. The
detailed write-up of methods, results and dead ends is in [`README_ARRAY.md`](README_ARRAY.md); the
working log is [`ARRAY_SIGNAL_PROCESSING_HANDOFF.md`](ARRAY_SIGNAL_PROCESSING_HANDOFF.md).

## Where to start (for verification)

| What | File |
|---|---|
| SENSE (+Ψ, diagonal loading, worst-case robust), GRAPPA, ESPIRiT | `src/recon.py` |
| Replication of the papers' quantitative claims (SNR law, GRAPPA error) | `validate_snr_law.py` |
| ESPIRiT correctness guards | `test_espirit.py` |
| Recreated figures from both papers | `make_paper_figures.py` → `figures/paper_*.png` |
| Map mismatch, ACS starvation, aperture on fastMRI | `run_fastmri.py`, `run_aperture_fixed.py` |
| MDL/AIC model order | `run_model_order.py` |
| Noise covariance shrinkage | `src/array_cov.py`, `run_noise_cov.py` |
| g-factor vs noise and error | `run_effective_g.py`, `analyze_effective_g.py` |
| Coil selection | `src/coil_select.py`, `run_coil_select.py` |
| Saved outputs | `results/*.csv` |

`README_ARRAY.md` ("Layout") lists every script.

## Running

Python 3 with `numpy scipy matplotlib pandas h5py scikit-image`; `torch` and the `fastmri` package
are needed only for `src/learned_recon.py`. Run from the repository root:

```bash
python validate_snr_law.py && python test_espirit.py     # simulation only, no data needed
python make_paper_figures.py
```

The fastMRI scripts expect the multicoil data at `../datasets/fastmri/multicoil_val` (and
`multicoil_balanced`). fastMRI is not redistributed here; request it at https://fastmri.med.nyu.edu.

Not included: the raw data, pretrained model weights, and the large per-pixel PSF table
(`results/resolution_psf.csv`, 25 MB), which `run_resolution_psf.py` regenerates.
