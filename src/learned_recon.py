"""Learned reconstructor: the pretrained fastMRI E2E-VarNet, as a fourth estimator.

Why it belongs in an array-processing study. SENSE and GRAPPA are the two
classical answers to the array inverse problem, and the finding here is that
they fail in complementary ways -- SENSE inverts an explicit steering-vector
model and inherits its conditioning, GRAPPA never forms one. An unrolled
network is the modern third answer, and it is the control that decides which
story the failure supports: if a learned reconstructor holds where SENSE
collapses, the fragility is specific to explicit model inversion; if it fails
in the same places, the limit is the measurement, not the estimator.

Nothing here is trained. The weights are Facebook's published brain
leaderboard model (12 cascades, sens_chans 8, chans 18 -- read off the
checkpoint, not assumed), used off the shelf. That is the only honest way to
include a learned method in a study that trains nothing else: it is the
strongest publicly available reconstructor for this exact anatomy and
undersampling regime, and it never saw these validation volumes as training
data.

Three conventions have to be bridged and each is a real source of error:

1. **Axis order.** `recon.py` works in (coil, ky, kx) with the phase encode on
   axis -2. fastMRI -- and therefore the network -- was trained on (coil, kx,
   ky) with the phase encode LAST. k-space is transposed on the way in, the
   image transposed back on the way out.

2. **Readout oversampling.** The network saw the 2x-oversampled readout every
   fastMRI brain volume is acquired with; `run_fastmri.py` removes it up front
   because the classical estimators are unaffected (1.4e-16) and it halves the
   pixel count. Removing it before the network is *not* obviously harmless --
   it changes the k-space support the convolutions were trained on. Both input
   widths are supported and `compare_readout_handling` measures which the
   network actually prefers instead of assuming.

3. **Scale.** VarNet is scale-equivariant: `NormUnet` normalizes per instance
   and unnormalizes, and the soft data-consistency term is linear in k-space.
   Raw k-space is therefore passed through unscaled, and the study's existing
   least-squares global-gain fit handles the rest.

The network estimates its own sensitivities from the ACS region, so it takes no
`maps` argument -- which is itself the point of comparison against SENSE.
"""
from __future__ import annotations

import functools
import os

import numpy as np
import torch
import fastmri

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.normpath(
    os.path.join(_HERE, os.pardir, "models", "varnet_brain_leaderboard_state_dict.pt"))

# Read off the checkpoint itself: sens_net first conv (8,2,3,3) -> sens_chans=8,
# cascades.0 first conv (18,2,3,3) -> chans=18, 12 distinct cascade indices.
ARCH = dict(num_cascades=12, sens_chans=8, sens_pools=4, chans=18, pools=4)


@functools.lru_cache(maxsize=2)
def load_varnet(ckpt: str | None = None, device: str | None = None):
    """Build the VarNet and load the pretrained brain weights (cached)."""
    from fastmri.models import VarNet

    ckpt = ckpt or DEFAULT_CKPT
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"VarNet weights not found at {ckpt}. Fetch with:\n"
            "  curl -L -o models/varnet_brain_leaderboard_state_dict.pt \\\n"
            "    https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/"
            "brain_leaderboard_state_dict.pt")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    try:
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    except Exception:
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    # Lightning checkpoints prefix everything with "varnet."; the leaderboard
    # state dict does not. Strip it if present so both forms load.
    sd = {(k[len("varnet."):] if k.startswith("varnet.") else k): v
          for k, v in sd.items()}

    model = VarNet(**ARCH)
    model.load_state_dict(sd, strict=True)   # strict: a silent partial load
    model.eval().to(dev)                     # would look like a bad baseline
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _to_fastmri_tensor(ksp):
    """(nc, ky, kx) complex -> (1, nc, kx, ky, 2) float32, phase encode last."""
    k = np.ascontiguousarray(np.moveaxis(np.asarray(ksp), -2, -1))   # (nc, kx, ky)
    t = np.stack([k.real, k.imag], axis=-1).astype(np.float32)
    return torch.from_numpy(t).unsqueeze(0)


def _mask_tensor(mask):
    """(ky, kx) bool -> (1, 1, 1, ky, 1) bool, the shape fastMRI broadcasts."""
    m = np.asarray(mask)
    line = m[:, 0] if m.ndim == 2 else m                # sampling is per PE line
    return torch.from_numpy(np.ascontiguousarray(line).astype(bool)
                            ).view(1, 1, 1, -1, 1)


def crop_readout(img, factor=2):
    """Centre-crop the readout axis of an image by `factor` (oversampling removal)."""
    nx = img.shape[-1]
    n = nx // factor
    x0 = nx // 2 - n // 2
    return img[..., x0:x0 + n]


def varnet_recon(ksp_us, mask, n_acs=None, crop_ro=False,
                 model=None, device=None, ckpt=None,
                 calib_ksp=None, calib_mask=None, calib_n_acs=None):
    """Reconstruct one undersampled slice with the pretrained VarNet.

    Args:
        ksp_us:  (nc, ky, kx) complex k-space. Masked internally, so passing
                 either the full or the already-zero-filled array is equivalent.
        mask:    (ky, kx) or (ky,) boolean sampling mask along the phase encode.
        n_acs:   width of the fully sampled centre. Passed as
                 `num_low_frequencies`; if None the network infers it from the
                 contiguous centre of the mask, which is correct for the
                 uniform+ACS masks used here but not for every pattern.
        crop_ro: crop the readout axis of the output by 2x, for when the input
                 still carries fastMRI's readout oversampling.

    Returns:
        (ky, kx) float64 magnitude image, in this study's orientation.
    """
    model = model or load_varnet(ckpt, device)
    dev = next(model.parameters()).device

    k = np.asarray(ksp_us)
    m = np.asarray(mask)
    if m.ndim == 1:
        m = m[:, None]
    k = k * m                                   # idempotent if already masked

    kt = _to_fastmri_tensor(k).to(dev)
    mt = _mask_tensor(m).to(dev)
    nlf = int(n_acs) if n_acs else None

    with torch.no_grad():
        if calib_ksp is None:
            img = model(kt, mt, num_low_frequencies=nlf)  # (1, kx, ky)
        else:
            # Calibration split. VarNet estimates its coil sensitivities from
            # the fully sampled centre of whatever k-space it is handed, so on
            # a strictly uniform mask there is no centre and `sens_net`
            # normalises by an RSS over an empty band -- it returns NaN, which
            # is why the uniform arm scored 0.000 rather than scoring badly.
            # SENSE and GRAPPA are not in that position: they calibrate on a
            # separate ACS survey and reconstruct the shared uniform samples.
            # This gives VarNet the same split, so every arm still reconstructs
            # an identical line set while each calibrates from the same survey.
            #
            # It is necessary but NOT sufficient: with correct sensitivities the
            # output is finite but still useless on a uniform mask (scale-matched
            # NRMSE 0.84-0.87, i.e. no better than an all-zero image), because
            # the cascades' learned prior and data consistency were trained on
            # masks that always carry a low-frequency anchor. The same model on
            # uniform+ACS reconstructs at NRMSE 0.245. The conclusion is that
            # these pretrained networks cannot be evaluated under strictly
            # uniform sampling at all without retraining -- so the learned arms
            # are reported under the shared-acs convention only.
            ck = np.asarray(calib_ksp)
            cm = np.asarray(calib_mask)
            if cm.ndim == 1:
                cm = cm[:, None]
            ckt = _to_fastmri_tensor(ck * cm).to(dev)
            cmt = _mask_tensor(cm).to(dev)
            cnlf = int(calib_n_acs) if calib_n_acs else None
            sens = model.sens_net(ckt, cmt, cnlf)
            pred = kt.clone()
            for cascade in model.cascades:
                pred = cascade(pred, kt, mt, sens)
            img = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(pred)), dim=1)

    out = img.squeeze(0).float().cpu().numpy().T          # -> (ky, kx)
    out = out.astype(np.float64)
    return crop_readout(out) if crop_ro else out


def compare_readout_handling(ksp_full, mask, n_acs=None, model=None, device=None):
    """Both ways of handling readout oversampling, for the same slice.

    Returns {'native': img, 'cropped': img} -- 'native' feeds the network the
    oversampled k-space it was trained on and crops the image afterwards;
    'cropped' removes the oversampling in k-space first, as the classical
    estimators see it. Which is better is measured, not assumed.
    """
    import fastmri_data as fd

    model = model or load_varnet(device=device)
    native = varnet_recon(ksp_full, mask, n_acs, crop_ro=True, model=model)

    k_small = fd.remove_readout_oversampling(ksp_full)
    m_small = np.asarray(mask)
    if m_small.ndim == 2 and m_small.shape[-1] != k_small.shape[-1]:
        m_small = m_small[:, :k_small.shape[-1]]
    cropped = varnet_recon(k_small, m_small, n_acs, crop_ro=False, model=model)
    return {"native": native, "cropped": cropped}


# --- second learned model: the fastMRI U-Net baseline -------------------------
#
# One checkpoint cannot distinguish "learned reconstruction behaves this way"
# from "this network behaves this way", so a second one is needed and it should
# not be a variant of the first. The fastMRI U-Net baseline is the right
# contrast because it is a different *paradigm*, not a different architecture:
#
#   VarNet  unrolled in k-space, 12 data-consistency steps, estimates its own
#           coil sensitivities from the ACS block. 29.9 M parameters.
#   U-Net   image-domain post-processing of the zero-filled RSS image. It never
#           sees k-space, never sees the sampling mask, never forms or uses a
#           sensitivity map, and enforces no data consistency. 496.4 M
#           parameters -- 17x larger, and structurally far weaker.
#
# If the two behave alike on the noise-predictor and resolution measurements,
# that is a property of learned reconstruction. If they diverge, it is a
# property of data consistency, and the study can say which.
#
# This is also the checkpoint Kleineisel et al. use (256 channels, 4.96e8
# parameters), so their reported numbers and these are the same model.

UNET_CKPT = os.path.normpath(
    os.path.join(_HERE, os.pardir, "models", "unet_brain_leaderboard_state_dict.pt"))
UNET_ARCH = dict(in_chans=1, out_chans=1, chans=256, num_pool_layers=4, drop_prob=0.0)


@functools.lru_cache(maxsize=2)
def load_unet(ckpt: str | None = None, device: str | None = None):
    """Build the U-Net and load the pretrained brain weights (cached)."""
    from fastmri.models import Unet

    ckpt = ckpt or UNET_CKPT
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"U-Net weights not found at {ckpt}. Fetch with:\n"
            "  curl -L -o models/unet_brain_leaderboard_state_dict.pt \\\n"
            "    https://dl.fbaipublicfiles.com/fastMRI/trained_models/unet/"
            "brain_leaderboard_state_dict.pt")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    except Exception:
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {(k[len("unet."):] if k.startswith("unet.") else k): v for k, v in sd.items()}

    model = Unet(**UNET_ARCH)
    model.load_state_dict(sd, strict=True)
    model.eval().to(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def unet_recon(ksp_us, mask, model=None, device=None, ckpt=None):
    """Reconstruct one undersampled slice with the pretrained fastMRI U-Net.

    Takes k-space only so the call site matches the other estimators, but the
    network sees just the zero-filled root-sum-of-squares image built from it.
    fastMRI's own `UnetDataTransform` crops the oversampled readout in *image*
    space before the RSS; `remove_readout_oversampling` does the identical crop
    in k-space beforehand, so passing the already-cropped k-space here
    reproduces their pipeline exactly rather than approximately.

    Instance normalization (zero mean, unit standard deviation, clamped to
    +/-6) is part of the trained model's input contract, not a preprocessing
    choice, and the output is un-normalized with the same statistics.
    """
    model = model or load_unet(ckpt, device)
    dev = next(model.parameters()).device

    k = np.asarray(ksp_us)
    m = np.asarray(mask)
    if m.ndim == 1:
        m = m[:, None]
    coil = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k * m, axes=(-2, -1)),
                                        norm="ortho"), axes=(-2, -1))
    zf = np.sqrt((np.abs(coil) ** 2).sum(0))                # zero-filled RSS

    x = torch.from_numpy(zf.astype(np.float32)).to(dev)
    mean, std = x.mean(), x.std()
    xn = ((x - mean) / (std + 1e-11)).clamp(-6, 6)
    with torch.no_grad():
        out = model(xn[None, None])[0, 0]
    return (out * std + mean).float().cpu().numpy().astype(np.float64)
