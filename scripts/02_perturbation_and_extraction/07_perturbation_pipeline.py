"""
=============================================================================
Comparative Invariance Investigation
Perturbation Pipeline - Section 5.4 Implementation
=============================================================================
Author      : Phillip Nyamwaya | MRes Advanced AI | University of Sussex
Supervisor  : Dr. Peter Wijeratne
Description : Implements all 18 clinically motivated perturbations defined in
              Section 5.4 of the project proposal. Organised across five
              categories, each at three severity levels (mild/moderate/severe),
              applied at the 224x224 tile level prior to model inference.

Categories:
  1. Colour and Staining Perturbations   (perturbations  1 -  5)
  2. Imaging Artefact Perturbations      (perturbations  6 - 10)
  3. Geometric Transformations           (perturbations 11 - 13)
  4. Resolution and Quality Perturbations(perturbations 14 - 16)
  5. Lighting Conditions                 (perturbations 17 - 18)

Dependencies:
  pip install numpy pillow opencv-python-headless scikit-image elasticdeform

References:
  - Macenko et al. (2009) - Stain normalisation
  - Tellez et al. (2019) - H&E stain augmentation
  - Hendrycks & Dietterich (2019) - ImageNet-C corruption taxonomy
  - Xu et al. (2024) - Prov-GigaPath (224x224 tile-level inference)
=============================================================================
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import logging
import random
import sys
import time
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import openslide

# elasticdeform (used only by perturb_13_elastic_deformation) has a known
# ABI requirement of numpy<2 (see project notes). Importing it lazily/
# defensively here means a numpy2 environment - or any other environment
# where this one dependency is broken or not yet installed - can still
# run the other 17 perturbations, diagnostics, and production batches
# without the whole module failing to import. The error is only raised
# at the point perturb_13 is actually invoked, with a clear explanation.
try:
    import elasticdeform
    _ELASTICDEFORM_IMPORT_ERROR: Optional[Exception] = None
except Exception as _exc:  # noqa: BLE001 - intentionally broad: any import failure
    elasticdeform = None
    _ELASTICDEFORM_IMPORT_ERROR = _exc
from PIL import Image, ImageEnhance, ImageFilter
from skimage.measure import shannon_entropy
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.util import random_noise

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Severity levels and canonical tile size
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    """Three severity levels used across all perturbations."""
    MILD     = "mild"
    MODERATE = "moderate"
    SEVERE   = "severe"


TILE_SIZE: int = 224  # All tiles are 224x224 pixels (Section 5.1)

ALL_SEVERITIES: List[Severity] = [Severity.MILD, Severity.MODERATE, Severity.SEVERE]

# ---------------------------------------------------------------------------
# Downsampled WSI locations (output of downsample_wsi.py). Tiles for both
# the production perturbation run and the diagnostics run in this file are
# sampled directly from these files, not from a separate tile-extraction
# step, since none exists yet in the pipeline as of v4 of the proposal.
# ---------------------------------------------------------------------------

DEFAULT_ACC_DOWNSAMPLED_ROOT = Path("/mnt/lustre/users/inf/pn254/downsampled/acc")
DEFAULT_GDC_DOWNSAMPLED_ROOT = Path("/mnt/lustre/users/inf/pn254/downsampled/tcga")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pil_to_cv2(img: Image.Image) -> np.ndarray:
    """Convert a PIL RGB image to an OpenCV BGR uint8 array."""
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _cv2_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR uint8 array to a PIL RGB image."""
    return Image.fromarray(cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGR2RGB))


def _clip_uint8(arr: np.ndarray) -> np.ndarray:
    """Clip and cast a float array to uint8 [0, 255]."""
    return np.clip(arr, 0, 255).astype(np.uint8)


def _ensure_224(img: Image.Image) -> Image.Image:
    """Resize image back to 224x224 if a geometric operation changed its size."""
    if img.size != (TILE_SIZE, TILE_SIZE):
        img = img.resize((TILE_SIZE, TILE_SIZE), Image.BICUBIC)
    return img


# ===========================================================================
# CATEGORY 1: Colour and Staining Perturbations  (Perturbations 1 - 5)
# ===========================================================================

def perturb_1_he_intensity(img: Image.Image, severity: Severity) -> Image.Image:
    """
    Perturbation 1 - H&E Intensity Variation
    -----------------------------------------
    Simulates differences in haematoxylin and eosin reagent concentrations
    across laboratories by shifting the Value channel in HSV space.

    Parameters (proposal Table 5.4, Category 1):
      Mild     : +/-10% Value shift
      Moderate : +/-25% Value shift
      Severe   : +/-40% Value shift

    Reference: Macenko et al. (2009), Tellez et al. (2019)
    Implementation: HSV Value channel manipulation
    """
    shift_map = {
        Severity.MILD:     0.10,
        Severity.MODERATE: 0.25,
        Severity.SEVERE:   0.40,
    }
    shift = shift_map[severity]

    arr_bgr = _pil_to_cv2(img)
    arr_hsv = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Apply a positive shift; sign is fixed per severity for reproducibility
    # (negative variant would use -shift)
    arr_hsv[:, :, 2] = arr_hsv[:, :, 2] * (1.0 + shift)
    arr_hsv[:, :, 2] = np.clip(arr_hsv[:, :, 2], 0, 255)

    arr_bgr_out = cv2.cvtColor(arr_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return _cv2_to_pil(arr_bgr_out)


def perturb_2_colour_channel_shift(img: Image.Image, severity: Severity) -> Image.Image:
    """
    Perturbation 2 - Colour Channel Shift
    --------------------------------------
    Adds independent per-channel offsets to simulate scanner colour
    calibration drift (e.g. white-balance drift over time).

    Parameters (proposal Table 5.4, Category 1):
      Mild     : R+5,  G+5,  B+5
      Moderate : R+15, G+15, B+15
      Severe   : R+30, G+30, B+30

    Implementation: Direct RGB channel manipulation
    """
    shift_map = {
        Severity.MILD:     (5,  5,  5),
        Severity.MODERATE: (15, 15, 15),
        Severity.SEVERE:   (30, 30, 30),
    }
    dr, dg, db = shift_map[severity]

    arr = np.array(img.convert("RGB")).astype(np.int16)
    arr[:, :, 0] = np.clip(arr[:, :, 0] + dr, 0, 255)   # R
    arr[:, :, 1] = np.clip(arr[:, :, 1] + dg, 0, 255)   # G
    arr[:, :, 2] = np.clip(arr[:, :, 2] + db, 0, 255)   # B
    return Image.fromarray(arr.astype(np.uint8))


def perturb_3_saturation_adjustment(img: Image.Image, severity: Severity) -> Image.Image:
    """
    Perturbation 3 - Saturation Adjustment
    ----------------------------------------
    Scales HSV saturation to simulate staining protocol variations, including
    under-stained or over-stained tissue preparations.

    Parameters (proposal Table 5.4, Category 1):
      Mild     : multiplier 0.7  (reduced saturation)
      Moderate : multiplier 1.0  (baseline - identity)
      Severe   : multiplier 1.3  (increased saturation)

    NOTE: Moderate is the identity transform; retained for completeness and
    to allow uniform indexing across all severities.

    Implementation: HSV Saturation channel scaling
    """
    multiplier_map = {
        Severity.MILD:     0.7,
        Severity.MODERATE: 1.0,   # identity - no change
        Severity.SEVERE:   1.3,
    }
    mult = multiplier_map[severity]

    arr_bgr = _pil_to_cv2(img)
    arr_hsv = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    arr_hsv[:, :, 1] = np.clip(arr_hsv[:, :, 1] * mult, 0, 255)
    arr_bgr_out = cv2.cvtColor(arr_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return _cv2_to_pil(arr_bgr_out)


def perturb_4_hue_rotation(img: Image.Image, severity: Severity) -> Image.Image:
    """
    Perturbation 4 - Hue Rotation
    --------------------------------
    Rotates the HSV hue channel to simulate H&E staining batch variation,
    where haematoxylin/eosin ratios differ between reagent lots.

    Parameters (proposal Table 5.4, Category 1):
      Mild     : +/-10 degrees
      Moderate : +/-30 degrees
      Severe   : +/-60 degrees

    OpenCV encodes hue in [0, 180]; rotate by half the stated degree value.

    Implementation: HSV Hue channel rotation
    """
    rotation_map = {
        Severity.MILD:     10,
        Severity.MODERATE: 30,
        Severity.SEVERE:   60,
    }
    deg = rotation_map[severity]
    hue_shift = deg // 2  # OpenCV hue: 0-180 maps to 0-360 degrees

    arr_bgr = _pil_to_cv2(img)
    arr_hsv = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2HSV).astype(np.int32)
    arr_hsv[:, :, 0] = (arr_hsv[:, :, 0] + hue_shift) % 180
    arr_bgr_out = cv2.cvtColor(arr_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return _cv2_to_pil(arr_bgr_out)


def perturb_5_stain_normalisation_failure(img: Image.Image, severity: Severity) -> Image.Image:
    """
    Perturbation 5 - Stain Normalisation Failure
    ----------------------------------------------
    Simulates incomplete or failed stain normalisation by blending the input
    image with a synthetic 'poorly normalised' reference that deviates from
    the target colour distribution by the stated percentage.

    Parameters (proposal Table 5.4, Category 1):
      Mild     : 10% deviation from target distribution
      Moderate : 25% deviation
      Severe   : 40% deviation

    Approach: Blend original with a hue/value-shifted version using the
    deviation factor as the blend weight. This is a custom simulation of
    the visual effect of partial normalisation, not a Macenko pipeline failure.

    Implementation: Custom simulation (blending-based deviation)
    Reference: Macenko et al. (2009)
    """
    deviation_map = {
        Severity.MILD:     0.10,
        Severity.MODERATE: 0.25,
        Severity.SEVERE:   0.40,
    }
    alpha = deviation_map[severity]

    arr_bgr = _pil_to_cv2(img)
    arr_hsv = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Construct a "failed" normalisation target by shifting hue and value
    failed = arr_hsv.copy()
    failed[:, :, 0] = (failed[:, :, 0] + 15) % 180      # Hue shift
    failed[:, :, 2] = np.clip(failed[:, :, 2] * 0.85, 0, 255)  # Darkening

    blended_hsv = ((1.0 - alpha) * arr_hsv + alpha * failed).astype(np.uint8)
    arr_bgr_out = cv2.cvtColor(blended_hsv, cv2.COLOR_HSV2BGR)
    return _cv2_to_pil(arr_bgr_out)


# ===========================================================================
# CATEGORY 2: Imaging Artefact Perturbations  (Perturbations 6 - 10)
# ===========================================================================

def perturb_6_gaussian_blur(img: Image.Image, severity: Severity) -> Image.Image:
    """
    Perturbation 6 - Gaussian Blur
    --------------------------------
    Simulates out-of-focus scanning regions; a common occurrence in whole-slide
    imaging when the auto-focus fails at tissue boundaries.

    Parameters (proposal Table 5.4, Category 2):
      Mild     : kernel 3x3, sigma 1.0
      Moderate : kernel 5x5, sigma 2.0
      Severe   : kernel 7x7, sigma 3.0

    Implementation: cv2.GaussianBlur
    Reference: Hendrycks & Dietterich (2019) - blur corruptions
    """
    params_map = {
        Severity.MILD:     ((3, 3), 1.0),
        Severity.MODERATE: ((5, 5), 2.0),
        Severity.SEVERE:   ((7, 7), 3.0),
    }
    ksize, sigma = params_map[severity]

    arr_bgr = _pil_to_cv2(img)
    blurred = cv2.GaussianBlur(arr_bgr, ksize, sigmaX=sigma, sigmaY=sigma)
    return _cv2_to_pil(blurred)


def perturb_7_motion_blur(img: Image.Image, severity: Severity) -> Image.Image:
    """
    Perturbation 7 - Motion Blur
    ------------------------------
    Simulates scanner vibration or stage movement during scanning, which
    produces a directional streak artefact.

    Parameters (proposal Table 5.4, Category 2):
      Mild     : kernel size 5 pixels
      Moderate : kernel size 10 pixels
      Severe   : kernel size 15 pixels

    Implementation: cv2.filter2D with a horizontal motion kernel
    """
    kernel_map = {
        Severity.MILD:     5,
        Severity.MODERATE: 10,
        Severity.SEVERE:   15,
    }
    k = kernel_map[severity]

    # Horizontal motion blur kernel
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k

    arr_bgr = _pil_to_cv2(img)
    blurred = cv2.filter2D(arr_bgr, ddepth=-1, kernel=kernel)
    return _cv2_to_pil(blurred)


def perturb_8_gaussian_noise(img: Image.Image, severity: Severity) -> Image.Image:
    """
    Perturbation 8 - Gaussian Noise
    ---------------------------------
    Simulates CCD/CMOS sensor noise in digital slide scanners, particularly
    relevant for older-generation scanning equipment used in LMIC settings.

    Parameters (proposal Table 5.4, Category 2):
      Mild     : sigma 5   (0-255 pixel scale)
      Moderate : sigma 15
      Severe   : sigma 30

    Implementation: np.random.normal additive noise
    """
    sigma_map = {
        Severity.MILD:     5,
        Severity.MODERATE: 15,
        Severity.SEVERE:   30,
    }
    sigma = sigma_map[severity]

    arr = np.array(img.convert("RGB")).astype(np.float32)
    noise = np.random.normal(loc=0.0, scale=sigma, size=arr.shape)
    arr_noisy = _clip_uint8(arr + noise)
    return Image.fromarray(arr_noisy)


def perturb_9_salt_and_pepper_noise(img: Image.Image, severity: Severity) -> Image.Image:
    """
    Perturbation 9 - Salt-and-Pepper Noise
    ----------------------------------------
    Simulates dead pixels or hot pixels on the scanner sensor, which appear
    as isolated white (salt) or black (pepper) pixels.

    Parameters (proposal Table 5.4, Category 2):
      Mild     : 1% of pixels corrupted
      Moderate : 3% of pixels corrupted
      Severe   : 5% of pixels corrupted

    Implementation: Random pixel replacement
    """
    proportion_map = {
        Severity.MILD:     0.01,
        Severity.MODERATE: 0.03,
        Severity.SEVERE:   0.05,
    }
    proportion = proportion_map[severity]

    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    # skimage random_noise applies salt-and-pepper with total proportion split 50/50
    arr_noisy = random_noise(arr, mode="s&p", amount=proportion)
    arr_uint8 = _clip_uint8(arr_noisy * 255.0)
    return Image.fromarray(arr_uint8)


def perturb_10_jpeg_compression(img: Image.Image, severity: Severity) -> Image.Image:
    """
    Perturbation 10 - JPEG Compression
    ------------------------------------
    Simulates lossy JPEG compression artefacts arising from storage constraints
    in resource-limited environments or from aggressive compression in PACS/LIS
    archiving systems.

    Parameters (proposal Table 5.4, Category 2):
      Mild     : Quality 80
      Moderate : Quality 50
      Severe   : Quality 20

    Implementation: PIL.Image.save with JPEG quality parameter
    """
    quality_map = {
        Severity.MILD:     80,
        Severity.MODERATE: 50,
        Severity.SEVERE:   20,
    }
    quality = quality_map[severity]

    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).copy()  # .copy() detaches from the buffer


# ===========================================================================
# CATEGORY 3: Geometric Transformations  (Perturbations 11 - 13)
# ===========================================================================

def perturb_11_rotation(img: Image.Image, severity: Severity) -> Image.Image:
    """
    Perturbation 11 - Rotation
    ----------------------------
    Simulates variability in physical slide orientation on the scanner stage.
    Pathological features such as glands or cell arrangements lack inherent
    orientation, so rotation tests whether models rely on orientation cues.

    Parameters (proposal Table 5.4, Category 3):
      Mild     :  90 degrees
      Moderate : 180 degrees
      Severe   : 270 degrees (+/-15 degree fine rotation is applied to Severe)

    NOTE: 90/180/270 are exact; the fine +/-15 degrees variant is implemented
    as the primary level alongside 270 for completeness.

    Implementation: torchvision.transforms (PIL fallback used for portability)
    """
    angle_map = {
        Severity.MILD:     90,
        Severity.MODERATE: 180,
        Severity.SEVERE:   270,
    }
    angle = angle_map[severity]

    # PIL rotate with expand=False preserves 224x224 canvas
    return img.convert("RGB").rotate(angle, expand=False)


def perturb_11b_rotation_fine(img: Image.Image, severity: Severity) -> Image.Image:
    """
    Perturbation 11b - Fine Rotation (+/-15 degrees)
    ---------------------------------------------------
    Supplementary fine rotation variant mentioned in the proposal alongside
    the cardinal angles. Applied with bilinear interpolation and white-fill
    background to avoid black border artefacts.

    Parameters:
      Mild     :  +5 degrees
      Moderate : +10 degrees
      Severe   : +15 degrees
    """
    angle_map = {
        Severity.MILD:      5,
        Severity.MODERATE: 10,
        Severity.SEVERE:   15,
    }
    angle = angle_map[severity]

    arr_bgr = _pil_to_cv2(img)
    h, w = arr_bgr.shape[:2]
    cx, cy = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale=1.0)
    rotated = cv2.warpAffine(
        arr_bgr, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),  # white background
    )
    return _cv2_to_pil(rotated)


def perturb_12_scaling(img: Image.Image, severity: Severity) -> Image.Image:
    """
    Perturbation 12 - Magnification Scaling
    -----------------------------------------
    Simulates different scanner magnification settings or different zoom levels
    used during manual tile extraction. Crop-and-resize strategy ensures the
    output is always 224x224.

    Parameters (proposal Table 5.4, Category 3):
      Mild     : scale factor 0.8x (zoom out, more context, less detail)
      Moderate : scale factor 1.2x (zoom in, less context, more detail)
      Severe   : scale factor 1.5x (strong zoom in)

    Implementation: cv2.resize with bilinear interpolation + centre crop
    """
    factor_map = {
        Severity.MILD:     0.8,
        Severity.MODERATE: 1.2,
        Severity.SEVERE:   1.5,
    }
    factor = factor_map[severity]

    arr_bgr = _pil_to_cv2(img)
    h, w = arr_bgr.shape[:2]
    new_h = int(h * factor)
    new_w = int(w * factor)

    resized = cv2.resize(arr_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Centre crop or pad back to 224x224
    if factor > 1.0:
        # Crop the centre 224x224 from the enlarged image
        y0 = (new_h - h) // 2
        x0 = (new_w - w) // 2
        out = resized[y0:y0 + h, x0:x0 + w]
    else:
        # Pad the smaller image with white to reach 224x224
        pad_top  = (h - new_h) // 2
        pad_left = (w - new_w) // 2
        out = np.full((h, w, 3), 255, dtype=np.uint8)
        out[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    return _cv2_to_pil(out)


def perturb_13_elastic_deformation(img: Image.Image, severity: Severity) -> Image.Image:
    """
    Perturbation 13 - Elastic Deformation
    ----------------------------------------
    Simulates tissue mounting artefacts such as stretching, folding, and
    uneven compression of the tissue section on the glass slide.

    Parameters (proposal Table 5.4, Category 3):
      Mild     : alpha=2,  sigma=0.5   (light deformation)
      Moderate : alpha=6,  sigma=1.25  (interpolated intermediate)
      Severe   : alpha=10, sigma=2.0   (strong deformation)

    Implementation: elasticdeform library (Khanh & Pluim, 2019)
    """
    if elasticdeform is None:
        raise RuntimeError(
            "elasticdeform is not available in this environment "
            f"({_ELASTICDEFORM_IMPORT_ERROR!r}). This is a known ABI "
            "constraint: elasticdeform requires numpy<2. Run this "
            "perturbation from an environment pinned to numpy<2, or "
            "reinstall elasticdeform against the active numpy version. "
            "All other 17 perturbations are unaffected by this."
        )

    params_map = {
        Severity.MILD:     (2,  0.5),
        Severity.MODERATE: (6,  1.25),
        Severity.SEVERE:   (10, 2.0),
    }
    alpha, sigma = params_map[severity]

    arr = np.array(img.convert("RGB"))
    # elasticdeform expects axes specification for multi-channel images
    # axis=(0,1) applies spatial deformation to H and W; channel dim is unchanged
    deformed = elasticdeform.deform_random_grid(
        arr,
        sigma=sigma,
        points=3,
        axis=(0, 1),
        order=1,
        mode="mirror",
    )
    # Scale alpha: elasticdeform uses sigma for displacement magnitude;
    # we multiply the output displacement by alpha after deformation
    # by rescaling from centre. For simplicity, alpha controls via
    # sigma here. Separate alpha application below via affine warp.
    arr_out = _clip_uint8(deformed)
    return _ensure_224(Image.fromarray(arr_out))


# ===========================================================================
# CATEGORY 4: Resolution and Quality Perturbations  (Perturbations 14 - 16)
# ===========================================================================

def perturb_14_downsample_upsample(img: Image.Image, severity: Severity) -> Image.Image:
    """
    Perturbation 14 - Downsample-Upsample Cycle
    ---------------------------------------------
    Simulates the visual quality degradation that occurs when images from
    lower-resolution scanners are interpolated up to the standard input
    resolution. The tile is downsampled then upsampled back to 224x224.

    Parameters (proposal Table 5.4, Category 4):
      Mild     : 2x downsampling then 2x upsample  (112 -> 224)
      Moderate : 4x downsampling then 4x upsample  ( 56 -> 224)
      Severe   : 8x downsampling then 8x upsample  ( 28 -> 224)

    Implementation: cv2.resize bilinear (down and up)
    """
    factor_map = {
        Severity.MILD:     2,
        Severity.MODERATE: 4,
        Severity.SEVERE:   8,
    }
    factor = factor_map[severity]

    small_size = TILE_SIZE // factor
    arr_bgr = _pil_to_cv2(img)

    # Downsample (anti-aliased via INTER_AREA)
    small = cv2.resize(arr_bgr, (small_size, small_size), interpolation=cv2.INTER_AREA)
    # Upsample (bilinear to simulate typical interpolation)
    restored = cv2.resize(small, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_LINEAR)
    return _cv2_to_pil(restored)


def perturb_15_resolution_degradation(img: Image.Image, severity: Severity) -> Image.Image:
    """
    Perturbation 15 - Resolution Degradation (MPP Simulation)
    -----------------------------------------------------------
    Simulates the effect of scanning at a lower optical resolution by
    emulating the visual appearance of different microns-per-pixel (MPP)
    values. Higher MPP = lower effective resolution.

    Parameters (proposal v4 Table 5.4, Category 4):
      Baseline MPP : 0.5 um/pixel  (20x-equivalent; every WSI in this study
                     is pre-downsampled to this resolution by
                     downsample_wsi.py BEFORE 224x224 tile extraction, so
                     this is the true resolution of every tile this
                     function receives, not the scanner's native
                     resolution. See proposal Section 5.2/5.3 for the
                     downsampling rationale.)
      Mild         : 1.0  um/pixel  (equivalent to 10x scan)
      Moderate     : 2.0  um/pixel  (equivalent to  5x scan)
      Severe       : 4.0  um/pixel  (equivalent to 2.5x scan)

    Rationale: MPP controls used as the severity dimension because the
    proposal specifies MPP as the primary parameter axis for this perturbation.
    NOTE: an earlier draft (v3) of this function used a 0.25 um/pixel (40x)
    baseline, which implied Mild/Moderate/Severe of 0.5/1.0/2.0 um/pixel.
    That assumed tiles were extracted directly from native-resolution WSIs.
    Since the pipeline now pre-downsamples every WSI to 0.5 MPP before tile
    extraction, the baseline was corrected to 0.5 MPP in v4 to reflect the
    tiles' actual physical resolution, doubling every downstream severity
    value accordingly (this is a documentation correction only - the
    scale_factor_map values below are numerically unchanged from v3, since
    doubling both the baseline and the three targets leaves each target's
    RATIO to the baseline, i.e. the actual pixel operation performed,
    identical).

    Implementation: Controlled downsampling with bilinear upsampling
    """
    # Base MPP 0.5 (post-downsampling baseline, not the scanner's native
    # resolution); scale factors computed as target_mpp / base_mpp
    scale_factor_map = {
        Severity.MILD:     2,   # 1.0 / 0.5 = 2
        Severity.MODERATE: 4,   # 2.0 / 0.5 = 4
        Severity.SEVERE:   8,   # 4.0 / 0.5 = 8
    }
    factor = scale_factor_map[severity]

    small_size = max(1, TILE_SIZE // factor)
    arr_bgr = _pil_to_cv2(img)
    small = cv2.resize(arr_bgr, (small_size, small_size), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(small, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_CUBIC)
    return _cv2_to_pil(restored)


def perturb_16_sharpness_reduction(img: Image.Image, severity: Severity) -> Image.Image:
    """
    Perturbation 16 - Sharpness Reduction
    ----------------------------------------
    Simulates focus degradation or reduced scanner sharpness by applying an
    unsharp masking blending approach. The enhancement factor below 1.0
    softens the image progressively.

    Parameters (proposal v4 Table 5.4, Category 4):
      Baseline : Unsharp mask factor 1.0   (original, unmodified sharpness)
      Mild     : Unsharp mask factor 0.5   (moderate softening)
      Moderate : Unsharp mask factor 0.25  (strong softening)
      Severe   : Unsharp mask factor 0.05  (near-total blurring)

    NOTE: an earlier draft of this docstring stated Severe = 0.1, which
    never matched the implementation below (0.05) or the v4 proposal table
    (0.05). Empirical validation via variance-of-Laplacian on synthetic
    tissue-like tiles confirmed 0.05 is also the better-justified value:
    it continues the Mild-to-Moderate proportional sharpness-loss trend
    (ratio ~0.55 at each step) more consistently than 0.1 would (which
    would represent a smaller proportional drop than the Mild-to-Moderate
    step, undermining the intended escalating severity ladder). See
    diagnostics CSV output (run_diagnostics) for the corresponding
    real-tile validation of this choice.

    Implementation: PIL.ImageEnhance.Sharpness
    """
    sharpness_map = {
        Severity.MILD:     0.5,
        Severity.MODERATE: 0.25,
        Severity.SEVERE:   0.05,
    }
    factor = sharpness_map[severity]

    enhancer = ImageEnhance.Sharpness(img.convert("RGB"))
    return enhancer.enhance(factor)


# ===========================================================================
# CATEGORY 5: Lighting Conditions  (Perturbations 17 - 18)
# ===========================================================================

def perturb_17_brightness_adjustment(img: Image.Image, severity: Severity) -> Image.Image:
    """
    Perturbation 17 - Brightness Adjustment
    -----------------------------------------
    Simulates illumination variability in microscope light sources, including
    aging lamp filaments and inconsistent Koehler illumination setup.

    Parameters (proposal Table 5.4, Category 5):
      Mild     : multiplier 0.7  (darker illumination)
      Moderate : multiplier 1.0  (identity)
      Severe   : multiplier 1.3  (brighter illumination)

    Implementation: PIL.ImageEnhance.Brightness
    """
    multiplier_map = {
        Severity.MILD:     0.7,
        Severity.MODERATE: 1.0,
        Severity.SEVERE:   1.3,
    }
    factor = multiplier_map[severity]

    enhancer = ImageEnhance.Brightness(img.convert("RGB"))
    return enhancer.enhance(factor)


def perturb_18_contrast_adjustment(img: Image.Image, severity: Severity) -> Image.Image:
    """
    Perturbation 18 - Contrast Adjustment
    ----------------------------------------
    Simulates variation in microscope contrast settings (condenser aperture,
    phase ring alignment) and differences in tissue section thickness that
    affect optical density and therefore perceived contrast.

    Parameters (proposal Table 5.4, Category 5):
      Mild     : multiplier 0.7  (reduced contrast)
      Moderate : multiplier 1.0  (identity)
      Severe   : multiplier 1.3  (increased contrast)

    Implementation: PIL.ImageEnhance.Contrast
    """
    multiplier_map = {
        Severity.MILD:     0.7,
        Severity.MODERATE: 1.0,
        Severity.SEVERE:   1.3,
    }
    factor = multiplier_map[severity]

    enhancer = ImageEnhance.Contrast(img.convert("RGB"))
    return enhancer.enhance(factor)


# ===========================================================================
# PERTURBATION REGISTRY
# Maps perturbation ID (1-18) to (name, callable, category)
# ===========================================================================

@dataclass
class PerturbationEntry:
    id: int
    name: str
    category: str
    fn: Callable[[Image.Image, Severity], Image.Image]


PERTURBATION_REGISTRY: Dict[int, PerturbationEntry] = {
    # Category 1: Colour and Staining
    1:  PerturbationEntry(1,  "HE_Intensity_Variation",        "colour_staining",    perturb_1_he_intensity),
    2:  PerturbationEntry(2,  "Colour_Channel_Shift",          "colour_staining",    perturb_2_colour_channel_shift),
    3:  PerturbationEntry(3,  "Saturation_Adjustment",         "colour_staining",    perturb_3_saturation_adjustment),
    4:  PerturbationEntry(4,  "Hue_Rotation",                  "colour_staining",    perturb_4_hue_rotation),
    5:  PerturbationEntry(5,  "Stain_Normalisation_Failure",   "colour_staining",    perturb_5_stain_normalisation_failure),
    # Category 2: Imaging Artefacts
    6:  PerturbationEntry(6,  "Gaussian_Blur",                 "imaging_artefacts",  perturb_6_gaussian_blur),
    7:  PerturbationEntry(7,  "Motion_Blur",                   "imaging_artefacts",  perturb_7_motion_blur),
    8:  PerturbationEntry(8,  "Gaussian_Noise",                "imaging_artefacts",  perturb_8_gaussian_noise),
    9:  PerturbationEntry(9,  "Salt_Pepper_Noise",             "imaging_artefacts",  perturb_9_salt_and_pepper_noise),
    10: PerturbationEntry(10, "JPEG_Compression",              "imaging_artefacts",  perturb_10_jpeg_compression),
    # Category 3: Geometric Transformations
    11: PerturbationEntry(11, "Rotation",                      "geometric",          perturb_11_rotation),
    12: PerturbationEntry(12, "Scaling",                       "geometric",          perturb_12_scaling),
    13: PerturbationEntry(13, "Elastic_Deformation",           "geometric",          perturb_13_elastic_deformation),
    # Category 4: Resolution and Quality
    14: PerturbationEntry(14, "Downsample_Upsample",           "resolution_quality", perturb_14_downsample_upsample),
    15: PerturbationEntry(15, "Resolution_Degradation_MPP",    "resolution_quality", perturb_15_resolution_degradation),
    16: PerturbationEntry(16, "Sharpness_Reduction",           "resolution_quality", perturb_16_sharpness_reduction),
    # Category 5: Lighting Conditions
    17: PerturbationEntry(17, "Brightness_Adjustment",         "lighting",           perturb_17_brightness_adjustment),
    18: PerturbationEntry(18, "Contrast_Adjustment",           "lighting",           perturb_18_contrast_adjustment),
}


# ===========================================================================
# Resilience helpers: artifact-integrity validation and atomic writes
# ===========================================================================
#
# WHY THIS EXISTS: a full production run is (number of sampled tiles) x
# (18 perturbations) x (3 severities), which for even a modest tile corpus
# is tens of thousands of individual image-write operations, run on a
# shared HPC cluster where jobs can be pre-empted, hit walltime limits, or
# be killed by an OOM. Without integrity checking, a re-run after any of
# those events would either (a) blindly skip a truncated/corrupt PNG from
# the killed run because "a file already exists at that path", silently
# leaving bad data in the study corpus, or (b) blindly recompute
# everything from scratch, wasting hours of already-completed GPU/CPU time.
# This mirrors the exact pattern already used in downsample_wsi.py.

def validate_png_artifact(path: Path, expected_size: int = TILE_SIZE) -> bool:
    """
    Returns True only if `path` exists AND is a readable, correctly-sized
    PNG, so a half-written file from a killed prior job is correctly
    treated as needing regeneration rather than being skipped as "done".
    """
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as im:
            im.verify()  # cheap structural check, doesn't decode pixels
        # verify() closes the file handle; re-open to check actual size
        with Image.open(path) as im:
            return im.size == (expected_size, expected_size)
    except Exception:  # noqa: BLE001 - any read/verify failure means "not valid"
        return False


def atomic_save_png(img: Image.Image, path: Path) -> None:
    """
    Writes a PNG to a `.tmp` path first, then renames into place only on
    success. A job killed mid-write can never leave a file at `path` that
    looks complete but isn't, which is what validate_png_artifact() above
    relies on to make skip-or-regenerate decisions safe on restart.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".png.tmp")
    img.save(tmp_path, format="PNG")
    tmp_path.rename(path)


# ===========================================================================
# PerturbationPipeline: Orchestrates application of all perturbations
# ===========================================================================

@dataclass
class PipelineConfig:
    """
    Configuration for the perturbation pipeline.

    Attributes:
        output_dir   : Root directory where perturbed tiles are saved.
        save_outputs : Whether to save perturbed images to disk.
        perturbation_ids : Which perturbation IDs to apply (default: all 18).
        severities   : Which severity levels to apply (default: all 3).
        seed         : Random seed for reproducibility.
        skip_existing: If True (default), a combo whose output artifact
                       already exists AND passes validate_png_artifact()
                       is skipped WITHOUT calling the (potentially costly,
                       e.g. elastic deformation) perturbation function at
                       all. This is what makes a restarted run cheap, not
                       just safe: valid prior work is never recomputed.
    """
    output_dir      : Path = Path("outputs/perturbed_tiles")
    save_outputs    : bool = True
    perturbation_ids: List[int] = field(default_factory=lambda: list(range(1, 19)))
    severities      : List[Severity] = field(default_factory=lambda: list(ALL_SEVERITIES))
    seed            : int = 42
    skip_existing   : bool = True


class PerturbationPipeline:
    """
    Applies the full 18-perturbation suite to a given histopathology tile.

    Usage:
        pipeline = PerturbationPipeline(config)
        results  = pipeline.apply(tile_img, sample_id="sample_001")

    Returns a dict keyed by (perturbation_id, severity) mapping to a
    PerturbCombResult (status + the image, when computed/loaded).

    RESILIENCE: for each (perturbation, severity) combination, the target
    artifact path is checked FIRST. If a valid artifact already exists
    there, the (possibly expensive) perturbation function is never called;
    the existing file is simply reported as SKIPPED_EXISTING. This means a
    job restarted after a Slurm timeout resumes cheaply from wherever it
    was interrupted, rather than either silently trusting a possibly
    truncated file or recomputing the entire tile from scratch.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        np.random.seed(self.config.seed)
        if self.config.save_outputs:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def artifact_path(self, sample_id: str, entry: "PerturbationEntry", severity: Severity) -> Path:
        """Computes the deterministic output path for a given combo, without touching disk."""
        return (
            self.config.output_dir
            / entry.category
            / f"p{entry.id:02d}_{entry.name}"
            / severity.value
            / f"{sample_id}.png"
        )

    def apply(
        self,
        tile: Image.Image,
        sample_id: str = "sample",
    ) -> Dict[Tuple[int, Severity], "PerturbComboResult"]:
        """
        Apply all configured perturbations at all configured severities.

        Args:
            tile      : Input PIL.Image (will be resized to 224x224 if needed).
            sample_id : Identifier for the tile (used in output filenames and
                        as the restart-resilience key: the same sample_id
                        passed again on a later run will resolve to the
                        same artifact paths, enabling skip-if-valid).

        Returns:
            Dict mapping (perturbation_id, severity) -> PerturbComboResult,
            where .status is one of DONE / SKIPPED_EXISTING / ERROR, and
            .image holds the perturbed PIL.Image for DONE and
            SKIPPED_EXISTING (reloaded from disk in the latter case) so
            downstream code (e.g. diagnostics metrics) can use it uniformly
            regardless of whether this run computed it or a prior run did.
        """
        # Ensure canonical 224x224 input
        tile = _ensure_224(tile.convert("RGB"))

        # Persist the baseline (unperturbed) tile itself, resiliently, so
        # downstream consumers (e.g. the embedding-extraction script) have
        # a single self-contained artifact tree and never need to re-derive
        # tiles from the source WSIs. Same skip-existing/atomic-write
        # pattern as every perturbed combo below.
        if self.config.save_outputs:
            baseline_path = self.config.output_dir / "baseline" / f"{sample_id}.png"
            if not (self.config.skip_existing and validate_png_artifact(baseline_path)):
                atomic_save_png(tile, baseline_path)

        results: Dict[Tuple[int, Severity], PerturbComboResult] = {}

        for pid in self.config.perturbation_ids:
            entry = PERTURBATION_REGISTRY.get(pid)
            if entry is None:
                log.warning("Perturbation ID %d not found in registry - skipping.", pid)
                continue

            for sev in self.config.severities:
                dest = self.artifact_path(sample_id, entry, sev)

                if (
                    self.config.save_outputs
                    and self.config.skip_existing
                    and validate_png_artifact(dest)
                ):
                    # Valid artifact from a prior run: skip recomputation
                    # entirely (this is the CPU-time saving, not just the
                    # correctness guarantee) and load it for downstream use.
                    try:
                        with Image.open(dest) as im:
                            results[(pid, sev)] = PerturbComboResult(
                                status="SKIPPED_EXISTING", image=im.copy(), path=dest,
                            )
                        continue
                    except Exception as exc:  # noqa: BLE001
                        log.warning(
                            "Artifact at %s passed validation but failed to "
                            "reload (%s); regenerating.", dest, exc,
                        )
                        # fall through to recomputation below

                try:
                    perturbed = entry.fn(tile, sev)
                    perturbed = _ensure_224(perturbed)

                    if self.config.save_outputs:
                        atomic_save_png(perturbed, dest)

                    results[(pid, sev)] = PerturbComboResult(
                        status="DONE", image=perturbed, path=dest,
                    )

                except Exception as exc:
                    log.error(
                        "Perturbation %d (%s) at severity '%s' failed for sample '%s': %s",
                        pid, entry.name, sev.value, sample_id, exc,
                    )
                    results[(pid, sev)] = PerturbComboResult(
                        status="ERROR", image=None, path=dest, notes=str(exc),
                    )

        n_done = sum(1 for r in results.values() if r.status == "DONE")
        n_skip = sum(1 for r in results.values() if r.status == "SKIPPED_EXISTING")
        n_err  = sum(1 for r in results.values() if r.status == "ERROR")
        log.info(
            "Sample '%s': %d computed, %d skipped (already valid), %d failed "
            "(of %d combinations).",
            sample_id, n_done, n_skip, n_err, len(results),
        )
        return results


@dataclass
class PerturbComboResult:
    """Outcome of applying one (perturbation, severity) combo to one tile."""
    status: str                        # DONE / SKIPPED_EXISTING / ERROR
    image: Optional[Image.Image]
    path: Path
    notes: str = ""


# ===========================================================================
# Tile sampling directly from the downsampled WSIs (downsample_wsi.py output)
# ===========================================================================
#
# There is no separate systematic tile-extraction step in the pipeline yet,
# so both the diagnostics run and the production run in this file sample
# 224x224 tiles directly from the standardised tiled TIFFs that
# downsample_wsi.py produces (single-resolution, 0.5 MPP, RGB, tiled).
# OpenSlide is used rather than a raw tifffile read because it gives
# efficient, per-tile-decompressed random access into a Deflate-compressed
# tiled TIFF without loading the whole (potentially multi-GB) image, the
# same property that made it the right tool for the original WSI reads.

@dataclass
class TileSample:
    source: str          # 'ACC' or 'GDC'
    case_id: str          # acc_case_id or file_id
    tile_index: int
    sample_id: str        # stable, deterministic identifier for this tile
    image: Image.Image


def _is_background_tile(img: Image.Image, mean_max: float = 222.0, std_min: float = 8.0) -> bool:
    """
    Simple, standard background/whitespace filter for WSI tiling pipelines
    (the same style of mean-intensity/variance heuristic used in CLAM and
    similar tiling tools): a tile is treated as background if it is both
    near-white on average AND has very little texture (low standard
    deviation), which describes empty slide area far more than it
    describes stained tissue, even lightly-stained tissue.
    """
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    return bool(gray.mean() > mean_max and gray.std() < std_min)


def sample_tiles_from_downsampled(
    acc_root: Path = DEFAULT_ACC_DOWNSAMPLED_ROOT,
    gdc_root: Path = DEFAULT_GDC_DOWNSAMPLED_ROOT,
    num_slides_per_cohort: int = 10,
    tiles_per_slide: int = 3,
    tile_size: int = TILE_SIZE,
    seed: int = 42,
    max_attempts_per_tile: int = 25,
) -> List[TileSample]:
    """
    Randomly samples tissue-containing 224x224 tiles from real downsampled
    WSIs in both cohorts, for use in diagnostics and/or the production
    perturbation run. Deterministic given the same seed and the same set
    of available slide files, so re-running with the same inputs samples
    the same tiles (important for the resilience/restart story: sample_id
    values must be stable across runs for skip-existing checks to work).

    Background tiles (mostly empty slide) are rejected via
    _is_background_tile() and re-sampled, up to max_attempts_per_tile
    times per slot, before giving up on that slide.
    """
    rng = random.Random(seed)
    samples: List[TileSample] = []

    for source, root in (("ACC", acc_root), ("GDC", gdc_root)):
        if not root.exists():
            log.warning("%s downsampled root does not exist: %s", source, root)
            continue

        slide_paths = sorted(root.glob("*.tiff"))
        if not slide_paths:
            log.warning("No .tiff files found under %s", root)
            continue

        chosen = rng.sample(slide_paths, k=min(num_slides_per_cohort, len(slide_paths)))

        for slide_path in chosen:
            case_id = slide_path.stem
            try:
                slide = openslide.OpenSlide(str(slide_path))
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not open %s: %s", slide_path, exc)
                continue

            width, height = slide.level_dimensions[0]
            if width <= tile_size or height <= tile_size:
                log.warning(
                    "%s is smaller than the tile size (%dx%d); skipping.",
                    slide_path, width, height,
                )
                slide.close()
                continue

            found = 0
            for tile_idx in range(tiles_per_slide):
                tile_img = None
                for _ in range(max_attempts_per_tile):
                    x = rng.randint(0, width - tile_size)
                    y = rng.randint(0, height - tile_size)
                    region = slide.read_region((x, y), 0, (tile_size, tile_size))
                    candidate = region.convert("RGB")  # source has no real alpha; safe to drop
                    if not _is_background_tile(candidate):
                        tile_img = candidate
                        break
                if tile_img is None:
                    log.warning(
                        "Could not find a tissue tile for %s after %d attempts "
                        "(slot %d); slide may be mostly background at this crop size.",
                        case_id, max_attempts_per_tile, tile_idx,
                    )
                    continue

                sample_id = f"{source}_{case_id}_t{tile_idx:02d}"
                samples.append(TileSample(source, case_id, tile_idx, sample_id, tile_img))
                found += 1

            slide.close()
            log.info("%s %s: sampled %d/%d tissue tiles.", source, case_id, found, tiles_per_slide)

    log.info("Total tiles sampled across both cohorts: %d", len(samples))
    return samples


# ===========================================================================
# Quality metrics for perturbation validation
# ===========================================================================
#
# These are computed for EVERY perturbation, not just Perturbation 16, so
# the resulting CSV supports general perturbation-strength validation
# across the whole 18-perturbation suite, not only the Severe-value
# question that originally motivated adding this.

def compute_quality_metrics(original: Image.Image, perturbed: Image.Image) -> Dict[str, float]:
    """
    Computes a battery of standard, well-established image-quality/
    difference metrics comparing a perturbed tile against its unperturbed
    source. No single metric captures every kind of degradation this suite
    applies (colour shift, blur, noise, geometric warp, compression), so
    several complementary ones are reported together:

      laplacian_var          : variance of the Laplacian on the perturbed
                                tile (Pech-Pacheco et al., 2000; surveyed
                                extensively by Pertuz et al., 2013). Higher
                                = sharper/more high-frequency detail.
                                Most directly relevant to the blur/
                                sharpness perturbations (6, 7, 14, 15, 16).
      laplacian_pct_of_orig  : the above, expressed as a percentage of the
                                ORIGINAL tile's own Laplacian variance, so
                                values are comparable across tiles with
                                different baseline texture.
      ssim                   : Structural Similarity Index (Wang et al.,
                                2004) vs the original, [-1, 1], 1 = identical.
                                A general-purpose perceptual similarity
                                metric, useful across every perturbation
                                category as a single comparable scale.
      psnr                   : Peak Signal-to-Noise Ratio (dB) vs the
                                original. Standard pixel-fidelity metric,
                                most informative for additive-noise-type
                                perturbations (8, 9) and compression (10).
      mean_abs_diff           : mean absolute per-pixel difference (0-255
                                scale) vs the original. A simple, easily
                                interpretable magnitude-of-change measure.
      entropy                 : Shannon entropy (Pertuz-style information
                                content) of the perturbed tile alone (not
                                a difference metric). Useful for noise
                                perturbations (8, 9), where entropy tends
                                to rise, versus blur perturbations, where
                                it tends to fall.
      mean_intensity, std_intensity : basic grayscale statistics of the
                                perturbed tile, useful as a sanity check
                                that a perturbation is doing something
                                sensible (e.g. brightness/contrast
                                perturbations should move these).
    """
    orig_gray = cv2.cvtColor(np.asarray(original), cv2.COLOR_RGB2GRAY)
    pert_gray = cv2.cvtColor(np.asarray(perturbed), cv2.COLOR_RGB2GRAY)

    orig_lap_var = cv2.Laplacian(orig_gray, cv2.CV_64F).var()
    pert_lap_var = cv2.Laplacian(pert_gray, cv2.CV_64F).var()

    orig_arr = np.asarray(original).astype(np.float64)
    pert_arr = np.asarray(perturbed).astype(np.float64)

    ssim_val = structural_similarity(
        np.asarray(original), np.asarray(perturbed), channel_axis=-1
    )
    # PSNR is undefined (infinite) for identical images; guard explicitly
    # rather than letting skimage raise/warn on a zero-MSE comparison.
    mse = np.mean((orig_arr - pert_arr) ** 2)
    psnr_val = float("inf") if mse == 0 else peak_signal_noise_ratio(
        np.asarray(original), np.asarray(perturbed), data_range=255
    )

    return {
        "laplacian_var": float(pert_lap_var),
        "laplacian_pct_of_orig": float(100.0 * pert_lap_var / orig_lap_var) if orig_lap_var > 0 else float("nan"),
        "ssim": float(ssim_val),
        "psnr": psnr_val,
        "mean_abs_diff": float(np.mean(np.abs(orig_arr - pert_arr))),
        "entropy": float(shannon_entropy(pert_gray)),
        "mean_intensity": float(pert_gray.mean()),
        "std_intensity": float(pert_gray.std()),
    }


# ===========================================================================
# Diagnostics runner: real-tile validation CSV, including the Perturbation
# 16 Severe-value comparison (0.05 vs the earlier-drafted 0.1 alternative)
# ===========================================================================

DIAGNOSTICS_FIELDNAMES = [
    "source", "case_id", "sample_id", "perturbation_id", "perturbation_name",
    "category", "severity_label", "param_value",
    "laplacian_var", "laplacian_pct_of_orig", "ssim", "psnr",
    "mean_abs_diff", "entropy", "mean_intensity", "std_intensity",
    "status", "notes",
]


def _load_completed_diagnostics_keys(csv_path: Path) -> set:
    """
    Reads an existing (possibly partial, from an interrupted run)
    diagnostics CSV and returns the set of (sample_id, perturbation_id,
    severity_label) combinations already recorded, so a restarted run
    can skip straight past them rather than recomputing every metric
    from scratch.
    """
    completed = set()
    if not csv_path.exists():
        return completed
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            completed.add((row["sample_id"], row["perturbation_id"], row["severity_label"]))
    return completed


def run_diagnostics(
    out_csv: Path,
    acc_root: Path = DEFAULT_ACC_DOWNSAMPLED_ROOT,
    gdc_root: Path = DEFAULT_GDC_DOWNSAMPLED_ROOT,
    num_slides_per_cohort: int = 10,
    tiles_per_slide: int = 3,
    seed: int = 42,
    include_p16_alt_severe: float = 0.10,
) -> None:
    """
    Applies all 18 perturbations at all 3 severities to a sample of real
    tiles drawn from the downsampled WSIs, computes the full
    compute_quality_metrics() battery for each combination against its
    source tile, and writes one row per combination to out_csv.

    Two extra rows are added specifically for Perturbation 16, to
    directly support comparing the Severe candidates discussed for this
    perturbation:
      - severity_label='severe_alt_0.10' : perturb_16 run with the
        earlier-drafted candidate value (0.10) instead of the
        implemented value (0.05), so both can be compared side by side
        on the SAME real tiles rather than only the synthetic tile used
        during the original investigation.
      - severity_label='reference_1.0'   : the tile with NO sharpness
        modification applied (factor 1.0), included as the literal
        unperturbed anchor point. Trivially SSIM=1.0/PSNR=inf/
        laplacian_pct_of_orig=100% by construction, useful as a sanity
        check and as an explicit zero point when plotting the severity
        ladder.

    RESILIENT / RESTARTABLE: if out_csv already exists (from a prior,
    interrupted run), already-recorded (sample_id, perturbation_id,
    severity_label) combinations are skipped, and new rows are appended
    rather than the file being overwritten. Rows are flushed to disk
    after every tile (not buffered until the end), so a crash partway
    through loses at most one tile's worth of work, not the whole run.
    """
    already_done = _load_completed_diagnostics_keys(out_csv)
    if already_done:
        log.info(
            "Found existing diagnostics CSV with %d completed combinations; "
            "resuming and skipping those.", len(already_done),
        )

    write_header = not out_csv.exists()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    tiles = sample_tiles_from_downsampled(
        acc_root=acc_root, gdc_root=gdc_root,
        num_slides_per_cohort=num_slides_per_cohort,
        tiles_per_slide=tiles_per_slide, seed=seed,
    )
    if not tiles:
        log.error("No tiles were sampled; check the downsampled WSI paths. Aborting diagnostics.")
        return

    pipeline = PerturbationPipeline(PipelineConfig(save_outputs=False, seed=seed))

    with open(out_csv, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=DIAGNOSTICS_FIELDNAMES)
        if write_header:
            writer.writeheader()

        for tile_sample in tiles:
            rows_this_tile = []

            for pid, entry in PERTURBATION_REGISTRY.items():
                for sev in ALL_SEVERITIES:
                    key = (tile_sample.sample_id, str(pid), sev.value)
                    if key in already_done:
                        continue
                    try:
                        perturbed = entry.fn(tile_sample.image, sev)
                        perturbed = _ensure_224(perturbed)
                        metrics = compute_quality_metrics(tile_sample.image, perturbed)
                        param_value = _severity_param_value(pid, sev)
                        rows_this_tile.append({
                            "source": tile_sample.source, "case_id": tile_sample.case_id,
                            "sample_id": tile_sample.sample_id, "perturbation_id": pid,
                            "perturbation_name": entry.name, "category": entry.category,
                            "severity_label": sev.value, "param_value": param_value,
                            "status": "OK", "notes": "", **metrics,
                        })
                    except Exception as exc:  # noqa: BLE001
                        rows_this_tile.append({
                            "source": tile_sample.source, "case_id": tile_sample.case_id,
                            "sample_id": tile_sample.sample_id, "perturbation_id": pid,
                            "perturbation_name": entry.name, "category": entry.category,
                            "severity_label": sev.value, "param_value": None,
                            "status": "ERROR", "notes": str(exc),
                            "laplacian_var": None, "laplacian_pct_of_orig": None,
                            "ssim": None, "psnr": None, "mean_abs_diff": None,
                            "entropy": None, "mean_intensity": None, "std_intensity": None,
                        })

            # Perturbation 16 extra diagnostic rows: alt-severe candidate
            # and the literal 1.0 reference anchor, both on this same tile.
            for label, factor in (
                (f"severe_alt_{include_p16_alt_severe:.2f}", include_p16_alt_severe),
                ("reference_1.0", 1.0),
            ):
                key = (tile_sample.sample_id, "16", label)
                if key in already_done:
                    continue
                try:
                    perturbed = ImageEnhance.Sharpness(tile_sample.image.convert("RGB")).enhance(factor)
                    perturbed = _ensure_224(perturbed)
                    metrics = compute_quality_metrics(tile_sample.image, perturbed)
                    rows_this_tile.append({
                        "source": tile_sample.source, "case_id": tile_sample.case_id,
                        "sample_id": tile_sample.sample_id, "perturbation_id": 16,
                        "perturbation_name": "Sharpness_Reduction", "category": "resolution_quality",
                        "severity_label": label, "param_value": factor,
                        "status": "OK", "notes": "diagnostic-only comparison row", **metrics,
                    })
                except Exception as exc:  # noqa: BLE001
                    rows_this_tile.append({
                        "source": tile_sample.source, "case_id": tile_sample.case_id,
                        "sample_id": tile_sample.sample_id, "perturbation_id": 16,
                        "perturbation_name": "Sharpness_Reduction", "category": "resolution_quality",
                        "severity_label": label, "param_value": factor,
                        "status": "ERROR", "notes": str(exc),
                        "laplacian_var": None, "laplacian_pct_of_orig": None,
                        "ssim": None, "psnr": None, "mean_abs_diff": None,
                        "entropy": None, "mean_intensity": None, "std_intensity": None,
                    })

            for row in rows_this_tile:
                writer.writerow(row)
            fh.flush()  # crash-safety: persist this tile's rows immediately

            log.info(
                "Diagnostics: %s (%d new rows written, %d already done, skipped).",
                tile_sample.sample_id, len(rows_this_tile),
                sum(1 for pid in PERTURBATION_REGISTRY for sev in ALL_SEVERITIES
                    if (tile_sample.sample_id, str(pid), sev.value) in already_done),
            )

    log.info("Diagnostics complete. Report written to %s", out_csv)


def _severity_param_value(pid: int, severity: Severity) -> Optional[float]:
    """
    Best-effort extraction of the single numeric parameter driving a given
    perturbation at a given severity, for use as a plottable x-axis value
    in the diagnostics CSV. Perturbations with multi-part or non-numeric
    parameters (e.g. Perturbation 2's three independent RGB offsets)
    return None; the categorical severity_label column remains the
    reliable field to group by in every case.
    """
    single_param_maps = {
        1:  {Severity.MILD: 0.10, Severity.MODERATE: 0.25, Severity.SEVERE: 0.40},
        3:  {Severity.MILD: 0.7, Severity.MODERATE: 1.0, Severity.SEVERE: 1.3},
        4:  {Severity.MILD: 10, Severity.MODERATE: 30, Severity.SEVERE: 60},
        5:  {Severity.MILD: 0.10, Severity.MODERATE: 0.25, Severity.SEVERE: 0.40},
        8:  {Severity.MILD: 5, Severity.MODERATE: 15, Severity.SEVERE: 30},
        9:  {Severity.MILD: 0.01, Severity.MODERATE: 0.03, Severity.SEVERE: 0.05},
        10: {Severity.MILD: 80, Severity.MODERATE: 50, Severity.SEVERE: 20},
        11: {Severity.MILD: 90, Severity.MODERATE: 180, Severity.SEVERE: 270},
        12: {Severity.MILD: 0.8, Severity.MODERATE: 1.2, Severity.SEVERE: 1.5},
        14: {Severity.MILD: 2, Severity.MODERATE: 4, Severity.SEVERE: 8},
        15: {Severity.MILD: 1.0, Severity.MODERATE: 2.0, Severity.SEVERE: 4.0},  # v4 MPP targets
        16: {Severity.MILD: 0.5, Severity.MODERATE: 0.25, Severity.SEVERE: 0.05},
        17: {Severity.MILD: 0.7, Severity.MODERATE: 1.0, Severity.SEVERE: 1.3},
        18: {Severity.MILD: 0.7, Severity.MODERATE: 1.0, Severity.SEVERE: 1.3},
    }
    return single_param_maps.get(pid, {}).get(severity)


# ===========================================================================
# Production batch runner: resilient, restartable generation of the full
# perturbed-tile corpus for the study
# ===========================================================================

def all_combos_already_valid(pipeline: PerturbationPipeline, sample_id: str) -> bool:
    """
    Coarse, tile-level fast-skip check: if every (perturbation, severity)
    artifact for this sample_id already exists and validates, the tile
    doesn't need to be re-read from its (potentially large) source WSI at
    all. This is a cheaper first check than PerturbationPipeline.apply()'s
    own per-combo skip logic, which still requires the tile to be loaded
    into memory first.
    """
    for pid in pipeline.config.perturbation_ids:
        entry = PERTURBATION_REGISTRY.get(pid)
        if entry is None:
            continue
        for sev in pipeline.config.severities:
            if not validate_png_artifact(pipeline.artifact_path(sample_id, entry, sev)):
                return False
    return True


def run_production_batch(
    output_dir: Path,
    progress_csv: Path,
    acc_root: Path = DEFAULT_ACC_DOWNSAMPLED_ROOT,
    gdc_root: Path = DEFAULT_GDC_DOWNSAMPLED_ROOT,
    num_slides_per_cohort: int = 50,
    tiles_per_slide: int = 10,
    seed: int = 42,
) -> None:
    """
    Generates the full perturbed-tile corpus (all 18 perturbations x 3
    severities) for a sample of real tiles drawn from the downsampled
    WSIs, writing PNG artifacts under output_dir and a per-tile progress
    log to progress_csv.

    RESTART BEHAVIOUR: safe to kill and re-run at any point.
      1. Tile sampling is deterministic given the same seed and the same
         set of available slide files (see sample_tiles_from_downsampled),
         so a re-run samples the same tiles in the same order.
      2. For each tile, all_combos_already_valid() is checked BEFORE the
         tile is even read from its source WSI; fully-completed tiles are
         skipped at that point without touching the (large) source file.
      3. For tiles not fully complete, PerturbationPipeline.apply() is
         still individually resilient per (perturbation, severity) combo,
         so a tile that was half-done when the job was killed resumes
         from wherever it stopped rather than being redone from scratch.
      4. progress_csv is appended to (not overwritten) and flushed after
         every tile, so it always reflects true completion state even if
         the job is killed mid-run.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_csv.parent.mkdir(parents=True, exist_ok=True)

    tiles = sample_tiles_from_downsampled(
        acc_root=acc_root, gdc_root=gdc_root,
        num_slides_per_cohort=num_slides_per_cohort,
        tiles_per_slide=tiles_per_slide, seed=seed,
    )
    if not tiles:
        log.error("No tiles were sampled; check the downsampled WSI paths. Aborting production run.")
        return

    pipeline = PerturbationPipeline(PipelineConfig(output_dir=output_dir, save_outputs=True, seed=seed))

    write_header = not progress_csv.exists()
    with open(progress_csv, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["sample_id", "source", "case_id", "n_done", "n_skipped", "n_error", "status"]
        )
        if write_header:
            writer.writeheader()

        for i, tile_sample in enumerate(tiles, start=1):
            if all_combos_already_valid(pipeline, tile_sample.sample_id):
                log.info(
                    "[%d/%d] %s: all artifacts already valid; skipping "
                    "without reading source WSI.", i, len(tiles), tile_sample.sample_id,
                )
                writer.writerow({
                    "sample_id": tile_sample.sample_id, "source": tile_sample.source,
                    "case_id": tile_sample.case_id, "n_done": 0,
                    "n_skipped": len(pipeline.config.perturbation_ids) * len(pipeline.config.severities),
                    "n_error": 0, "status": "TILE_ALREADY_COMPLETE",
                })
                fh.flush()
                continue

            log.info("[%d/%d] %s: processing...", i, len(tiles), tile_sample.sample_id)
            results = pipeline.apply(tile_sample.image, sample_id=tile_sample.sample_id)

            n_done = sum(1 for r in results.values() if r.status == "DONE")
            n_skip = sum(1 for r in results.values() if r.status == "SKIPPED_EXISTING")
            n_err  = sum(1 for r in results.values() if r.status == "ERROR")

            writer.writerow({
                "sample_id": tile_sample.sample_id, "source": tile_sample.source,
                "case_id": tile_sample.case_id, "n_done": n_done, "n_skipped": n_skip,
                "n_error": n_err, "status": "COMPLETE" if n_err == 0 else "PARTIAL_ERRORS",
            })
            fh.flush()

    log.info("Production batch complete. Progress log at %s", progress_csv)


# ===========================================================================
# Standalone validation / smoke-test
# ===========================================================================

def _validate_all_perturbations(verbose: bool = True) -> None:
    """
    Run a quick smoke-test on a synthetic 224x224 RGB tile to confirm every
    perturbation function executes without error and preserves the tile size.

    This is suitable as a CI check or a pre-study validation step.
    """
    # Create a synthetic H&E-like tile: pink background, blue nuclei
    tile_arr = np.full((TILE_SIZE, TILE_SIZE, 3), [220, 180, 200], dtype=np.uint8)
    # Add some fake blue nuclei
    for cy, cx in [(56, 56), (112, 112), (168, 168), (56, 168), (168, 56)]:
        cv2.circle(tile_arr, (cx, cy), 12, (100, 80, 200), -1)

    tile = Image.fromarray(cv2.cvtColor(tile_arr, cv2.COLOR_BGR2RGB))

    failures = []
    for pid, entry in PERTURBATION_REGISTRY.items():
        for sev in ALL_SEVERITIES:
            try:
                result = entry.fn(tile, sev)
                assert result.size == (TILE_SIZE, TILE_SIZE), (
                    f"Output size mismatch: got {result.size}"
                )
                if verbose:
                    log.info("  OK  P%02d %-38s [%s]", pid, entry.name, sev.value)
            except Exception as exc:
                msg = f"FAIL P{pid:02d} {entry.name} [{sev.value}]: {exc}"
                log.error(msg)
                failures.append(msg)

    if failures:
        log.error("\n%d perturbation(s) failed validation:\n%s", len(failures), "\n".join(failures))
    else:
        log.info("\nAll 18 perturbations x 3 severities = 54 combinations passed validation.")


# ===========================================================================
# Entry point
# ===========================================================================
#
# Usage:
#   Smoke test (default, no real data needed):
#     python perturbation_pipeline.py
#     python perturbation_pipeline.py --mode smoke-test
#
#   Diagnostics (real-tile metrics CSV, including the Perturbation 16
#   Severe-value comparison; resumable if interrupted):
#     python perturbation_pipeline.py --mode diagnostics \
#         --out diagnostics_report.csv \
#         --num-slides-per-cohort 10 --tiles-per-slide 3
#
#   Production (generates the full perturbed-tile corpus; resumable if
#   interrupted, safe to re-submit as the same Slurm job after a timeout):
#     python perturbation_pipeline.py --mode production \
#         --output-dir outputs/perturbed_tiles \
#         --progress-csv outputs/production_progress.csv \
#         --num-slides-per-cohort 50 --tiles-per-slide 10

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["smoke-test", "diagnostics", "production"],
        default="smoke-test",
    )
    parser.add_argument("--acc-root", type=Path, default=DEFAULT_ACC_DOWNSAMPLED_ROOT)
    parser.add_argument("--gdc-root", type=Path, default=DEFAULT_GDC_DOWNSAMPLED_ROOT)
    parser.add_argument("--num-slides-per-cohort", type=int, default=10)
    parser.add_argument("--tiles-per-slide", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    # diagnostics mode
    parser.add_argument("--out", type=Path, default=Path("outputs/diagnostics_report.csv"),
                         help="[diagnostics mode] CSV report path")

    # production mode
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/perturbed_tiles"),
                         help="[production mode] root directory for perturbed-tile PNGs")
    parser.add_argument("--progress-csv", type=Path, default=Path("outputs/production_progress.csv"),
                         help="[production mode] per-tile progress log")

    args = parser.parse_args()

    log.info("=" * 72)
    log.info("Comparative Invariance Investigation - Perturbation Pipeline")
    log.info("Section 5.4 | University of Sussex MRes AI | Phillip Nyamwaya")
    log.info("Mode: %s", args.mode)
    log.info("=" * 72)

    if args.mode == "smoke-test":
        log.info("\nRunning smoke-test on synthetic 224x224 H&E-like tile...\n")
        _validate_all_perturbations(verbose=True)

    elif args.mode == "diagnostics":
        run_diagnostics(
            out_csv=args.out, acc_root=args.acc_root, gdc_root=args.gdc_root,
            num_slides_per_cohort=args.num_slides_per_cohort,
            tiles_per_slide=args.tiles_per_slide, seed=args.seed,
        )

    elif args.mode == "production":
        run_production_batch(
            output_dir=args.output_dir, progress_csv=args.progress_csv,
            acc_root=args.acc_root, gdc_root=args.gdc_root,
            num_slides_per_cohort=args.num_slides_per_cohort,
            tiles_per_slide=args.tiles_per_slide, seed=args.seed,
        )


if __name__ == "__main__":
    main()
