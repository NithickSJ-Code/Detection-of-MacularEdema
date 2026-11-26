"""
validation.py — Input validity gate for the OCT Analysis Pipeline.

Called by inference.analyze_image() BEFORE running diagnosis.
Two-stage rejection:

  Stage 1 — Color check (always runs)
      OCT scans are near-grayscale.  Compute mean HSV saturation; reject if
      it exceeds COLOR_SAT_THRESHOLD (config.py).

  Stage 2 — Feature-space Mahalanobis check (runs only if ood_stats.npz exists)
      Extract the 1024-d DenseNet pooled feature vector, compute the minimum
      Mahalanobis distance to the per-class mean vectors (fitted on real OCT
      images by scripts/fit_ood.py), and reject if that distance exceeds the
      99th-percentile threshold stored in the npz.

If ood_stats.npz is missing, Stage 2 is skipped with a logged warning.

Public API
----------
    result = validate_image(pil_image, model=None)
    # result.valid  : bool
    # result.reason : str | None   (human-readable rejection reason)
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from src.config import (
    COLOR_SAT_THRESHOLD,
    FEATURE_DIM,
    IMAGE_SIZE,
    MAHAL_THRESHOLD_OVERRIDE,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    OOD_STATS_PATH,
)

log = logging.getLogger(__name__)

_transform = T.Compose([
    T.Resize(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
])


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    valid:  bool
    reason: Optional[str] = None   # populated only when valid=False


# ---------------------------------------------------------------------------
# Stage 1 — Color / saturation check
# ---------------------------------------------------------------------------

def _check_color(pil_image: Image.Image) -> ValidationResult:
    """
    Reject non-OCT images using TWO complementary grayscale checks:

    Check A — HSV saturation:
        OCT scans have very low saturation. Reject if mean HSV saturation
        exceeds COLOR_SAT_THRESHOLD.

    Check B — Per-pixel RGB channel spread:
        In a truly grayscale image, R ≈ G ≈ B for every pixel.
        We measure mean(max(R,G,B) - min(R,G,B)) across all pixels.
        Natural photos (faces, posters) have a high channel spread even under
        dim/washed-out lighting.  Reject if spread > COLOR_CHANNEL_SPREAD_THRESHOLD.

    A face photo held to the webcam will typically fail at least one of these.
    """
    from src.config import COLOR_CHANNEL_SPREAD_THRESHOLD

    img_rgb = np.array(pil_image.convert("RGB"), dtype=np.uint8).astype(np.float32)

    # --- Check A: HSV saturation ---
    img_uint8 = img_rgb.astype(np.uint8)
    img_hsv   = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
    mean_sat  = float(img_hsv[:, :, 1].mean())

    if mean_sat > COLOR_SAT_THRESHOLD:
        return ValidationResult(
            valid=False,
            reason=(
                f"Color check A failed: mean HSV saturation {mean_sat:.1f} "
                f"exceeds threshold {COLOR_SAT_THRESHOLD:.1f}. "
                "OCT scans are near-grayscale; this image appears to be a "
                "colour photograph or non-OCT image."
            ),
        )

    # --- Check B: per-pixel RGB channel spread ---
    r, g, b       = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    channel_max   = np.maximum(np.maximum(r, g), b)
    channel_min   = np.minimum(np.minimum(r, g), b)
    mean_spread   = float((channel_max - channel_min).mean())

    if mean_spread > COLOR_CHANNEL_SPREAD_THRESHOLD:
        return ValidationResult(
            valid=False,
            reason=(
                f"Color check B failed: mean per-pixel RGB channel spread "
                f"{mean_spread:.1f} exceeds threshold "
                f"{COLOR_CHANNEL_SPREAD_THRESHOLD:.1f}. "
                "This image has colour variation inconsistent with a grayscale "
                "OCT scan — please retake with the actual scan image."
            ),
        )

    return ValidationResult(valid=True)



# ---------------------------------------------------------------------------
# Stage 2 — Feature-space Mahalanobis distance check
# ---------------------------------------------------------------------------

def _extract_features(model: nn.Module, input_tensor: torch.Tensor) -> np.ndarray:
    """
    Extract the 1024-d pooled feature vector from before model.classifier.

    DenseNet-121 forward pass (simplified):
        features = model.features(x)          # conv features
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, 1)   # global avg pool → (B,1024,1,1)
        out = torch.flatten(out, 1)           # (B, 1024)
        out = model.classifier(out)

    We capture the output of model.features, then apply the remaining
    steps ourselves to get the 1024-d vector.
    """
    import torch.nn.functional as F

    captured: list[torch.Tensor] = []

    def _hook(_module, _inp, output):
        captured.append(output.detach())

    handle = model.features.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            model(input_tensor)   # full forward pass to fire the hook
    finally:
        handle.remove()

    feat = captured[0]                            # (1, 1024, 7, 7) approx
    feat = F.relu(feat, inplace=False)
    feat = F.adaptive_avg_pool2d(feat, (1, 1))    # (1, 1024, 1, 1)
    feat = torch.flatten(feat, 1)                 # (1, 1024)
    return feat.squeeze(0).cpu().numpy()           # (1024,)


def _load_ood_stats() -> Optional[dict]:
    """Load the ood_stats.npz file; return None if missing."""
    if not OOD_STATS_PATH.is_file():
        warnings.warn(
            f"[validation.py] OOD stats file not found at {OOD_STATS_PATH}. "
            "Skipping feature-space Mahalanobis check. "
            "Run `python -m scripts.fit_ood <oct_folder>` to generate it.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None
    data = np.load(OOD_STATS_PATH)
    return {
        "class_means":      data["class_means"],       # (n_classes, 1024)
        "precision_matrix": data["precision_matrix"],  # (1024, 1024)
        "threshold":        float(data["threshold"]),  # scalar
    }


def _mahalanobis_distance(
    x: np.ndarray,           # (D,)
    mean: np.ndarray,        # (D,)
    precision: np.ndarray,   # (D, D)
) -> float:
    """Mahalanobis distance: sqrt( (x-μ)^T Σ^{-1} (x-μ) )."""
    diff = x - mean
    dist = float(np.sqrt(diff @ precision @ diff))
    return dist


def _check_mahalanobis(
    model: nn.Module,
    input_tensor: torch.Tensor,
    stats: dict,
) -> ValidationResult:
    feat = _extract_features(model, input_tensor)   # (1024,)

    class_means = stats["class_means"]       # (n_classes, 1024)
    precision   = stats["precision_matrix"]  # (1024, 1024)
    threshold   = (
        MAHAL_THRESHOLD_OVERRIDE
        if MAHAL_THRESHOLD_OVERRIDE is not None
        else stats["threshold"]
    )

    # Minimum distance across all class centroids
    distances = [
        _mahalanobis_distance(feat, class_means[k], precision)
        for k in range(class_means.shape[0])
    ]
    min_dist = min(distances)

    log.debug(
        "[validation] Mahalanobis min_dist=%.3f  threshold=%.3f",
        min_dist, threshold,
    )

    if min_dist > threshold:
        return ValidationResult(
            valid=False,
            reason=(
                f"Feature-space check failed: Mahalanobis distance {min_dist:.2f} "
                f"exceeds threshold {threshold:.2f}. "
                "This image does not resemble the OCT scans the model was trained on."
            ),
        )
    return ValidationResult(valid=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_image(
    pil_image: Image.Image,
    model: Optional[nn.Module] = None,
) -> ValidationResult:
    """
    Run the two-stage input validity gate.

    Parameters
    ----------
    pil_image : PIL.Image (any mode).
    model     : Loaded nn.Module for Stage 2.  If None, Stage 2 is skipped
                even if ood_stats.npz exists.

    Returns
    -------
    ValidationResult with .valid (bool) and .reason (str | None).
    """
    # Stage 1 — always
    color_result = _check_color(pil_image)
    if not color_result.valid:
        return color_result

    # Stage 2 — only if model provided and stats file exists
    if model is not None:
        stats = _load_ood_stats()
        if stats is not None:
            img_rgb    = pil_image.convert("RGB").resize(
                (IMAGE_SIZE[1], IMAGE_SIZE[0])
            )
            input_tensor = _transform(img_rgb).unsqueeze(0).to(
                next(model.parameters()).device
            )
            mahal_result = _check_mahalanobis(model, input_tensor, stats)
            if not mahal_result.valid:
                return mahal_result

    return ValidationResult(valid=True)
