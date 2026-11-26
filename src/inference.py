"""
inference.py — High-level analysis pipeline.

analyze_image(pil_image) is the single public entry point.
It returns an AnalysisResult dataclass.

Two outcomes:
  result.status == "OK"            — normal diagnosis; all fields populated.
  result.status == "INVALID_INPUT" — image rejected by the validation gate;
                                     result.rejection_reason explains why;
                                     all diagnosis / image fields are None.

The module lazily loads the model on first call so that importing inference
does NOT trigger model loading or any other side effect.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from src.config import (
    CLASS_NAMES,
    CONFIDENCE_REVIEW_THRESHOLD,
    GRADCAM_THRESHOLD,
    IMAGE_SIZE,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
)

# ---------------------------------------------------------------------------
# Preprocessing transform  (FIXED — matches training pipeline)
# ---------------------------------------------------------------------------
_transform = T.Compose([
    T.Resize(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
])

# ---------------------------------------------------------------------------
# Lazy model singleton
# ---------------------------------------------------------------------------
_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:   # double-checked locking
                from src.model import load_model
                _model = load_model()
    return _model


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class AnalysisResult:
    """
    Returned by analyze_image() for every call.

    Always present
    --------------
    status           : "OK" or "INVALID_INPUT"
    rejection_reason : human-readable string when status == "INVALID_INPUT",
                       else None.

    Present only when status == "OK"
    ---------------------------------
    All other fields; None when status == "INVALID_INPUT".
    """

    # ---- Status (always present) ----------------------------------------
    status:              str             = "OK"
    rejection_reason:    Optional[str]   = None

    # ---- Classification (OK only) ----------------------------------------
    predicted_label:     Optional[str]   = None
    confidence:          Optional[float] = None   # softmax of predicted class
    per_class_probs:     Optional[dict]  = None   # {class_name: float}

    # ---- Uncertainty (OK only) -------------------------------------------
    uncertainty_std:     Optional[float] = None
    uncertainty_entropy: Optional[float] = None
    uncertainty_level:   Optional[str]   = None   # LOW / MODERATE / HIGH
    mc_passes:           Optional[int]   = None

    # ---- Flags (OK only) -------------------------------------------------
    review_recommended:  Optional[bool]  = None

    # ---- Images  (RGB uint8 numpy arrays, shape (224,224,3)) (OK only) ---
    original_image:      Optional[np.ndarray] = None
    heatmap_overlay:     Optional[np.ndarray] = None
    overlay_with_box:    Optional[np.ndarray] = None

    # ---- Grad-CAM raw map  float32 (224,224) in [0,1] (OK only) ----------
    cam_array:           Optional[np.ndarray] = None

    # ---- Bounding box (x,y,w,h) or None (OK only) ------------------------
    box_coords:          Optional[Tuple[int, int, int, int]] = None


def _invalid(reason: str) -> AnalysisResult:
    """Convenience constructor for a rejected result."""
    return AnalysisResult(status="INVALID_INPUT", rejection_reason=reason)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _build_visualisations(
    img_rgb: np.ndarray,    # (224,224,3) uint8 RGB
    cam: np.ndarray,        # (224,224) float32 [0,1]
) -> tuple[np.ndarray, np.ndarray, Optional[tuple]]:
    """
    Returns (heatmap_overlay, overlay_with_box, box_coords).
    All images are RGB uint8.
    """
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    overlay = (0.5 * img_rgb + 0.5 * heatmap_rgb).astype(np.uint8)

    thresh = (cam > GRADCAM_THRESHOLD).astype(np.uint8)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay_box = overlay.copy()
    box_coords = None
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        box_coords = (x, y, w, h)
        cv2.rectangle(overlay_box, (x, y), (x + w, y + h), (255, 0, 0), 3)

    return overlay, overlay_box, box_coords


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_image(pil_image: Image.Image) -> AnalysisResult:
    """
    Full pipeline:
        validation gate → preprocessing → Grad-CAM → MC Dropout → AnalysisResult.

    Parameters
    ----------
    pil_image : PIL.Image in any mode; converted to RGB internally.

    Returns
    -------
    AnalysisResult with status == "OK" on success, or
    status == "INVALID_INPUT" (with rejection_reason set) if the image
    fails the input validity gate.  When rejected, no diagnosis is produced
    and no PDF should be generated.
    """
    from src.gradcam import GradCAM
    from src.uncertainty import mc_dropout_predict
    from src.validation import validate_image

    model = _get_model()

    # ---- Stage 0: Validation gate ------------------------------------
    val = validate_image(pil_image, model=model)
    if not val.valid:
        return _invalid(val.reason)

    device = next(model.parameters()).device

    # ---- Preprocessing ------------------------------------------------
    img_rgb     = pil_image.convert("RGB")
    img_resized = img_rgb.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]))   # PIL: (W,H)
    img_array   = np.array(img_resized, dtype=np.uint8)            # (224,224,3)

    input_tensor = _transform(img_resized).unsqueeze(0).to(device) # (1,3,224,224)

    # ---- Deterministic Grad-CAM pass (dropout OFF) --------------------
    gradcam = GradCAM(model, model.features.denseblock4.denselayer16.conv2)
    cam_array, pred_class_idx = gradcam.generate(input_tensor)
    gradcam._remove_hooks()

    predicted_label = CLASS_NAMES[pred_class_idx]

    with torch.no_grad():
        logits    = model(input_tensor)
        probs_det = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    confidence      = float(probs_det[pred_class_idx])
    per_class_probs = {cls: float(probs_det[i]) for i, cls in enumerate(CLASS_NAMES)}

    # ---- MC Dropout uncertainty pass (dropout ON) ---------------------
    mc_result           = mc_dropout_predict(model, input_tensor)
    uncertainty_std     = mc_result["std_pred"]
    uncertainty_entropy = mc_result["entropy"]
    uncertainty_level   = mc_result["uncertainty_level"]
    mc_passes           = mc_result["mc_passes"]

    # ---- Review flag --------------------------------------------------
    review_recommended = (
        confidence < CONFIDENCE_REVIEW_THRESHOLD
        or uncertainty_level == "HIGH"
    )

    # ---- Visualisation ------------------------------------------------
    heatmap_overlay, overlay_with_box, box_coords = _build_visualisations(
        img_array, cam_array
    )

    return AnalysisResult(
        status="OK",
        predicted_label=predicted_label,
        confidence=confidence,
        per_class_probs=per_class_probs,
        uncertainty_std=uncertainty_std,
        uncertainty_entropy=uncertainty_entropy,
        uncertainty_level=uncertainty_level,
        mc_passes=mc_passes,
        review_recommended=review_recommended,
        original_image=img_array,
        heatmap_overlay=heatmap_overlay,
        overlay_with_box=overlay_with_box,
        cam_array=cam_array,
        box_coords=box_coords,
    )
