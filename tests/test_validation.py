"""
tests/test_validation.py — Unit tests for src/validation.py

These tests are intentionally lightweight: they use synthetic PIL images and
do NOT load the AI model.  This means only Stage 1 (color check) is exercised
by default; Stage 2 (Mahalanobis) requires ood_stats.npz and a loaded model,
so it is tested with the model=None shortcut which skips Stage 2 entirely.

Run with:
    python -m pytest tests/test_validation.py -v

Or without pytest:
    python tests/test_validation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Make sure the project root is on sys.path so `src` is importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.validation import validate_image, _check_color
from src.config import COLOR_SAT_THRESHOLD


# ---------------------------------------------------------------------------
# Synthetic image helpers
# ---------------------------------------------------------------------------

def _make_colorful_image(size=(224, 224)) -> Image.Image:
    """
    A vivid, highly saturated colour image (e.g. a face photo or red/green
    pattern).  Mean HSV saturation will be far above COLOR_SAT_THRESHOLD.
    """
    arr = np.zeros((*size, 3), dtype=np.uint8)
    # Top half: bright red
    arr[:size[0] // 2, :, 0] = 220
    arr[:size[0] // 2, :, 1] = 20
    arr[:size[0] // 2, :, 2] = 20
    # Bottom half: bright green
    arr[size[0] // 2:, :, 0] = 20
    arr[size[0] // 2:, :, 1] = 200
    arr[size[0] // 2:, :, 2] = 20
    return Image.fromarray(arr, mode="RGB")


def _make_grayscale_image(size=(224, 224), value: int = 128) -> Image.Image:
    """
    A flat grayscale image (constant luminance).
    RGB channels are equal → saturation = 0 in HSV space.
    """
    arr = np.full((*size, 3), value, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _make_noisy_grayscale_image(size=(224, 224)) -> Image.Image:
    """
    Grayscale image with mild Gaussian noise — resembles a real OCT scan.
    Saturation will still be very low (< 5/255 on average).
    """
    rng  = np.random.default_rng(42)
    base = rng.integers(60, 180, size=(*size,), dtype=np.uint8)
    arr  = np.stack([base, base, base], axis=-1)
    return Image.fromarray(arr, mode="RGB")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_colorful_image_is_rejected():
    """A bright, saturated colour image must fail the color check."""
    img = _make_colorful_image()
    result = validate_image(img, model=None)   # model=None → skip Stage 2
    assert not result.valid, (
        "Expected colorful image to be REJECTED, but it was accepted."
    )
    assert result.reason is not None
    assert "Color check" in result.reason, (
        f"Unexpected rejection reason: {result.reason}"
    )
    print(f"[PASS] Colorful image correctly rejected. Reason: {result.reason[:80]}")


def test_flat_grayscale_image_is_accepted():
    """A pure flat-grey image (zero saturation) must pass the color check."""
    img = _make_grayscale_image(value=100)
    result = validate_image(img, model=None)
    assert result.valid, (
        f"Expected flat-grey image to be ACCEPTED, but it was rejected: "
        f"{result.reason}"
    )
    print("[PASS] Flat grayscale image correctly accepted.")


def test_noisy_grayscale_image_is_accepted():
    """A grayscale image with noise (like a real OCT) must pass the color check."""
    img = _make_noisy_grayscale_image()
    result = validate_image(img, model=None)
    assert result.valid, (
        f"Expected noisy-grey image to be ACCEPTED, but it was rejected: "
        f"{result.reason}"
    )
    print("[PASS] Noisy grayscale image correctly accepted.")


def test_color_check_threshold_boundary():
    """
    Verify that an image with saturation clearly below the threshold is accepted.

    Note: testing sat == threshold exactly is unreliable because the
    HSV -> RGB -> HSV round-trip involves uint8 quantisation that introduces
    a +-2 rounding error per channel.  We therefore use a value that is
    safely below threshold after round-trip to avoid flaky results.
    """
    import cv2
    h, w = 224, 224
    # Use a saturation well below the threshold to survive rounding
    safe_sat = max(0, int(COLOR_SAT_THRESHOLD) - 10)
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[:, :, 0] = 10
    hsv[:, :, 1] = safe_sat
    hsv[:, :, 2] = 200
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    img = Image.fromarray(rgb, mode="RGB")

    result = _check_color(img)
    assert result.valid, (
        f"Image with safe_sat={safe_sat} (well below threshold "
        f"{COLOR_SAT_THRESHOLD}) should be ACCEPTED. "
        f"Got rejected: {result.reason}"
    )
    print(f"[PASS] Boundary image (safe_sat={safe_sat}) correctly accepted.")



def test_high_saturation_image_is_rejected():
    """An image with saturation clearly above threshold must be rejected."""
    import cv2
    h, w = 224, 224
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[:, :, 1] = 200  # very high saturation
    hsv[:, :, 2] = 200
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    img = Image.fromarray(rgb, mode="RGB")

    result = _check_color(img)
    assert not result.valid, "High-saturation image should be REJECTED."
    print(f"[PASS] High-saturation image correctly rejected. Reason: {result.reason[:80]}")


def test_pil_mode_L_accepted():
    """Single-channel greyscale PIL image (mode L) must also be accepted."""
    arr = np.random.default_rng(0).integers(0, 255, (224, 224), dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    result = validate_image(img, model=None)
    assert result.valid, f"Mode-L grayscale image should be accepted. Got: {result.reason}"
    print("[PASS] Mode-L grayscale image correctly accepted.")


# ---------------------------------------------------------------------------
# Runner (no pytest required)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_colorful_image_is_rejected,
        test_flat_grayscale_image_is_accepted,
        test_noisy_grayscale_image_is_accepted,
        test_color_check_threshold_boundary,
        test_high_saturation_image_is_rejected,
        test_pil_mode_L_accepted,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {t.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed.")
    sys.exit(0 if failed == 0 else 1)
