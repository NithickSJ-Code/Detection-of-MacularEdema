"""
config.py — Central configuration for the OCT Analysis Pipeline.
All tunable constants live here. Import this module; never hardcode these
values in other modules.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Resolve relative to this file so the package works regardless of CWD.
_HERE = Path(__file__).resolve().parent.parent   # project root

MODEL_PATH: Path = _HERE / "densenet_pathwayA_HARDENED.pth"
REPORTS_DIR: Path = _HERE / "reports"

# ---------------------------------------------------------------------------
# Model / data constants  (FIXED — do NOT change without retraining)
# ---------------------------------------------------------------------------
CLASS_NAMES: list[str] = ["CNV", "DME", "DRUSEN", "NORMAL"]
IMAGE_SIZE: tuple[int, int] = (224, 224)          # (H, W) expected by DenseNet

NORMALIZE_MEAN: list[float] = [0.485, 0.456, 0.406]
NORMALIZE_STD:  list[float] = [0.229, 0.224, 0.225]

# Grad-CAM
GRADCAM_THRESHOLD: float = 0.5    # cam values above this define the bounding box

# ---------------------------------------------------------------------------
# Monte Carlo Dropout
# ---------------------------------------------------------------------------
MC_T: int   = 20       # number of stochastic forward passes
MC_P: float = 0.3      # dropout probability injected via pre-hook

# Uncertainty thresholds (UNCALIBRATED — see README)
# These map std-dev of predicted-class probability to a human-readable level.
UNCERTAINTY_LOW_MAX:      float = 0.05   # std < this  → LOW
UNCERTAINTY_MODERATE_MAX: float = 0.15  # std < this  → MODERATE (else HIGH)

# Confidence threshold below which "human review recommended" appears in PDF
CONFIDENCE_REVIEW_THRESHOLD: float = 0.75

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
MODEL_NAME: str = "DenseNet-121 (Hardened — Pathway A)"
PROJECT_TITLE: str = (
    "Early Detection of Macular Edema in OCT Scans "
    "using Deep Learning with Grad-CAM Localization"
)
INSTITUTION: str = "Aravind Eye Hospital Research Collaboration"

# ---------------------------------------------------------------------------
# Input validation / OOD rejection
# ---------------------------------------------------------------------------
OOD_STATS_PATH: Path = _HERE / "models" / "ood_stats.npz"

# Color check — OCT scans are near-grayscale.
# Check A: Mean HSV saturation in [0, 255]; reject if above this.
COLOR_SAT_THRESHOLD: float = 30.0   # lowered from 40 — face photos can sneak under 40

# Check B: Mean per-pixel RGB channel spread = mean(max(R,G,B) - min(R,G,B)).
# In a truly grayscale image this is close to 0.
# A face photo has a spread of ~25-60 even under dim/washed-out webcam lighting.
COLOR_CHANNEL_SPREAD_THRESHOLD: float = 18.0  # out of 255

# Mahalanobis distance threshold for feature-space OOD check.
# None means "use the 99th-percentile value stored in ood_stats.npz".
# Set to a float to override the fitted threshold (useful during development).
MAHAL_THRESHOLD_OVERRIDE: float | None = None

# Dimensionality of the feature vector (DenseNet-121 pooled output).
FEATURE_DIM: int = 1024
