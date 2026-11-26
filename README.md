# OCT Macular Edema Detection — DenseNet-121 + Grad-CAM

**Early Detection of Macular Edema in Retinal OCT Scans using Deep Learning with Grad-CAM Localization**

> Aravind Eye Hospital Research Collaboration — Team 05, IV CSE-A

---

## ⚠️ Disclaimer

**RESEARCH PROTOTYPE ONLY.**
This software is **not a medical device** and is **not approved for clinical diagnosis**.
All outputs must be reviewed and confirmed by a qualified ophthalmologist before any clinical decision is made.

---

## Overview

This project implements a weakly-supervised classification and localization pipeline for retinal OCT scans.  
Key components:

| Component | Description |
|---|---|
| **DenseNet-121** | Classifier trained on the Kermany 2018 dataset (83 k images, 4 classes) |
| **Grad-CAM** | Localizes disease regions without pixel-level annotations |
| **Monte Carlo Dropout** | Approximates uncertainty via stochastic forward passes |
| **PDF Report** | Per-scan A4 report with images, table, interpretation, and disclaimer |

---

## Project Structure

```
project_root/
│
├── densenet_pathwayA_HARDENED.pth   # Trained model weights (DO NOT MODIFY)
│
├── src/                             # Importable Python package
│   ├── __init__.py
│   ├── config.py        ← All constants (paths, class names, thresholds)
│   ├── model.py         ← load_model(path, device)
│   ├── gradcam.py       ← GradCAM class (hooks cleaned after use)
│   ├── uncertainty.py   ← MC Dropout via forward pre-hook
│   ├── inference.py     ← analyze_image(pil) → AnalysisResult
│   └── report.py        ← generate_pdf / save_pdf
│
├── scripts/
│   ├── __init__.py
│   └── analyze_image.py ← CLI: python -m scripts.analyze_image <image>
│
├── reports/             ← Output directory for generated PDFs
│
├── live_camera_scanner.py   ← Interactive webcam demo (refactored)
├── upload_scanner.py        ← File-upload variant (Tkinter dialog)
├── requirements.txt
├── START_LIVE_CAMERA.bat
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Analyze a single image (CLI)

```bash
# Basic — prints results to console
python -m scripts.analyze_image path/to/scan.png

# Save PDF report
python -m scripts.analyze_image scan.png --pdf reports/result.pdf

# Save PDF with patient ID and display result window
python -m scripts.analyze_image scan.png --pdf reports/out.pdf --patient-id P-2026-001 --show
```

### 3. Live webcam demo

```bash
python live_camera_scanner.py
# or double-click START_LIVE_CAMERA.bat
```

| Key | Action |
|---|---|
| `SPACE` | Capture scan in green box → analyze → save PDF |
| `Q` | Quit |

### 4. File upload (no webcam required)

```bash
python upload_scanner.py
```

A file dialog will open. Select any `.png` / `.jpg` / `.bmp` OCT image.

---

## Classes

| Index | Class | Description |
|---|---|---|
| 0 | **CNV** | Choroidal Neovascularisation |
| 1 | **DME** | Diabetic Macular Edema |
| 2 | **DRUSEN** | Drusen deposits |
| 3 | **NORMAL** | Healthy macula |

---

## Technical Details

### Model
- Architecture: `torchvision.models.densenet121(weights=None)`
- Classifier head replaced: `nn.Linear(1024, 4)`
- Checkpoint: `densenet_pathwayA_HARDENED.pth`
- Input preprocessing: `Resize(224,224) → ToTensor → Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])`
- Default device: CPU; CUDA used automatically if available

### Grad-CAM
- Target layer: `model.features.denseblock4.denselayer16.conv2`
- Post-processing: ReLU → min-max normalize → bilinear upsample to 224×224
- Bounding box: largest contour from threshold at 0.5

### Monte Carlo Dropout  ⚠️ Approximation Only

The hardened checkpoint was trained with **no dropout** in the classifier.  
To obtain a stochastic signal we inject dropout via a `register_forward_pre_hook` on `model.classifier`:

```python
# Conceptual — see src/uncertainty.py
F.dropout(x, p=MC_P, training=True)   # injected at inference time only
```

**Consequences:**
- BatchNorm layers remain in `eval()` mode (correct — they use running statistics).
- The injected dropout rate (`MC_P = 0.3`) was **not** present during training.
- Uncertainty scores are **uncalibrated approximations**.
- Thresholds (`UNCERTAINTY_LOW_MAX`, `UNCERTAINTY_MODERATE_MAX` in `config.py`) are **placeholder values** and must be calibrated against clinical data before any real use.

### Uncertainty Levels

| Level | Condition (std of predicted-class prob) |
|---|---|
| LOW | std < 0.05 |
| MODERATE | 0.05 ≤ std < 0.15 |
| HIGH | std ≥ 0.15 |

Human review is recommended when:
- Confidence < 75%, or
- Uncertainty level = HIGH

---

## PDF Report Contents

Each report (one A4 page) includes:

1. **Header** — project title, institution, timestamp, optional patient/scan ID
2. **Three images** — Original scan | Grad-CAM overlay | Overlay + bounding box
3. **Results table** — predicted class, confidence, per-class probabilities, uncertainty std, entropy, uncertainty level, MC pass count, model name
4. **Interpretation** — plain-language description of the predicted class
5. **Review warning** — shown in red if confidence is low or uncertainty is high
6. **Footer disclaimer** — research prototype, not for clinical use

---

## Team

| Name | Roll No. | Role |
|---|---|---|
| Nishant J S | 714025104171 | AI Architecture & MC Dropout |
| Nishant J R | 714025104170 | Results, Evaluation & Testing |
| Nithin S B | 714025104179 | System Integration & Documentation |
| Nithick S J | 714025104178 | Data Pipeline & Flow |

**Guide:** Mrs. Deepthi Nair, AP/CSE  
**Institution:** IV CSE-A — Batch 05

---

## Dataset

Kermany, D.S. et al. (2018). *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning.* Cell, 172(5), 1122–1131.  
Dataset: [Kaggle — Retinal OCT Images](https://www.kaggle.com/datasets/paultimothymooney/kermany2018)

---

## Input Validation / OOD Rejection

The model gives high-confidence predictions on non-OCT images (e.g. face photos, screenshots).
`src/validation.py` adds a two-stage gate that runs **before** diagnosis.

### Stage 1 — Color / Saturation Check (always active)

OCT scans are near-grayscale.  The gate converts the image to HSV and measures
mean saturation.  Images with `mean_saturation > COLOR_SAT_THRESHOLD` (default: **40/255**)
are rejected immediately as non-OCT.

Tune `COLOR_SAT_THRESHOLD` in `src/config.py` — increase it to be more permissive,
decrease to be stricter.

### Stage 2 — Feature-Space Mahalanobis Check (requires `models/ood_stats.npz`)

Extracts the 1024-d DenseNet pooled feature vector and computes the minimum
Mahalanobis distance to the per-class centroids fitted on real OCT images.
If that distance exceeds the stored 99th-percentile threshold, the image is rejected.

**Fitting the OOD statistics** (one-time step, needs your OCT training set):

```bash
python -m scripts.fit_ood path/to/kermany_train_folder --max-per-class 2000
# Writes models/ood_stats.npz
```

Expected folder layout:

```
kermany_train_folder/
    CNV/      *.jpeg
    DME/
    DRUSEN/
    NORMAL/
```

If `models/ood_stats.npz` is **missing**, Stage 2 is silently skipped with a
`RuntimeWarning` and only Stage 1 runs.

### Behavior on Rejection

When either check fails, `analyze_image()` returns:

```python
result.status           == "INVALID_INPUT"
result.rejection_reason == "Color check failed: mean HSV saturation ..."
# All diagnosis fields (predicted_label, confidence, ...) are None
```

- **CLI**: prints `Not a valid OCT scan, please retake.` and exits with code 2.
- **Live camera**: displays `NOT A VALID OCT SCAN` in red on screen for 2 seconds.
- **No PDF is generated** for rejected images.

### Known Limitation

> **Stage 2 is calibrated on in-distribution OCT images only.**
> The shared covariance matrix is 1024×1024 with ridge regularisation (default λ=1e-4).
> For datasets much smaller than the feature dimension, the covariance estimate
> is unreliable and the threshold will need manual adjustment via
> `MAHAL_THRESHOLD_OVERRIDE` in `src/config.py`.
>
> **Fix / mitigation**: collect at least ~500 images per class before running
> `fit_ood.py`.  Alternatively, use PCA to reduce the feature dimension before
> fitting the Gaussian (future work).

---

## Web application

The original CLI and Tkinter demos are unchanged. A FastAPI + React layer now wraps `src.inference.analyze_image`.

### Setup (Windows)

Python **3.12** is required (Torch is not installed for the 3.14 default). From the project root:

```bat
run_app.bat
```

This starts the API at `http://127.0.0.1:8000` and the UI at `http://localhost:5173`.

Manual equivalent:

```bat
py -3.12 -m venv .venv --system-site-packages
.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-api.txt
.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

```bat
cd frontend
npm install
npm run dev
```

Optional single-process demo after `cd frontend && npm run build`:

```bat
set OCT_SERVE_FRONTEND=1
.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000`.

### API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Model status, device, OOD Stage 2 flag |
| POST | `/api/analyze` | Multipart `image` + optional `scan_id` / `source` |
| GET | `/api/scans` | List / search / filter |
| GET | `/api/scans/{id}` | One scan |
| DELETE | `/api/scans/{id}` | Delete row, images, PDF |
| GET | `/api/stats` | Dashboard / analytics numbers |
| GET | `/api/reports` | PDF list |
| GET | `/api/scans/{id}/report.pdf` | Inline PDF (`?download=1` to attach) |
| GET | `/api/scans/{id}/images/{original\|gradcam\|boxed}.png` | Result images |

SQLite lives at `backend/storage/oct.db`. Images are stored under `backend/storage/scans/<uuid>/`. PDFs still go to `reports/`.

See `DEMO_SCRIPT.md` for a five-minute viva walkthrough.

### API tests

```bat
.venv\Scripts\python.exe -m pytest backend/tests -v -m "not slow"
```

---

## Tests

```bash
# Run all validation unit tests (no model loading required)
python tests/test_validation.py

# Or with pytest
python -m pytest tests/ -v
```

| Test | What it checks |
|---|---|
| `test_colorful_image_is_rejected` | Bright red/green image is rejected by Stage 1 |
| `test_flat_grayscale_image_is_accepted` | Pure grey (sat=0) passes Stage 1 |
| `test_noisy_grayscale_image_is_accepted` | OCT-like noisy grey passes Stage 1 |
| `test_color_check_threshold_boundary` | Image well below threshold is accepted |
| `test_high_saturation_image_is_rejected` | sat=200 image is rejected |
| `test_pil_mode_L_accepted` | Single-channel (mode L) image is accepted |


