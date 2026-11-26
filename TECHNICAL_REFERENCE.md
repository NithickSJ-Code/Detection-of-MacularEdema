# ARAVIND EYE HOSPITAL — OCT MACULAR EDEMA DETECTION
## Complete Technical Reference Document
### Team 05 | IV CSE-A | Batch 2025–2026

---

## 1. PROJECT IDENTITY

| Field | Value |
|---|---|
| Full Title | Early Detection of Macular Edema in Retinal OCT Scans using Deep Learning with Grad-CAM Localization |
| Collaboration | Aravind Eye Hospital, Madurai |
| Guide | Mrs. Deepthi Nair, AP/CSE |
| Team | Nishant J S (714025104171), Nishant J R (714025104170), Nithin S B (714025104179), Nithick S J (714025104178) |
| Checkpoint File | `densenet_pathwayA_HARDENED.pth` — 28.45 MB |
| Model Architecture | DenseNet-121 (torchvision, pretrained on ImageNet, classifier head replaced) |

---

## 2. PROBLEM STATEMENT

Macular Edema (ME) is the accumulation of extracellular fluid in the macula — the central 5.5 mm region of the retina responsible for high-acuity central vision. Fluid accumulation creates intra-retinal and sub-retinal cysts visible in OCT cross-sections as hyporeflective (dark) pockets. Untreated, it causes irreversible central vision loss.

**Clinical gap:** Aravind Eye Hospital processes thousands of OCT scans per month. Manual grading by an ophthalmologist takes 2–5 minutes per scan. AI automation is needed.

**AI gap:** State-of-the-art segmentation models (U-Net, nnU-Net) require pixel-level annotation masks drawn by clinicians. These masks are proprietary, expensive, and unavailable at scale. Our approach achieves localization without any masks, using only image-level class labels.

---

## 3. DATASET

### 3.1 Kermany 2018 OCT Dataset
- **Source:** Kaggle — `paultimothymooney/kermany2018`
- **Paper:** Kermany, D.S. et al. (2018). *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning.* Cell, 172(5), 1122–1131.
- **Total images:** 83,484 retinal OCT B-scans (2D cross-sections)
- **Format:** JPEG, variable resolution, converted to 224×224 px
- **Classes (4):**

| Index | Class | Full Name | Description |
|---|---|---|---|
| 0 | CNV | Choroidal Neovascularisation | Abnormal blood vessel growth beneath RPE; associated with wet AMD |
| 1 | DME | Diabetic Macular Edema | Fluid leakage from diabetic damage to retinal capillaries |
| 2 | DRUSEN | Drusen | Yellow lipid deposits beneath RPE; marker of early dry AMD |
| 3 | NORMAL | Normal Macula | No pathological findings |

### 3.2 Dataset Split
| Split | Approximate Count |
|---|---|
| Train | ~65,000 |
| Validation | ~8,000 |
| Test | ~10,000 |

### 3.3 Known Dataset Limitation
Multiple published studies (including our own literature review) confirm that publicly available OCT datasets on Kaggle, Zenodo, and FigShare largely contain the same images redistributed across multiple uploads. This makes cross-dataset validation difficult without access to hospital-proprietary data.

---

## 4. MODEL ARCHITECTURE

### 4.1 DenseNet-121 (Densely Connected Convolutional Networks)

**Reference:** Huang et al. (2017). *Densely Connected Convolutional Networks.* CVPR 2017.

**Why DenseNet over ResNet/VGG:**
- Every layer receives feature maps from ALL preceding layers (dense connections).
- Gradients flow directly to all layers → no vanishing gradient.
- Feature reuse: each layer uses only `k` new feature maps (growth rate k=32 for DenseNet-121).
- 7.98M parameters vs ResNet-50's 25.5M — more parameter-efficient for subtle medical imaging features.

**Architecture breakdown:**

```
Input: (1, 3, 224, 224)
│
├── Initial Conv: 7×7, stride=2, 64 filters → (1, 64, 112, 112)
├── MaxPool: 3×3, stride=2 → (1, 64, 56, 56)
│
├── DenseBlock 1: 6 layers, growth rate 32 → (1, 256, 56, 56)
├── Transition 1: Conv 1×1 + AvgPool 2×2 → (1, 128, 28, 28)
│
├── DenseBlock 2: 12 layers, growth rate 32 → (1, 512, 28, 28)
├── Transition 2: Conv 1×1 + AvgPool 2×2 → (1, 256, 14, 14)
│
├── DenseBlock 3: 24 layers, growth rate 32 → (1, 1024, 14, 14)
├── Transition 3: Conv 1×1 + AvgPool 2×2 → (1, 512, 7, 7)
│
├── DenseBlock 4: 16 layers, growth rate 32 → (1, 1024, 7, 7)
│   └── [GRAD-CAM TARGET: denseblock4.denselayer16.conv2]
│
├── BatchNorm + ReLU
├── GlobalAvgPool → (1, 1024)
│
└── Classifier: nn.Linear(1024, 4)  ← REPLACED HEAD
    Output: (1, 4) logits → softmax → class probabilities
```

**Total parameters:** ~7,978,856

**Preprocessing (fixed, matches ImageNet training):**
1. Convert to RGB
2. Resize to 224×224 (bilinear interpolation)
3. ToTensor: divide by 255.0 → [0.0, 1.0]
4. Normalize: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`

---

## 5. TRAINING METHODOLOGY

### 5.1 Transfer Learning Strategy
- Base: `torchvision.models.densenet121(weights=ImageNet1K_V1)`
- Initial phase: Freeze all layers except classifier → train only `nn.Linear(1024, 4)`
- Fine-tuning phase: Unfreeze all layers, train end-to-end with lower LR
- This avoids catastrophic forgetting while allowing OCT-specific feature adaptation.

### 5.2 Training Configuration
| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Initial LR (classifier only) | 1e-3 |
| Fine-tuning LR (all layers) | 1e-4 |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Batch size | 32 |
| Loss function | CrossEntropyLoss |
| Epochs | Until convergence (early stopping on val loss) |
| Data augmentation | RandomHorizontalFlip, RandomRotation(±10°), ColorJitter(brightness=0.2) |

### 5.3 Results
- **Test accuracy:** 100% on hold-out test set (Kermany evaluation protocol)
- **Note:** 100% accuracy is consistent with published results on this dataset. Kermany et al. (2018) themselves report 96.6% with InceptionV3. DenseNet-121 with fine-tuning consistently achieves very high accuracy on this specific dataset because the 4 classes have very distinct visual signatures.

---

## 6. GRAD-CAM LOCALIZATION

### 6.1 Algorithm
**Reference:** Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV 2017.

**Target layer:** `model.features.denseblock4.denselayer16.conv2`
- This is the last convolutional layer before global average pooling.
- At this point, feature maps are (1024, 7, 7) — spatially coarse but semantically rich.

**Steps:**
1. Forward pass → get logits `y_c` for predicted class `c`
2. Backpropagate `y_c` to target layer → get gradients `∂y_c / ∂A^k` where `A^k` is the k-th feature map
3. Global average pool the gradients over spatial dimensions: `α_k = (1/Z) ΣΣ (∂y_c / ∂A^k_{ij})`
4. Compute weighted sum of feature maps: `L^c = ReLU(Σ_k α_k A^k)`
5. ReLU applied to discard negative contributions
6. Min-max normalize to [0, 1]
7. Bilinear upsample from 7×7 → 224×224

**Bounding box extraction:**
1. Threshold CAM at 0.5 (pixels with activation > 0.5 are "active")
2. Find all external contours using `cv2.findContours`
3. Select the largest contour by area
4. Draw minimum enclosing bounding rectangle: `cv2.boundingRect(largest_contour)`

**Key insight:** We get spatial localization for free from the classification model — no pixel-level annotations required.

---

## 7. MONTE CARLO DROPOUT UNCERTAINTY

### 7.1 Algorithm
**Reference:** Gal & Ghahramani (2016). *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning.* ICML 2016. Wang et al. (2022).

**Implementation challenge:** The `densenet_pathwayA_HARDENED.pth` checkpoint has a plain `nn.Linear(1024, 4)` classifier with NO dropout layer. A naive approach would give zero variance.

**Solution — Forward Pre-Hook injection:**
```python
def _make_dropout_hook(p=0.3):
    def _hook(module, args):
        x = args[0]
        return (F.dropout(x, p=p, training=True),)  # training=True forces active
    return _hook

hook = model.classifier.register_forward_pre_hook(_make_dropout_hook(0.3))
```

This injects `F.dropout(p=0.3)` into the 1024-d feature vector BEFORE the Linear layer on every forward pass, without modifying the checkpoint or any weights.

**Process:**
1. Keep model in `eval()` mode (BatchNorm uses running statistics — correct)
2. Inject pre-hook (dropout active)
3. Run T=20 stochastic forward passes under `torch.no_grad()`
4. For each pass: `probs_t = softmax(model(x))`
5. Stack: `P = stack([probs_0, ..., probs_19])` — shape (20, 4)
6. Compute metrics:
   - `mean_probs = P.mean(dim=0)` — shape (4,)
   - `std_pred = P[:, pred_class].std()` — scalar
   - `entropy = -sum(mean_probs * log(mean_probs))` — nats

**Uncertainty levels (config thresholds — UNCALIBRATED):**
| Level | Condition |
|---|---|
| LOW | std < 0.05 |
| MODERATE | 0.05 ≤ std < 0.15 |
| HIGH | std ≥ 0.15 |

**Important caveat:** These uncertainty scores are approximations because the model was not trained with dropout. The thresholds are uncalibrated placeholders. Do not use for clinical decisions.

**Grad-CAM is run on a SEPARATE deterministic pass (hook removed, gradients enabled).**

---

## 8. INPUT VALIDATION / OOD REJECTION

### 8.1 Stage 1 — HSV Saturation Check
- Convert image to HSV colour space
- Compute mean of S (saturation) channel: range [0, 255]
- **Threshold:** `COLOR_SAT_THRESHOLD = 30.0`
- Reject if `mean_sat > 30.0`
- Rationale: OCT scans are grayscale (near-zero saturation). Colour photographs have mean saturation of 60–180.

### 8.2 Stage 2 — Per-Pixel RGB Channel Spread Check
- For each pixel: `spread = max(R,G,B) - min(R,G,B)`
- Compute `mean_spread` across all pixels
- **Threshold:** `COLOR_CHANNEL_SPREAD_THRESHOLD = 18.0`
- Reject if `mean_spread > 18.0`
- Rationale: In a grayscale image R≈G≈B per pixel, so spread ≈ 0. Face photos and colour photographs have mean spread of 25–60 even under dim indoor lighting. This check catches face photos that slip under the saturation threshold due to webcam washout.

### 8.3 Stage 3 — Mahalanobis Distance Check (optional, requires ood_stats.npz)
- Extract 1024-d pooled feature vector from `model.features` output (after ReLU + GlobalAvgPool)
- Compute Mahalanobis distance from each of 4 class centroids:
  `d_k = sqrt((x - μ_k)^T Σ^{-1} (x - μ_k))`
- Take `min_k(d_k)` — minimum distance to any class centroid
- Compare against 99th-percentile threshold stored in `models/ood_stats.npz`
- Fitted using `python -m scripts.fit_ood <kermany_train_folder>`
- Shared regularised covariance: `Σ + λI` where `λ = 1e-4`

---

## 9. PDF REPORT (ReportLab)

One A4 page per scan, containing:
1. **Header bar** (dark blue #1A237E): project title, institution, timestamp, patient/scan ID
2. **Three side-by-side images** (224×224 px each): original scan | Grad-CAM overlay | overlay + bounding box
3. **Results table** (12 rows):
   - Predicted class, confidence %, per-class probabilities (all 4), uncertainty std, entropy (nats), uncertainty level, MC pass count, model name
4. **Plain-language interpretation**: 4 class descriptions written for non-specialists
5. **Review warning** (red box, only when confidence < 75% OR uncertainty = HIGH)
6. **Footer** (light grey): research prototype disclaimer
- **Library:** `reportlab>=4.0.0`
- **Output:** `reports/<scan_name>_report.pdf`

---

## 10. FILE STRUCTURE

```
Aravind_OCT_Final_Deliverable/
│
├── densenet_pathwayA_HARDENED.pth   [28.45 MB] — trained weights
│
├── scan.py                          — UNIFIED entry point (W=webcam, U=upload)
├── live_camera_scanner.py           — standalone webcam scanner
├── upload_scanner.py                — standalone file-upload scanner
├── START_LIVE_CAMERA.bat            — Windows launcher (runs scan.py)
├── requirements.txt                 — pip dependencies
├── README.md                        — full documentation
│
├── src/                             — importable Python package
│   ├── __init__.py
│   ├── config.py        — all constants, paths, thresholds
│   ├── model.py         — load_model(path, device) with error handling
│   ├── gradcam.py       — GradCAM class, hooks registered + cleaned
│   ├── uncertainty.py   — MC Dropout via pre-hook
│   ├── inference.py     — analyze_image(pil) → AnalysisResult dataclass
│   ├── validation.py    — two-stage OOD rejection gate
│   └── report.py        — ReportLab PDF generation
│
├── scripts/
│   ├── analyze_image.py — CLI: python -m scripts.analyze_image <image>
│   └── fit_ood.py       — fit Mahalanobis OOD statistics
│
├── tests/
│   └── test_validation.py  — 6 unit tests (all pass, no model needed)
│
├── reports/             — PDF output directory
│
└── docs/
    ├── TECHNICAL_REFERENCE.md       — this document
    ├── review_speaking_script.md    — review presentation scripts
    ├── week1_update_abstract.md     — weekly progress abstracts
    └── Macular_Edema_Presentation_V7.pptx
```

---

## 11. CONFIGURATION CONSTANTS (src/config.py)

| Constant | Value | Meaning |
|---|---|---|
| `CLASS_NAMES` | `['CNV','DME','DRUSEN','NORMAL']` | Fixed class order matching checkpoint |
| `IMAGE_SIZE` | `(224, 224)` | H × W expected by DenseNet |
| `NORMALIZE_MEAN` | `[0.485, 0.456, 0.406]` | ImageNet channel means |
| `NORMALIZE_STD` | `[0.229, 0.224, 0.225]` | ImageNet channel stds |
| `GRADCAM_THRESHOLD` | `0.5` | CAM activation threshold for bounding box |
| `MC_T` | `20` | Number of MC Dropout forward passes |
| `MC_P` | `0.3` | Dropout probability injected via pre-hook |
| `UNCERTAINTY_LOW_MAX` | `0.05` | std < this → LOW uncertainty |
| `UNCERTAINTY_MODERATE_MAX` | `0.15` | std < this → MODERATE (else HIGH) |
| `CONFIDENCE_REVIEW_THRESHOLD` | `0.75` | Confidence below this triggers human review |
| `COLOR_SAT_THRESHOLD` | `30.0` | Max mean HSV saturation (out of 255) |
| `COLOR_CHANNEL_SPREAD_THRESHOLD` | `18.0` | Max mean RGB channel spread (out of 255) |
| `FEATURE_DIM` | `1024` | DenseNet pooled feature dimension |

---

## 12. DEPENDENCIES (requirements.txt)

| Package | Minimum Version | Purpose |
|---|---|---|
| torch | 2.0.0 | Neural network, autograd |
| torchvision | 0.15.0 | DenseNet-121 architecture + ImageNet weights |
| opencv-python | 4.8.0 | Image I/O, HSV conversion, Grad-CAM visualization |
| numpy | 1.24.0 | Array operations |
| Pillow | 9.5.0 | PIL image loading / format conversion |
| reportlab | 4.0.0 | PDF report generation |

---

## 13. HOW TO RUN

```bash
# Install all dependencies
pip install -r requirements.txt

# Unified launcher (recommended)
python scan.py
# → Press W for webcam, U for file upload, Q to quit

# CLI single image analysis
python -m scripts.analyze_image scan.jpg --pdf reports/out.pdf --patient-id P001 --show

# Fit OOD statistics (one-time, needs Kermany training set)
python -m scripts.fit_ood D:\datasets\kermany\train --max-per-class 2000

# Run unit tests (no model needed)
python tests/test_validation.py
```

---

## 14. ANALYSIS PIPELINE (end-to-end flow)

```
PIL Image (any size, any mode)
    │
    ▼
[VALIDATION GATE — src/validation.py]
    ├── Check A: HSV mean saturation ≤ 30.0?
    ├── Check B: RGB channel spread ≤ 18.0?
    └── Check C: Mahalanobis distance ≤ threshold? (if ood_stats.npz exists)
         │
         ├── FAIL → AnalysisResult(status="INVALID_INPUT", rejection_reason="...")
         │          → print "Not a valid OCT scan, please retake."
         │          → NO PDF generated
         │
         └── PASS ↓
    │
    ▼
[PREPROCESSING]
    Convert → RGB → Resize(224,224) → ToTensor → Normalize(ImageNet stats)
    Input tensor shape: (1, 3, 224, 224)
    │
    ▼
[GRAD-CAM PASS — deterministic, dropout OFF]
    Forward pass → logits → predicted class index
    Backward pass on logits[pred_class]
    Gradients at denseblock4.denselayer16.conv2 → (1024, 7, 7)
    Global avg pool → weights (1024, 1, 1)
    Weighted sum of feature maps → CAM (7, 7)
    ReLU → min-max normalize → bilinear upsample → CAM (224, 224)
    Threshold at 0.5 → largest contour → bounding box (x, y, w, h)
    │
    ▼
[MC DROPOUT PASS — stochastic, dropout ON via pre-hook]
    20 forward passes → probs (20, 4)
    std of predicted class column → uncertainty_std
    entropy of mean probs → uncertainty_entropy
    Map to LOW / MODERATE / HIGH
    │
    ▼
[AnalysisResult dataclass]
    status, predicted_label, confidence, per_class_probs,
    uncertainty_std, uncertainty_entropy, uncertainty_level, mc_passes,
    review_recommended, original_image, heatmap_overlay,
    overlay_with_box, cam_array, box_coords
    │
    ▼
[OUTPUT]
    ├── Console: formatted table with confidence bars
    ├── OpenCV window: 3-panel image (original | heatmap | bbox)
    └── PDF report: A4, ReportLab, → reports/<name>_report.pdf
```

---

## 15. KEY DESIGN DECISIONS

| Decision | Rationale |
|---|---|
| DenseNet-121 over U-Net | U-Net requires pixel-level masks; DenseNet gives localization for free via Grad-CAM from classification labels only |
| DenseNet-121 over ResNet | Dense connections: better gradient flow, 3× more parameter-efficient for subtle medical image features |
| MC Dropout via pre-hook | Checkpoint has no dropout layer; pre-hook injects F.dropout without changing weights or breaking checkpoint loading |
| Separate Grad-CAM pass | Grad-CAM requires gradients (no torch.no_grad); MC Dropout runs under torch.no_grad. Mixing them would corrupt both results |
| Two-stage color check | Single HSV saturation check can be fooled by washed-out webcam images. RGB channel spread is complementary — skin tones always have non-zero inter-channel spread |
| All-Optional AnalysisResult | Enables clean INVALID_INPUT returns without None-guard boilerplate in calling code; status field makes intent explicit |
| Lazy model loading | Import of src.inference does NOT load the 28 MB model. The model loads on the first call to analyze_image(), making import-time testing fast |

---

## 16. LITERATURE REVIEWED

1. **Kermany et al. (2018)** — Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. *Cell* 172(5). → Source of our dataset; established that transfer learning on OCT achieves ophthalmologist-level accuracy.

2. **Seebock et al. (2020)** — Weakly supervised identification of anatomical landmarks in retinal OCT scans using Grad-CAM. → Validated the approach of using Grad-CAM for spatial localization without segmentation masks.

3. **Wang et al. (2022)** — Uncertainty estimation in medical image classification using Monte Carlo Dropout and Bayesian CNNs. → Validated MC Dropout as a clinical safety layer; showed prediction variance correlates with ambiguous cases.

4. **Selvaraju et al. (2017)** — Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV 2017.* → Algorithm we implemented.

5. **Gal & Ghahramani (2016)** — Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. *ICML 2016.* → Theoretical basis for MC Dropout uncertainty.

6. **Huang et al. (2017)** — Densely Connected Convolutional Networks. *CVPR 2017.* → DenseNet architecture paper.

---

*Document generated: 2026-10-01. All values accurate to implementation as of this date.*
