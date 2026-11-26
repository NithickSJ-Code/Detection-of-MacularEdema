"""
scripts/fit_ood.py — Fit per-class Gaussian statistics for OOD detection.

Extracts 1024-d pooled DenseNet features from a folder of labelled real OCT
images, computes per-class mean vectors and a shared regularised covariance
matrix, determines the 99th-percentile Mahalanobis distance as the rejection
threshold, and saves everything to models/ood_stats.npz.

Expected folder layout (Kermany-style):
    <root>/
        CNV/       *.jpeg  *.jpg  *.png
        DME/
        DRUSEN/
        NORMAL/

Unlabelled images (all in one flat folder) are also supported — they will be
treated as a single "unknown" class for covariance estimation only; per-class
means will not be saved for classes that have no subfolder.

Usage
-----
    python -m scripts.fit_ood path/to/oct_train_folder [--max-per-class 2000]

The script prints progress and writes models/ood_stats.npz when done.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
try:
    from tqdm import tqdm   # optional progress bar
except ImportError:
    tqdm = None             # gracefully skipped if not installed

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _image_paths(folder: Path, exts=(".jpeg", ".jpg", ".png", ".bmp")):
    return [p for p in folder.rglob("*") if p.suffix.lower() in exts]


def _find_class_folders(root: Path, class_names: list[str]) -> dict[str, Path]:
    """Map class_name → subfolder path for any class that exists."""
    result = {}
    for name in class_names:
        candidate = root / name
        if candidate.is_dir():
            result[name] = candidate
    return result


def _get_transform(image_size=(224, 224)):
    return T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _extract_features_batch(
    model,
    paths: list[Path],
    transform,
    device: torch.device,
    max_images: int,
) -> np.ndarray:
    """
    Extract pooled 1024-d features for up to max_images images.
    Returns ndarray of shape (N, 1024).
    """
    paths = paths[:max_images]
    feats = []

    try:
        _iter = tqdm(paths, desc="  Extracting", leave=False, ncols=80) if tqdm else paths
    except Exception:
        _iter = paths

    # We hook model.features to capture the conv output
    captured: list[torch.Tensor] = []

    def _hook(_m, _i, o):
        captured.append(o.detach())

    handle = model.features.register_forward_hook(_hook)

    try:
        for p in _iter:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                continue
            inp = transform(img).unsqueeze(0).to(device)
            captured.clear()
            with torch.no_grad():
                model(inp)
            if not captured:
                continue
            feat = F.relu(captured[0], inplace=False)
            feat = F.adaptive_avg_pool2d(feat, (1, 1))
            feat = torch.flatten(feat, 1).squeeze(0).cpu().numpy()
            feats.append(feat)
    finally:
        handle.remove()

    return np.array(feats, dtype=np.float32) if feats else np.empty((0, 1024), np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.fit_ood",
        description="Fit OOD statistics for the Mahalanobis input validation gate.",
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Root folder containing class subfolders (CNV / DME / DRUSEN / NORMAL).",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=2000,
        metavar="N",
        help="Maximum images per class to use (default: 2000).",
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=1e-4,
        help="Ridge regularisation added to covariance diagonal (default: 1e-4).",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        help="Percentile of in-distribution distances used as the OOD threshold (default: 99).",
    )
    args = parser.parse_args()

    if not args.folder.is_dir():
        print(f"[ERROR] Folder not found: {args.folder}", file=sys.stderr)
        sys.exit(1)

    from src.config import CLASS_NAMES, OOD_STATS_PATH, FEATURE_DIM
    from src.model import load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[fit_ood] Loading model on {device} …")
    model = load_model(device=device)
    transform = _get_transform()

    class_folders = _find_class_folders(args.folder, CLASS_NAMES)
    if not class_folders:
        print(
            f"[WARNING] No class subfolders ({CLASS_NAMES}) found in {args.folder}.\n"
            "Treating all images as a single pool (covariance only, no per-class means).",
            file=sys.stderr,
        )

    # ---- Extract features per class -----------------------------------
    all_feats_list: list[np.ndarray] = []
    class_means: dict[str, np.ndarray] = {}

    for cls_name, cls_path in class_folders.items():
        print(f"[fit_ood] Class '{cls_name}' — {cls_path}")
        paths = _image_paths(cls_path)
        print(f"  Found {len(paths)} images; using up to {args.max_per_class}.")
        feats = _extract_features_batch(
            model, paths, transform, device, args.max_per_class
        )
        if feats.shape[0] == 0:
            print(f"  [WARNING] No features extracted for {cls_name}. Skipping.")
            continue
        class_means[cls_name] = feats.mean(axis=0)
        all_feats_list.append(feats)
        print(f"  Extracted {feats.shape[0]} feature vectors.")

    # If no class folders, treat entire folder as one pool
    if not class_folders:
        paths = _image_paths(args.folder)
        print(f"[fit_ood] Flat folder — {len(paths)} images total.")
        feats = _extract_features_batch(
            model, paths, transform, device, args.max_per_class
        )
        all_feats_list.append(feats)

    if not all_feats_list:
        print("[ERROR] No features could be extracted. Aborting.", file=sys.stderr)
        sys.exit(1)

    all_feats = np.vstack(all_feats_list)   # (N_total, 1024)
    print(f"\n[fit_ood] Total feature vectors: {all_feats.shape[0]}")

    # ---- Shared covariance (within-class scatter) ---------------------
    # Subtract per-class means (if available) before computing scatter
    centred_feats = all_feats.copy()
    if class_means:
        # Build per-sample labels to centre correctly
        idx = 0
        for feats in all_feats_list:
            cls_name = CLASS_NAMES[list(class_folders.keys()).index(
                list(class_folders.keys())[all_feats_list.index(feats)]
            )]
            n = feats.shape[0]
            centred_feats[idx:idx + n] -= class_means[cls_name]
            idx += n
    else:
        centred_feats -= all_feats.mean(axis=0)

    cov = (centred_feats.T @ centred_feats) / max(centred_feats.shape[0] - 1, 1)
    cov += np.eye(FEATURE_DIM, dtype=np.float32) * args.ridge   # ridge regularisation
    precision = np.linalg.inv(cov).astype(np.float32)
    print(f"[fit_ood] Covariance computed (ridge={args.ridge}).")

    # ---- Build ordered class_means array (n_classes, 1024) ------------
    means_arr = np.zeros((len(CLASS_NAMES), FEATURE_DIM), dtype=np.float32)
    for i, cls in enumerate(CLASS_NAMES):
        if cls in class_means:
            means_arr[i] = class_means[cls]
        else:
            # fall back to global mean for missing classes
            means_arr[i] = all_feats.mean(axis=0)
            print(f"  [WARNING] No data for class '{cls}'; using global mean.")

    # ---- 99th-percentile threshold (in-distribution) ------------------
    print(f"[fit_ood] Computing {args.percentile}th-percentile threshold …")
    in_dist_dists = []
    for i, cls in enumerate(CLASS_NAMES):
        start = sum(
            all_feats_list[j].shape[0]
            for j in range(list(class_folders.keys()).index(cls))
            if cls in class_folders
        ) if cls in class_folders else 0
        feats_cls = class_means.get(cls)
        if feats_cls is None:
            continue
        # Use the block corresponding to this class
        cls_idx = list(class_folders.keys()).index(cls) if cls in class_folders else -1
        if cls_idx < 0:
            continue
        cls_feats = all_feats_list[cls_idx]
        for x in cls_feats:
            diff = x - means_arr[i]
            d = float(np.sqrt(diff @ precision @ diff))
            in_dist_dists.append(d)

    if in_dist_dists:
        threshold = float(np.percentile(in_dist_dists, args.percentile))
    else:
        # Fallback: use global statistics
        global_mean = all_feats.mean(axis=0)
        dists = [
            float(np.sqrt((x - global_mean) @ precision @ (x - global_mean)))
            for x in all_feats
        ]
        threshold = float(np.percentile(dists, args.percentile))

    print(f"[fit_ood] Threshold (p{args.percentile:.0f}): {threshold:.4f}")

    # ---- Save ---------------------------------------------------------
    OOD_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OOD_STATS_PATH,
        class_means=means_arr,
        precision_matrix=precision,
        threshold=np.float32(threshold),
        class_names=np.array(CLASS_NAMES),
        ridge=np.float32(args.ridge),
        percentile=np.float32(args.percentile),
        n_samples=np.int32(all_feats.shape[0]),
    )
    print(f"\n[fit_ood] ✓ Saved OOD stats → {OOD_STATS_PATH}")
    print(f"  Classes : {CLASS_NAMES}")
    print(f"  Samples : {all_feats.shape[0]}")
    print(f"  Threshold: {threshold:.4f}")


if __name__ == "__main__":
    main()
