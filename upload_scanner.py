"""
upload_scanner.py — File-upload OCT scanner.

Opens a Windows file-picker dialog, lets you select any OCT scan image,
runs the full analysis pipeline (validation → Grad-CAM → MC Dropout →
uncertainty → PDF report), and displays the result in an OpenCV window.

Usage:
    python upload_scanner.py
"""
from __future__ import annotations

import platform
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import cv2
import numpy as np
from PIL import Image

# Full src pipeline
from src.inference import analyze_image
from src.report import save_pdf


# ---------------------------------------------------------------------------
# Cross-platform file opener
# ---------------------------------------------------------------------------

def _open_file(path: Path) -> None:
    if platform.system() == "Windows":
        import os
        os.startfile(str(path))
    elif platform.system() == "Darwin":
        subprocess.run(["open", str(path)], check=False)
    else:
        subprocess.run(["xdg-open", str(path)], check=False)


# ---------------------------------------------------------------------------
# File picker
# ---------------------------------------------------------------------------

def _pick_file() -> Path | None:
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", True)   # bring dialog to front
    file_str = filedialog.askopenfilename(
        title="Select OCT Scan Image",
        filetypes=[
            ("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.tif"),
            ("All Files",   "*.*"),
        ],
    )
    root.destroy()
    return Path(file_str) if file_str else None


# ---------------------------------------------------------------------------
# Display result in OpenCV window
# ---------------------------------------------------------------------------

def _show_result(result) -> None:
    combined = np.hstack([
        result.original_image,
        result.heatmap_overlay,
        result.overlay_with_box,
    ])
    combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

    # Diagnosis text
    label = (
        f"DIAGNOSIS: {result.predicted_label}  |  "
        f"CONF: {result.confidence * 100:.1f}%  |  "
        f"UNCERTAINTY: {result.uncertainty_level}"
    )
    color = (0, 200, 0) if result.predicted_label == "NORMAL" else (0, 100, 255)
    cv2.putText(combined_bgr, label, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)

    # Review warning
    if result.review_recommended:
        cv2.putText(combined_bgr, "WARNING: HUMAN REVIEW RECOMMENDED",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Column captions
    h, w, _ = combined_bgr.shape
    for i, caption in enumerate(["Original", "Grad-CAM Overlay", "+ Bounding Box"]):
        x = int((i + 0.15) * w / 3)
        cv2.putText(combined_bgr, caption, (x, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("OCT Analysis Result — press any key to close", combined_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  Aravind Eye Hospital — OCT Upload Scanner")
    print("=" * 60)

    # ---- Step 1: Pick file -------------------------------------------
    print("\n[INFO] Opening file dialog …")
    file_path = _pick_file()

    if file_path is None:
        print("[INFO] No file selected. Exiting.")
        sys.exit(0)

    if not file_path.is_file():
        print(f"[ERROR] File not found: {file_path}")
        sys.exit(1)

    print(f"[INFO] Selected: {file_path}")

    # ---- Step 2: Load image ------------------------------------------
    try:
        pil_img = Image.open(file_path)
    except Exception as e:
        print(f"[ERROR] Could not open image: {e}")
        sys.exit(1)

    # ---- Step 3: Run full pipeline (validation + diagnosis) ----------
    print("[INFO] Running analysis pipeline …")
    result = analyze_image(pil_img)

    # ---- Step 4: Handle INVALID_INPUT --------------------------------
    if result.status == "INVALID_INPUT":
        print("\n" + "!" * 60)
        print("  Not a valid OCT scan, please retake.")
        print(f"  Reason: {result.rejection_reason}")
        print("!" * 60 + "\n")
        sys.exit(2)

    # ---- Step 5: Print summary ---------------------------------------
    print("\n" + "=" * 60)
    print(f"  PREDICTED CLASS  : {result.predicted_label}")
    print(f"  CONFIDENCE       : {result.confidence * 100:.1f}%")
    print(f"  UNCERTAINTY STD  : {result.uncertainty_std:.4f}")
    print(f"  UNCERTAINTY LVL  : {result.uncertainty_level}")
    print(f"  REVIEW NEEDED    : {'YES ⚠' if result.review_recommended else 'No'}")
    print("-" * 60)
    from src.config import CLASS_NAMES
    for cls in CLASS_NAMES:
        p = result.per_class_probs.get(cls, 0.0)
        bar = "█" * int(p * 30)
        print(f"  {cls:8s}  {p * 100:5.1f}%  {bar}")
    print("=" * 60 + "\n")

    # ---- Step 6: Save PDF report ------------------------------------
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    pdf_path = reports_dir / f"{file_path.stem}_report.pdf"
    save_pdf(result, pdf_path, patient_id=file_path.stem)
    print(f"[INFO] PDF report saved → {pdf_path}")

    # ---- Step 7: Show result window and open PDF --------------------
    print("[INFO] Displaying result … (press any key in the window to close)")
    _show_result(result)

    # Open the PDF automatically
    _open_file(pdf_path)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
