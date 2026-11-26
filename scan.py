"""
scan.py — Unified OCT Scanner entry point.

Run this single script to access both scan modes:

    python scan.py

You will be prompted to choose:
    [W]  Webcam scan   — hold the OCT scan up to your camera
    [U]  Upload scan   — pick an image file from your computer

The full pipeline (validation → Grad-CAM → MC Dropout → PDF report) runs
identically in both modes.
"""
from __future__ import annotations

import sys


# ---------------------------------------------------------------------------
# Shared result display & PDF helpers (used by both modes)
# ---------------------------------------------------------------------------

def _print_summary(result) -> None:
    from src.config import CLASS_NAMES
    print("\n" + "=" * 60)
    print(f"  PREDICTED CLASS  : {result.predicted_label}")
    print(f"  CONFIDENCE       : {result.confidence * 100:.1f}%")
    print(f"  UNCERTAINTY STD  : {result.uncertainty_std:.4f}")
    print(f"  UNCERTAINTY LVL  : {result.uncertainty_level}")
    print(f"  REVIEW NEEDED    : {'YES ⚠' if result.review_recommended else 'No'}")
    print("-" * 60)
    for cls in CLASS_NAMES:
        p = result.per_class_probs.get(cls, 0.0)
        bar = "█" * int(p * 30)
        print(f"  {cls:8s}  {p * 100:5.1f}%  {bar}")
    print("=" * 60 + "\n")


def _show_cv_result(result) -> None:
    import cv2
    import numpy as np
    combined = np.hstack([
        result.original_image,
        result.heatmap_overlay,
        result.overlay_with_box,
    ])
    combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
    label = (
        f"DIAGNOSIS: {result.predicted_label}  |  "
        f"CONF: {result.confidence * 100:.1f}%  |  "
        f"UNCERTAINTY: {result.uncertainty_level}"
    )
    color = (0, 200, 0) if result.predicted_label == "NORMAL" else (0, 100, 255)
    cv2.putText(combined_bgr, label, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)
    if result.review_recommended:
        cv2.putText(combined_bgr, "WARNING: HUMAN REVIEW RECOMMENDED",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    h, w, _ = combined_bgr.shape
    for i, cap in enumerate(["Original", "Grad-CAM Overlay", "+ Bounding Box"]):
        cv2.putText(combined_bgr, cap,
                    (int((i + 0.1) * w / 3), h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.imshow("OCT Analysis Result — press any key to close", combined_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _save_and_open_pdf(result, stem: str = "scan") -> None:
    import platform, subprocess, os
    from pathlib import Path
    from src.report import save_pdf

    Path("reports").mkdir(exist_ok=True)
    pdf_path = Path("reports") / f"{stem}_report.pdf"
    save_pdf(result, pdf_path, patient_id=stem)
    print(f"[INFO] PDF saved → {pdf_path}")

    # Cross-platform open
    if platform.system() == "Windows":
        os.startfile(str(pdf_path))
    elif platform.system() == "Darwin":
        subprocess.run(["open", str(pdf_path)], check=False)
    else:
        subprocess.run(["xdg-open", str(pdf_path)], check=False)


def _handle_invalid(result) -> None:
    print("\n" + "!" * 60)
    print("  Not a valid OCT scan, please retake.")
    print(f"  Reason: {result.rejection_reason}")
    print("!" * 60 + "\n")


# ---------------------------------------------------------------------------
# Mode W — Webcam
# ---------------------------------------------------------------------------

def _run_webcam() -> None:
    import cv2
    from PIL import Image
    from src.inference import analyze_image
    from src.report import save_pdf

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check device connection.")
        return

    print("\n[WEBCAM] Camera started.")
    print("[WEBCAM] Align the OCT scan inside the green box.")
    print("[WEBCAM] SPACE = Analyze   |   Q = Quit\n")

    BOX_SIZE = 300

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame.")
            break

        h, w, _ = frame.shape
        display  = frame.copy()
        cx, cy   = w // 2, h // 2
        tl = (cx - BOX_SIZE // 2, cy - BOX_SIZE // 2)
        br = (cx + BOX_SIZE // 2, cy + BOX_SIZE // 2)

        # Guide box
        cv2.rectangle(display, tl, br, (0, 255, 255), 1)
        GREEN, THICK, L = (0, 255, 0), 2, 30
        for (px, py), (dx, dy) in [
            (tl, (+L, 0)), (tl, (0, +L)),
            ((br[0], tl[1]), (-L, 0)), ((br[0], tl[1]), (0, +L)),
            ((tl[0], br[1]), (+L, 0)), ((tl[0], br[1]), (0, -L)),
            (br, (-L, 0)),             (br, (0, -L)),
        ]:
            cv2.line(display, (px, py), (px + dx, py + dy), GREEN, THICK)

        cv2.putText(display, "SYSTEM: STANDBY",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
        cv2.putText(display, "ALIGN OCT SCAN IN BOX",
                    (cx - 120, tl[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)
        cv2.putText(display, "SPACE: Analyze   |   Q: Quit",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2)

        cv2.imshow("Aravind Eye Hospital — Webcam Scanner", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == 32:   # SPACE
            print("[INFO] Capturing and analysing …")
            crop    = frame[tl[1]:br[1], tl[0]:br[0]]
            pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            result  = analyze_image(pil_img)

            if result.status == "INVALID_INPUT":
                print(f"[REJECTED] {result.rejection_reason}")
                reject_frame = display.copy()
                cv2.putText(reject_frame, "NOT A VALID OCT SCAN",
                            (cx - 180, cy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                cv2.putText(reject_frame, "Please retake the scan.",
                            (cx - 150, cy + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                cv2.imshow("Aravind Eye Hospital — Webcam Scanner", reject_frame)
                cv2.waitKey(2000)
                continue

            _print_summary(result)
            _save_and_open_pdf(result, stem="webcam_scan")
            _show_cv_result(result)

    cap.release()
    cv2.destroyAllWindows()
    print("[WEBCAM] Camera released.")


# ---------------------------------------------------------------------------
# Mode U — Upload
# ---------------------------------------------------------------------------

def _run_upload() -> None:
    import tkinter as tk
    from tkinter import filedialog
    from pathlib import Path
    from PIL import Image
    from src.inference import analyze_image

    # File picker
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", True)
    file_str = filedialog.askopenfilename(
        title="Select OCT Scan Image",
        filetypes=[
            ("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.tif"),
            ("All Files",   "*.*"),
        ],
    )
    root.destroy()

    if not file_str:
        print("[INFO] No file selected.")
        return

    file_path = Path(file_str)
    print(f"\n[UPLOAD] Selected: {file_path.name}")

    try:
        pil_img = Image.open(file_path)
    except Exception as e:
        print(f"[ERROR] Could not open image: {e}")
        return

    print("[INFO] Running analysis pipeline …")
    result = analyze_image(pil_img)

    if result.status == "INVALID_INPUT":
        _handle_invalid(result)
        return

    _print_summary(result)
    _save_and_open_pdf(result, stem=file_path.stem)
    print("[INFO] Displaying result window … (press any key to close)")
    _show_cv_result(result)


# ---------------------------------------------------------------------------
# Main — mode selector
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║    Aravind Eye Hospital — OCT Analysis System        ║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║                                                      ║")
    print("║   [W]  Webcam Scan  — hold scan up to camera         ║")
    print("║   [U]  Upload Scan  — select image from computer     ║")
    print("║   [Q]  Quit                                          ║")
    print("║                                                      ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    while True:
        try:
            choice = input("  Enter your choice (W / U / Q): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n[INFO] Exiting.")
            sys.exit(0)

        if choice == "w":
            _run_webcam()
            break
        elif choice == "u":
            _run_upload()
            break
        elif choice in ("q", "quit", "exit"):
            print("[INFO] Goodbye.")
            sys.exit(0)
        else:
            print("  Invalid choice. Please press W, U, or Q.")


if __name__ == "__main__":
    main()
