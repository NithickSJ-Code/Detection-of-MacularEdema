"""
live_camera_scanner.py  (refactored)
=====================================
Webcam-based OCT scan demo using the src package.

Controls
--------
SPACE   — Capture the scan in the green targeting box and run analysis.
          A PDF report is saved to reports/latest_scan.pdf and the result
          is displayed in a new OpenCV window.
Q       — Quit.
"""
from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# All AI logic lives in the src package
from src.inference import analyze_image
from src.report import save_pdf


# ---------------------------------------------------------------------------
# Cross-platform file opener (replaces Windows-only os.startfile)
# ---------------------------------------------------------------------------

def _open_file(path: Path) -> None:
    """Open *path* with the OS default application."""
    if platform.system() == "Windows":
        import os
        os.startfile(str(path))
    elif platform.system() == "Darwin":                 # macOS
        subprocess.run(["open", str(path)], check=False)
    else:                                               # Linux / WSL
        subprocess.run(["xdg-open", str(path)], check=False)


# ---------------------------------------------------------------------------
# Result display helper
# ---------------------------------------------------------------------------

def _show_result(result) -> None:
    """Build composite RGB image and show it in a new OpenCV window."""
    combined = np.hstack([
        result.original_image,
        result.heatmap_overlay,
        result.overlay_with_box,
    ])
    combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

    label = (
        f"DIAGNOSIS: {result.predicted_label}  |  "
        f"CONF: {result.confidence*100:.1f}%  |  "
        f"UNCERTAINTY: {result.uncertainty_level}"
    )
    color = (0, 200, 0) if result.predicted_label == "NORMAL" else (0, 80, 255)
    cv2.putText(combined_bgr, label, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)

    if result.review_recommended:
        cv2.putText(combined_bgr, "⚠ HUMAN REVIEW RECOMMENDED", (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    cv2.imshow("OCT Analysis Result — press any key to close", combined_bgr)


# ---------------------------------------------------------------------------
# Main camera loop
# ---------------------------------------------------------------------------

def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check device connection.")
        sys.exit(1)

    print("[INFO] Live camera started.")
    print("[INFO] Align the OCT scan inside the green box, then press SPACE.")
    print("[INFO] Press Q to quit.\n")

    REPORTS_DIR = Path("reports")
    REPORTS_DIR.mkdir(exist_ok=True)

    BOX_SIZE = 300

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame.")
            break

        h, w, _ = frame.shape
        display = frame.copy()
        cx, cy  = w // 2, h // 2
        tl = (cx - BOX_SIZE // 2, cy - BOX_SIZE // 2)
        br = (cx + BOX_SIZE // 2, cy + BOX_SIZE // 2)

        # Guide box
        cv2.rectangle(display, tl, br, (0, 255, 255), 1)

        # Corner L-brackets
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
                    (cx - 120, tl[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2)
        cv2.putText(display, "SPACE: Analyze   |   Q: Quit",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2)

        cv2.imshow("Aravind Eye Hospital — Live Diagnostic Camera", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        elif key == 32:   # SPACE
            print("[INFO] Capturing and analysing …")
            crop    = frame[tl[1]:br[1], tl[0]:br[0]]
            pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

            result = analyze_image(pil_img)

            # ---- Handle invalid / non-OCT input ----------------------
            if result.status == "INVALID_INPUT":
                print(f"[REJECTED] Not a valid OCT scan. {result.rejection_reason}")
                # Flash rejection message on camera feed for 2 seconds
                reject_frame = display.copy()
                cv2.putText(reject_frame, "NOT A VALID OCT SCAN",
                            (cx - 180, cy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                cv2.putText(reject_frame, "Please retake the scan.",
                            (cx - 150, cy + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                cv2.imshow(
                    "Aravind Eye Hospital \u2014 Live Diagnostic Camera",
                    reject_frame,
                )
                cv2.waitKey(2000)   # show for 2 s then resume
                continue

            # ---- Normal result ---------------------------------------
            print(f"[RESULT] {result.predicted_label}  "
                  f"conf={result.confidence*100:.1f}%  "
                  f"uncertainty={result.uncertainty_level}")

            # Save PDF report
            pdf_path = REPORTS_DIR / "latest_scan.pdf"
            save_pdf(result, pdf_path)
            print(f"[INFO] PDF saved → {pdf_path}")

            # Show result panel (non-blocking — camera keeps running)
            _show_result(result)

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera released. Goodbye.")


if __name__ == "__main__":
    main()
