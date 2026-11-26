"""
scripts/analyze_image.py — Command-line interface for single-image analysis.

Usage
-----
    python -m scripts.analyze_image path/to/scan.png
    python -m scripts.analyze_image scan.jpg --pdf reports/result.pdf
    python -m scripts.analyze_image scan.png --pdf reports/out.pdf --patient-id P001
    python -m scripts.analyze_image scan.png --show          # display result with cv2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.analyze_image",
        description=(
            "Run the OCT Macular Edema classifier + Grad-CAM on a single image "
            "and optionally save a PDF report."
        ),
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Path to the input OCT scan (PNG / JPG / BMP).",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=None,
        metavar="OUTPUT_PDF",
        help="Save a PDF report to this path (e.g. reports/result.pdf).",
    )
    parser.add_argument(
        "--patient-id",
        type=str,
        default=None,
        metavar="ID",
        help="Optional patient / scan ID to embed in the PDF header.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the three visualisation panels in an OpenCV window.",
    )
    parser.add_argument(
        "--save-jpg",
        type=Path,
        default=None,
        metavar="OUTPUT_JPG",
        help="Save the composite visualisation to a JPEG file.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # ---- Validate input -----------------------------------------------
    if not args.image.is_file():
        print(f"[ERROR] Image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    # ---- Load and run -------------------------------------------------
    from PIL import Image as PILImage
    from src.inference import analyze_image
    from src.config import CLASS_NAMES

    print(f"[INFO] Analyzing: {args.image}")
    pil_img = PILImage.open(args.image)
    result  = analyze_image(pil_img)

    # ---- Handle rejected / invalid input ------------------------------
    if result.status == "INVALID_INPUT":
        print("\n" + "!" * 60)
        print("  Not a valid OCT scan, please retake.")
        print(f"  Reason: {result.rejection_reason}")
        print("!" * 60 + "\n")
        sys.exit(2)

    # ---- Print summary to console -------------------------------------
    print("\n" + "=" * 60)
    print(f"  PREDICTED CLASS  : {result.predicted_label}")
    print(f"  CONFIDENCE       : {result.confidence * 100:.1f}%")
    print(f"  UNCERTAINTY STD  : {result.uncertainty_std:.4f}")
    print(f"  UNCERTAINTY LVL  : {result.uncertainty_level}")
    print(f"  ENTROPY          : {result.uncertainty_entropy:.4f} nats")
    print(f"  REVIEW NEEDED    : {'YES ⚠' if result.review_recommended else 'No'}")
    print("-" * 60)
    for cls in CLASS_NAMES:
        p = result.per_class_probs.get(cls, 0.0)
        bar = "█" * int(p * 30)
        print(f"  {cls:8s}  {p*100:5.1f}%  {bar}")
    print("=" * 60 + "\n")

    # ---- Optional: display with OpenCV --------------------------------
    if args.show or args.save_jpg:
        import cv2
        import numpy as np

        combined = np.hstack([
            result.original_image,
            result.heatmap_overlay,
            result.overlay_with_box,
        ])
        # Convert RGB → BGR for OpenCV
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        label_text = (
            f"DIAGNOSIS: {result.predicted_label}  |  "
            f"CONF: {result.confidence*100:.1f}%  |  "
            f"UNCERTAINTY: {result.uncertainty_level}"
        )
        cv2.putText(
            combined_bgr, label_text, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
        )

        if args.save_jpg:
            args.save_jpg.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.save_jpg), combined_bgr)
            print(f"[INFO] Visualisation saved → {args.save_jpg}")

        if args.show:
            cv2.imshow("OCT Analysis — press any key to close", combined_bgr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # ---- Optional: generate PDF ---------------------------------------
    if args.pdf:
        from src.report import save_pdf
        save_pdf(result, args.pdf, patient_id=args.patient_id)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
