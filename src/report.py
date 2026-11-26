"""
report.py — Per-scan PDF report generator using ReportLab.

Public API
----------
    pdf_bytes = generate_pdf(result, patient_id=None)  -> bytes
    save_pdf(result, path, patient_id=None)            -> None

The report is a single A4 page containing:
  • Header  (project title, timestamp, patient/scan ID)
  • Three side-by-side images with captions
  • Results table
  • Plain-language interpretation
  • Disclaimer footer
"""
from __future__ import annotations

import io
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# ReportLab imports
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
)

from src.config import (
    CLASS_NAMES,
    CONFIDENCE_REVIEW_THRESHOLD,
    MODEL_NAME,
    PROJECT_TITLE,
    INSTITUTION,
    MC_P,
    MC_T,
)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_DARK_BLUE  = colors.HexColor("#1A237E")
_MED_BLUE   = colors.HexColor("#3949AB")
_LIGHT_GREY = colors.HexColor("#F5F5F5")
_AMBER      = colors.HexColor("#FF8F00")
_RED        = colors.HexColor("#C62828")
_GREEN      = colors.HexColor("#2E7D32")
_WHITE      = colors.white
_BLACK      = colors.black


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ndarray_to_rl_image(
    arr: np.ndarray,
    width_cm: float,
    height_cm: float,
) -> RLImage:
    """Convert an RGB uint8 numpy array to a ReportLab Image object."""
    pil = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return RLImage(buf, width=width_cm * cm, height=height_cm * cm)


def _uncertainty_color(level: str) -> colors.Color:
    return {"LOW": _GREEN, "MODERATE": _AMBER, "HIGH": _RED}.get(level, _BLACK)


def _class_description(label: str) -> str:
    descriptions = {
        "CNV": (
            "Choroidal Neovascularisation (CNV) — Abnormal blood vessels growing "
            "beneath the retina. Associated with wet Age-related Macular Degeneration."
        ),
        "DME": (
            "Diabetic Macular Edema (DME) — Fluid accumulation in the macula caused "
            "by diabetic damage to retinal blood vessels. A leading cause of vision loss "
            "in working-age adults."
        ),
        "DRUSEN": (
            "Drusen — Small yellow deposits under the retina. A hallmark of early "
            "dry Age-related Macular Degeneration. Monitoring is advised."
        ),
        "NORMAL": (
            "Normal Macula — No pathological fluid or drusen deposits detected in "
            "this scan. Regular follow-up recommended as per clinical protocol."
        ),
    }
    return descriptions.get(label, f"Predicted class: {label}")


# ---------------------------------------------------------------------------
# Page-level decoration
# ---------------------------------------------------------------------------

def _draw_header_footer(canvas, doc):
    """Called by ReportLab for every page."""
    W, H = A4
    canvas.saveState()

    # Top bar
    canvas.setFillColor(_DARK_BLUE)
    canvas.rect(0, H - 2.2 * cm, W, 2.2 * cm, fill=1, stroke=0)

    # Title text inside bar
    canvas.setFillColor(_WHITE)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(1 * cm, H - 1.2 * cm, PROJECT_TITLE)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(1 * cm, H - 1.8 * cm, INSTITUTION)

    # Footer disclaimer
    canvas.setFillColor(_LIGHT_GREY)
    canvas.rect(0, 0, W, 1.1 * cm, fill=1, stroke=0)
    canvas.setFillColor(colors.HexColor("#757575"))
    canvas.setFont("Helvetica-Oblique", 7)
    disclaimer = (
        "RESEARCH PROTOTYPE ONLY — NOT A MEDICAL DEVICE — NOT FOR CLINICAL DIAGNOSIS. "
        "Results must be reviewed and confirmed by a qualified ophthalmologist."
    )
    canvas.drawCentredString(W / 2, 0.4 * cm, disclaimer)

    canvas.restoreState()


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_pdf(
    result,          # src.inference.AnalysisResult
    patient_id: Optional[str] = None,
) -> bytes:
    """
    Generate a one-page A4 PDF report.

    Parameters
    ----------
    result     : AnalysisResult from inference.analyze_image()
    patient_id : Optional string label (scan ID / patient ID) for the header.

    Returns
    -------
    bytes : Raw PDF bytes ready to be written to a file or sent over HTTP.
    """
    buf = io.BytesIO()
    W, H = A4

    # Usable content area (leaving room for header and footer)
    MARGIN_L = 1.5 * cm
    MARGIN_R = 1.5 * cm
    MARGIN_T = 2.8 * cm   # below the top bar
    MARGIN_B = 1.5 * cm   # above the footer

    content_width  = W - MARGIN_L - MARGIN_R
    content_height = H - MARGIN_T - MARGIN_B

    doc = BaseDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=MARGIN_L,
        rightMargin=MARGIN_R,
        topMargin=MARGIN_T,
        bottomMargin=MARGIN_B,
    )

    frame = Frame(
        MARGIN_L, MARGIN_B, content_width, content_height,
        id="main", showBoundary=0,
    )
    template = PageTemplate(
        id="full", frames=[frame], onPage=_draw_header_footer
    )
    doc.addPageTemplates([template])

    styles = getSampleStyleSheet()

    # Custom styles
    h1 = ParagraphStyle(
        "h1", parent=styles["Normal"],
        fontSize=13, textColor=_DARK_BLUE, fontName="Helvetica-Bold",
        spaceAfter=4,
    )
    h2 = ParagraphStyle(
        "h2", parent=styles["Normal"],
        fontSize=10, textColor=_MED_BLUE, fontName="Helvetica-Bold",
        spaceAfter=3,
    )
    body = ParagraphStyle(
        "body", parent=styles["Normal"],
        fontSize=8.5, leading=13,
    )
    caption_style = ParagraphStyle(
        "caption", parent=styles["Normal"],
        fontSize=7.5, textColor=colors.HexColor("#616161"),
        alignment=1,   # centre
    )
    warning_style = ParagraphStyle(
        "warning", parent=styles["Normal"],
        fontSize=9, textColor=_RED, fontName="Helvetica-Bold",
        borderColor=_RED, borderWidth=1, borderPadding=6,
        backColor=colors.HexColor("#FFEBEE"),
    )

    story = []

    # ---- Sub-header (scan info) ----------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    pid_str   = f"  |  Scan / Patient ID: <b>{patient_id}</b>" if patient_id else ""
    story.append(Paragraph(
        f"Generated: {timestamp}{pid_str}", body
    ))
    story.append(Spacer(1, 0.3 * cm))

    # ---- Section 1 — Images -------------------------------------------
    story.append(Paragraph("Scan Visualisations", h1))

    img_w_cm = (content_width / cm - 0.6) / 3   # 3 images with small gaps
    img_h_cm = img_w_cm * (224 / 224)            # square

    orig_img = _ndarray_to_rl_image(result.original_image,    img_w_cm, img_h_cm)
    heat_img = _ndarray_to_rl_image(result.heatmap_overlay,   img_w_cm, img_h_cm)
    bbox_img = _ndarray_to_rl_image(result.overlay_with_box,  img_w_cm, img_h_cm)

    img_table = Table(
        [[orig_img,                               heat_img,                                     bbox_img],
         [Paragraph("Original OCT Scan", caption_style),
          Paragraph("Grad-CAM Heatmap Overlay", caption_style),
          Paragraph("Heatmap + Bounding Box", caption_style)]],
        colWidths=[img_w_cm * cm] * 3,
    )
    img_table.setStyle(TableStyle([
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 0.4 * cm))

    # ---- Section 2 — Results table ------------------------------------
    story.append(Paragraph("Analysis Results", h1))

    unc_color = _uncertainty_color(result.uncertainty_level)

    def _prob_row(cls):
        p = result.per_class_probs.get(cls, 0.0)
        bar = "█" * int(p * 20)
        return [cls, f"{p*100:.1f}%", bar]

    prob_rows = [_prob_row(c) for c in CLASS_NAMES]

    table_data = [
        ["Field", "Value"],
        ["Predicted Class", result.predicted_label],
        ["Confidence",      f"{result.confidence * 100:.1f}%"],
        ["— CNV prob.",     f"{result.per_class_probs.get('CNV', 0)*100:.1f}%"],
        ["— DME prob.",     f"{result.per_class_probs.get('DME', 0)*100:.1f}%"],
        ["— DRUSEN prob.",  f"{result.per_class_probs.get('DRUSEN', 0)*100:.1f}%"],
        ["— NORMAL prob.",  f"{result.per_class_probs.get('NORMAL', 0)*100:.1f}%"],
        ["Uncertainty Std (pred. class)", f"{result.uncertainty_std:.4f}"],
        ["Predictive Entropy",            f"{result.uncertainty_entropy:.4f} nats"],
        ["Uncertainty Level",             result.uncertainty_level],
        ["MC Dropout Passes",             str(result.mc_passes)],
        ["Model",                         MODEL_NAME],
    ]

    tbl = Table(table_data, colWidths=[content_width * 0.48, content_width * 0.52])
    tbl_style = TableStyle([
        ("BACKGROUND",    (0, 0),  (-1, 0),  _DARK_BLUE),
        ("TEXTCOLOR",     (0, 0),  (-1, 0),  _WHITE),
        ("FONTNAME",      (0, 0),  (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0),  (-1, -1), 8.5),
        ("ROWBACKGROUNDS",(0, 1),  (-1, -1), [_LIGHT_GREY, _WHITE]),
        ("GRID",          (0, 0),  (-1, -1), 0.3, colors.HexColor("#BDBDBD")),
        ("LEFTPADDING",   (0, 0),  (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0),  (-1, -1), 6),
        ("TOPPADDING",    (0, 0),  (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0),  (-1, -1), 3),
    ])

    # Colour the Uncertainty Level row
    for i, row in enumerate(table_data):
        if row[0] == "Uncertainty Level":
            tbl_style.add("TEXTCOLOR",   (1, i), (1, i), unc_color)
            tbl_style.add("FONTNAME",    (1, i), (1, i), "Helvetica-Bold")
        if row[0] == "Predicted Class":
            tbl_style.add("TEXTCOLOR",   (1, i), (1, i), _MED_BLUE)
            tbl_style.add("FONTNAME",    (1, i), (1, i), "Helvetica-Bold")

    tbl.setStyle(tbl_style)
    story.append(tbl)
    story.append(Spacer(1, 0.4 * cm))

    # ---- Section 3 — Interpretation -----------------------------------
    story.append(Paragraph("Interpretation", h1))
    story.append(Paragraph(_class_description(result.predicted_label), body))
    story.append(Spacer(1, 0.2 * cm))

    # Uncertainty caveat (always shown — document uncalibrated status)
    story.append(Paragraph(
        "<i>Note: Uncertainty scores are computed via MC Dropout injected post-training "
        "and are uncalibrated approximations. Thresholds are placeholders only.</i>",
        body,
    ))

    # Human review warning box (conditional)
    if result.review_recommended:
        story.append(Spacer(1, 0.2 * cm))
        reason = []
        if result.confidence < CONFIDENCE_REVIEW_THRESHOLD:
            reason.append(f"low confidence ({result.confidence*100:.1f}%)")
        if result.uncertainty_level == "HIGH":
            reason.append("high uncertainty")
        story.append(Paragraph(
            f"⚠  HUMAN REVIEW RECOMMENDED — This scan was flagged due to: "
            f"{' and '.join(reason)}. Do not act on this result without "
            f"manual verification by a qualified ophthalmologist.",
            warning_style,
        ))

    doc.build(story)
    return buf.getvalue()


def save_pdf(
    result,
    path: Path | str,
    patient_id: Optional[str] = None,
) -> None:
    """
    Generate a PDF and write it to *path*.  Creates parent directories.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf_bytes = generate_pdf(result, patient_id=patient_id)
    path.write_bytes(pdf_bytes)
    print(f"[report.py] PDF saved → {path}")
