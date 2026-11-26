"""Glue: run analysis, persist images/PDF/DB row, assemble API records."""
from __future__ import annotations

import io
import json
import logging
import re
import shutil
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from PIL import Image, UnidentifiedImageError

from backend.db import get_conn, row_to_dict
from backend.schemas import (
    ClassCount,
    DayCount,
    ReportItem,
    ReportListResponse,
    ScanImages,
    ScanListResponse,
    ScanRecord,
    StatsResponse,
)
from backend.settings import (
    ALLOWED_CONTENT_TYPES,
    ALLOWED_EXTENSIONS,
    MAX_IMAGE_PIXELS,
    MAX_UPLOAD_BYTES,
    REPORTS_DIR,
    SCAN_ID_MAX_LEN,
    STORAGE_DIR,
)
from src.config import (
    CLASS_NAMES,
    CONFIDENCE_REVIEW_THRESHOLD,
    MODEL_NAME,
    OOD_STATS_PATH,
)
from src.inference import analyze_image
from src.report import save_pdf

log = logging.getLogger("oct.api")

INFERENCE_LOCK = threading.Lock()

DISALLOWED_SCAN_ID = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


def ood_stage2_active() -> bool:
    return OOD_STATS_PATH.is_file()


def review_reasons_for(confidence: Optional[float], uncertainty_level: Optional[str]) -> list[str]:
    reasons: list[str] = []
    if confidence is not None and confidence < CONFIDENCE_REVIEW_THRESHOLD:
        pct = int(CONFIDENCE_REVIEW_THRESHOLD * 100)
        reasons.append(f"Confidence below {pct}% threshold")
    if uncertainty_level == "HIGH":
        reasons.append("High prediction uncertainty")
    return reasons


def sanitize_scan_id(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    s = raw.strip()
    if not s:
        return None
    s = DISALLOWED_SCAN_ID.sub("", s)
    s = s[:SCAN_ID_MAX_LEN].strip()
    return s or None


def _today_prefix() -> str:
    return datetime.now().strftime("SCAN-%Y%m%d-")


def allocate_scan_id(preferred: Optional[str]) -> str:
    preferred = sanitize_scan_id(preferred)
    with get_conn() as conn:
        if preferred:
            exists = conn.execute(
                "SELECT 1 FROM scans WHERE scan_id = ?", (preferred,)
            ).fetchone()
            if not exists:
                return preferred
            for i in range(2, 1000):
                candidate = f"{preferred}-{i}"[:SCAN_ID_MAX_LEN]
                exists = conn.execute(
                    "SELECT 1 FROM scans WHERE scan_id = ?", (candidate,)
                ).fetchone()
                if not exists:
                    return candidate
        prefix = _today_prefix()
        rows = conn.execute(
            "SELECT scan_id FROM scans WHERE scan_id LIKE ? ORDER BY scan_id DESC",
            (prefix + "%",),
        ).fetchall()
        max_n = 0
        for row in rows:
            tail = str(row["scan_id"])[len(prefix) :]
            if tail.isdigit():
                max_n = max(max_n, int(tail))
        return f"{prefix}{max_n + 1:03d}"


def decode_image(data: bytes, filename: str, content_type: Optional[str]) -> Image.Image:
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.")
    ext = Path(filename or "").suffix.lower()
    if ext and ext not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported file type. Please upload a PNG, JPEG, or BMP image.")
    if content_type and content_type.lower() not in ALLOWED_CONTENT_TYPES and content_type.lower() != "application/octet-stream":
        raise ValueError("Unsupported content type. Please upload a PNG, JPEG, or BMP image.")

    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        image = Image.open(io.BytesIO(data))
        image.load()
    except UnidentifiedImageError as exc:
        raise ValueError("File is not a valid image. Please upload a real OCT scan image.") from exc
    except Image.DecompressionBombError as exc:
        raise ValueError("Image is too large to process safely.") from exc
    except Exception as exc:
        raise ValueError("Could not read the uploaded file as an image.") from exc
    return image


def _image_urls(scan_uuid: str, status: str) -> ScanImages:
    base = f"/api/scans/{scan_uuid}/images"
    if status == "OK":
        return ScanImages(
            original=f"{base}/original.png",
            gradcam=f"{base}/gradcam.png",
            boxed=f"{base}/boxed.png",
        )
    orig = STORAGE_DIR / "scans" / scan_uuid / "original.png"
    return ScanImages(original=f"{base}/original.png" if orig.is_file() else None)


def record_from_row(row: dict) -> ScanRecord:
    scan_uuid = row["id"]
    status = row["status"]
    report_url = None
    if row.get("report_path"):
        report_url = f"/api/scans/{scan_uuid}/report.pdf"
    probs = json.loads(row["probs_json"]) if row.get("probs_json") else None
    reasons = json.loads(row["review_reasons_json"]) if row.get("review_reasons_json") else []
    box = json.loads(row["box_json"]) if row.get("box_json") else None
    review = row.get("review_recommended")
    return ScanRecord(
        id=scan_uuid,
        scan_id=row["scan_id"],
        status=status,
        rejection_reason=row.get("rejection_reason"),
        predicted_label=row.get("predicted_label"),
        confidence=row.get("confidence"),
        per_class_probs=probs,
        uncertainty_level=row.get("uncertainty_level"),
        uncertainty_std=row.get("uncertainty_std"),
        uncertainty_entropy=row.get("uncertainty_entropy"),
        mc_passes=row.get("mc_passes"),
        review_recommended=bool(review) if review is not None else None,
        review_reasons=reasons,
        box_coords=box,
        source=row.get("source") or "upload",
        images=_image_urls(scan_uuid, status),
        report_url=report_url,
        created_at=row["created_at"],
        processing_ms=row.get("processing_ms") or 0,
    )


def _save_png(arr, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype("uint8")).save(path, format="PNG")


def run_analysis(
    data: bytes,
    filename: str,
    content_type: Optional[str],
    scan_id: Optional[str],
    source: str = "upload",
) -> ScanRecord:
    pil = decode_image(data, filename, content_type)
    source = source if source in {"upload", "webcam"} else "upload"
    scan_uuid = str(uuid.uuid4())
    public_id = allocate_scan_id(scan_id)
    scan_dir = STORAGE_DIR / "scans" / scan_uuid
    scan_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    with INFERENCE_LOCK:
        result = analyze_image(pil)
    processing_ms = int((time.perf_counter() - t0) * 1000)

    original_path = scan_dir / "original.png"
    try:
        if result.status == "OK" and result.original_image is not None:
            _save_png(result.original_image, original_path)
            _save_png(result.heatmap_overlay, scan_dir / "gradcam.png")
            _save_png(result.overlay_with_box, scan_dir / "boxed.png")
        else:
            pil.convert("RGB").save(original_path, format="PNG")
    except Exception:
        log.exception("Failed to persist scan images for %s", scan_uuid)

    report_path = None
    reasons: list[str] = []
    if result.status == "OK":
        reasons = review_reasons_for(result.confidence, result.uncertainty_level)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = str(REPORTS_DIR / f"{scan_uuid}.pdf")
        save_pdf(result, report_path, patient_id=public_id)

    created_at = datetime.now(timezone.utc).isoformat()
    box = list(result.box_coords) if result.box_coords else None
    row = {
        "id": scan_uuid,
        "scan_id": public_id,
        "status": result.status,
        "rejection_reason": result.rejection_reason,
        "predicted_label": result.predicted_label,
        "confidence": result.confidence,
        "probs_json": json.dumps(result.per_class_probs) if result.per_class_probs else None,
        "uncertainty_level": result.uncertainty_level,
        "uncertainty_std": result.uncertainty_std,
        "uncertainty_entropy": result.uncertainty_entropy,
        "mc_passes": result.mc_passes,
        "review_recommended": (
            int(bool(result.review_recommended)) if result.review_recommended is not None else None
        ),
        "review_reasons_json": json.dumps(reasons),
        "box_json": json.dumps(box) if box else None,
        "source": source,
        "report_path": report_path,
        "processing_ms": processing_ms,
        "created_at": created_at,
    }
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO scans (
                id, scan_id, status, rejection_reason, predicted_label, confidence,
                probs_json, uncertainty_level, uncertainty_std, uncertainty_entropy,
                mc_passes, review_recommended, review_reasons_json, box_json, source,
                report_path, processing_ms, created_at
            ) VALUES (
                :id, :scan_id, :status, :rejection_reason, :predicted_label, :confidence,
                :probs_json, :uncertainty_level, :uncertainty_std, :uncertainty_entropy,
                :mc_passes, :review_recommended, :review_reasons_json, :box_json, :source,
                :report_path, :processing_ms, :created_at
            )
            """,
            row,
        )
    return record_from_row(row)


def get_scan(scan_uuid: str) -> Optional[ScanRecord]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM scans WHERE id = ?", (scan_uuid,)).fetchone()
    if row is None:
        return None
    return record_from_row(row_to_dict(row))


def list_scans(
    q: Optional[str] = None,
    label: Optional[str] = None,
    review: Optional[bool] = None,
    limit: int = 20,
    offset: int = 0,
    sort: str = "created_at_desc",
) -> ScanListResponse:
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    clauses = ["1=1"]
    params: list = []
    if q:
        clauses.append("scan_id LIKE ?")
        params.append(f"%{q.strip()}%")
    if label:
        clauses.append("predicted_label = ?")
        params.append(label.upper())
    if review is True:
        clauses.append("review_recommended = 1")
    elif review is False:
        clauses.append("review_recommended = 0")

    order = {
        "created_at_desc": "created_at DESC",
        "created_at_asc": "created_at ASC",
        "confidence_desc": "confidence DESC",
        "confidence_asc": "confidence ASC",
        "scan_id_asc": "scan_id ASC",
        "scan_id_desc": "scan_id DESC",
    }.get(sort, "created_at DESC")

    where = " AND ".join(clauses)
    with get_conn() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) AS n FROM scans WHERE {where}", params
        ).fetchone()["n"]
        rows = conn.execute(
            f"SELECT * FROM scans WHERE {where} ORDER BY {order} LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()
    items = [record_from_row(row_to_dict(r)) for r in rows]
    return ScanListResponse(items=items, total=total, limit=limit, offset=offset)


def delete_scan(scan_uuid: str) -> bool:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM scans WHERE id = ?", (scan_uuid,)).fetchone()
        if row is None:
            return False
        report_path = row["report_path"]
        conn.execute("DELETE FROM scans WHERE id = ?", (scan_uuid,))
    scan_dir = STORAGE_DIR / "scans" / scan_uuid
    if scan_dir.exists():
        shutil.rmtree(scan_dir, ignore_errors=True)
    if report_path:
        Path(report_path).unlink(missing_ok=True)
    return True


def image_path(scan_uuid: str, kind: str) -> Optional[Path]:
    if kind not in {"original", "gradcam", "boxed"}:
        return None
    path = STORAGE_DIR / "scans" / scan_uuid / f"{kind}.png"
    return path if path.is_file() else None


def report_file(scan_uuid: str) -> Optional[Path]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT report_path FROM scans WHERE id = ?", (scan_uuid,)
        ).fetchone()
    if not row or not row["report_path"]:
        return None
    path = Path(row["report_path"])
    return path if path.is_file() else None


def list_reports() -> ReportListResponse:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, scan_id, report_path, created_at, predicted_label
            FROM scans
            WHERE report_path IS NOT NULL
            ORDER BY created_at DESC
            """
        ).fetchall()
    items: list[ReportItem] = []
    for row in rows:
        path = Path(row["report_path"]) if row["report_path"] else None
        if not path or not path.is_file():
            continue
        items.append(
            ReportItem(
                id=row["id"],
                scan_id=row["scan_id"],
                report_url=f"/api/scans/{row['id']}/report.pdf",
                size_bytes=path.stat().st_size,
                created_at=row["created_at"],
                predicted_label=row["predicted_label"],
            )
        )
    return ReportListResponse(items=items, total=len(items))


def compute_stats() -> StatsResponse:
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) AS n FROM scans").fetchone()["n"]
        rejected = conn.execute(
            "SELECT COUNT(*) AS n FROM scans WHERE status = 'INVALID_INPUT'"
        ).fetchone()["n"]
        ok = conn.execute(
            "SELECT COUNT(*) AS n FROM scans WHERE status = 'OK'"
        ).fetchone()["n"]
        normal = conn.execute(
            "SELECT COUNT(*) AS n FROM scans WHERE status = 'OK' AND predicted_label = 'NORMAL'"
        ).fetchone()["n"]
        review = conn.execute(
            "SELECT COUNT(*) AS n FROM scans WHERE status = 'OK' AND review_recommended = 1"
        ).fetchone()["n"]
        mean_row = conn.execute(
            "SELECT AVG(confidence) AS m FROM scans WHERE status = 'OK' AND confidence IS NOT NULL"
        ).fetchone()
        class_rows = conn.execute(
            """
            SELECT predicted_label AS label, COUNT(*) AS n
            FROM scans
            WHERE status = 'OK' AND predicted_label IS NOT NULL
            GROUP BY predicted_label
            """
        ).fetchall()
        unc_rows = conn.execute(
            """
            SELECT uncertainty_level AS lvl, COUNT(*) AS n
            FROM scans
            WHERE status = 'OK' AND uncertainty_level IS NOT NULL
            GROUP BY uncertainty_level
            """
        ).fetchall()
        recent_rows = conn.execute(
            "SELECT * FROM scans ORDER BY created_at DESC LIMIT 5"
        ).fetchall()

    counts = {r["label"]: r["n"] for r in class_rows}
    per_class = []
    for name in CLASS_NAMES:
        n = counts.get(name, 0)
        percent = (n / ok * 100.0) if ok else 0.0
        per_class.append(ClassCount(label=name, count=n, percent=round(percent, 1)))

    unc = {"LOW": 0, "MODERATE": 0, "HIGH": 0}
    for r in unc_rows:
        if r["lvl"] in unc:
            unc[r["lvl"]] = r["n"]

    days: list[DayCount] = []
    today = datetime.now(timezone.utc).date()
    with get_conn() as conn:
        for i in range(13, -1, -1):
            day = today - timedelta(days=i)
            start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc).isoformat()
            end = datetime(day.year, day.month, day.day, 23, 59, 59, tzinfo=timezone.utc).isoformat()
            n = conn.execute(
                "SELECT COUNT(*) AS n FROM scans WHERE created_at >= ? AND created_at <= ?",
                (start, end),
            ).fetchone()["n"]
            days.append(DayCount(date=day.isoformat(), count=n))

    mean_conf = mean_row["m"] if mean_row else None
    review_rate = (review / ok) if ok else 0.0
    return StatsResponse(
        total_scans=total,
        ok_scans=ok,
        rejected_uploads=rejected,
        normal_count=normal,
        review_count=review,
        review_rate=round(review_rate, 4),
        mean_confidence=float(mean_conf) if mean_conf is not None else None,
        per_class=per_class,
        uncertainty_levels=unc,
        scans_per_day=days,
        recent=[record_from_row(row_to_dict(r)) for r in recent_rows],
    )


def model_device_name() -> str:
    try:
        from src.inference import _get_model

        model = _get_model()
        return str(next(model.parameters()).device)
    except Exception:
        return "unknown"


__all__ = [
    "MODEL_NAME",
    "INFERENCE_LOCK",
    "analyze_image",
    "compute_stats",
    "decode_image",
    "delete_scan",
    "get_scan",
    "image_path",
    "list_reports",
    "list_scans",
    "model_device_name",
    "ood_stage2_active",
    "report_file",
    "run_analysis",
]
