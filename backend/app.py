"""FastAPI application wrapping src.inference.analyze_image."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import os

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.db import init_db
from backend.schemas import HealthResponse, ScanRecord
from backend import services
from backend.settings import (
    API_VERSION,
    CORS_ORIGINS,
    FRONTEND_DIST,
    MAX_UPLOAD_BYTES,
    SERVE_FRONTEND,
)
from src.config import MC_T, MODEL_NAME

log = logging.getLogger("oct.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

_STATE = {
    "model_loaded": False,
    "device": "cpu",
    "model_error": None,
}


def _try_load_model() -> None:
    if os.getenv("OCT_SKIP_MODEL_LOAD") in {"1", "true", "True"}:
        _STATE["model_loaded"] = True
        _STATE["device"] = "cpu"
        _STATE["model_error"] = None
        return
    try:
        from src.inference import _get_model

        model = _get_model()
        _STATE["model_loaded"] = True
        _STATE["device"] = str(next(model.parameters()).device)
        _STATE["model_error"] = None
    except Exception as exc:
        log.exception("Model failed to load")
        _STATE["model_loaded"] = False
        _STATE["device"] = "unknown"
        _STATE["model_error"] = str(exc)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    _try_load_model()
    yield


app = FastAPI(
    title="OCT Macular Edema Detection API",
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_error(_request: Request, exc: Exception):
    if isinstance(exc, (HTTPException, StarletteHTTPException, RequestValidationError)):
        raise exc
    log.exception("Unhandled server error")
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if not _STATE["model_loaded"]:
        _try_load_model()
    return HealthResponse(
        status="ok" if _STATE["model_loaded"] else "degraded",
        model_loaded=bool(_STATE["model_loaded"]),
        device=_STATE["device"],
        model_name=MODEL_NAME,
        mc_passes=MC_T,
        ood_stage2_active=services.ood_stage2_active(),
        version=API_VERSION,
        model_error=_STATE["model_error"],
    )


@app.post("/api/analyze", response_model=ScanRecord)
async def analyze(
    image: UploadFile = File(...),
    scan_id: Optional[str] = Form(None),
    source: str = Form("upload"),
) -> ScanRecord:
    if not _STATE["model_loaded"]:
        _try_load_model()
    if not _STATE["model_loaded"]:
        raise HTTPException(
            status_code=503,
            detail="AI model is offline. Analysis is disabled until the checkpoint loads.",
        )
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
        )
    try:
        return services.run_analysis(
            data=data,
            filename=image.filename or "upload.png",
            content_type=image.content_type,
            scan_id=scan_id,
            source=source,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/scans")
def scans(
    q: Optional[str] = None,
    label: Optional[str] = None,
    review: Optional[bool] = None,
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    sort: str = "created_at_desc",
):
    return services.list_scans(
        q=q, label=label, review=review, limit=limit, offset=offset, sort=sort
    )


@app.get("/api/scans/{scan_uuid}", response_model=ScanRecord)
def scan_detail(scan_uuid: str) -> ScanRecord:
    rec = services.get_scan(scan_uuid)
    if rec is None:
        raise HTTPException(status_code=404, detail="Scan not found.")
    return rec


@app.delete("/api/scans/{scan_uuid}")
def scan_delete(scan_uuid: str):
    ok = services.delete_scan(scan_uuid)
    if not ok:
        raise HTTPException(status_code=404, detail="Scan not found.")
    return {"ok": True}


@app.get("/api/stats")
def stats():
    return services.compute_stats()


@app.get("/api/reports")
def reports():
    return services.list_reports()


@app.get("/api/scans/{scan_uuid}/report.pdf")
def scan_report(scan_uuid: str, download: int = 0):
    path = services.report_file(scan_uuid)
    if path is None:
        raise HTTPException(status_code=404, detail="Report not found for this scan.")
    rec = services.get_scan(scan_uuid)
    filename = f"{rec.scan_id if rec else scan_uuid}_report.pdf"
    disposition = "attachment" if download else "inline"
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=filename,
        headers={"Content-Disposition": f'{disposition}; filename="{filename}"'},
    )


@app.get("/api/scans/{scan_uuid}/images/{kind}.png")
def scan_image(scan_uuid: str, kind: str):
    path = services.image_path(scan_uuid, kind)
    if path is None:
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(path, media_type="image/png")


if SERVE_FRONTEND and FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
