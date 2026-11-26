"""Environment-driven settings with safe defaults."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_cors = os.getenv("OCT_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
CORS_ORIGINS: list[str] = [o.strip() for o in _cors.split(",") if o.strip()]

STORAGE_DIR = Path(os.getenv("OCT_STORAGE_DIR", str(ROOT / "backend" / "storage")))
DB_PATH = Path(os.getenv("OCT_DB_PATH", str(STORAGE_DIR / "oct.db")))
REPORTS_DIR = Path(os.getenv("OCT_REPORTS_DIR", str(ROOT / "reports")))
MAX_UPLOAD_BYTES = int(os.getenv("OCT_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
API_VERSION = os.getenv("OCT_API_VERSION", "1.0.0")
SERVE_FRONTEND = os.getenv("OCT_SERVE_FRONTEND", "0") in {"1", "true", "True", "yes"}
FRONTEND_DIST = Path(os.getenv("OCT_FRONTEND_DIST", str(ROOT / "frontend" / "dist")))

ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/bmp",
    "image/x-ms-bmp",
    "image/x-windows-bmp",
}
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
MAX_IMAGE_PIXELS = int(os.getenv("OCT_MAX_IMAGE_PIXELS", "40000000"))
SCAN_ID_MAX_LEN = 64
