"""Shared fixtures. Tests use a temp SQLite DB and never write to the real storage tree."""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def tmp_env(tmp_path, monkeypatch):
    storage = tmp_path / "storage"
    reports = tmp_path / "reports"
    storage.mkdir()
    reports.mkdir()
    monkeypatch.setenv("OCT_STORAGE_DIR", str(storage))
    monkeypatch.setenv("OCT_DB_PATH", str(storage / "oct.db"))
    monkeypatch.setenv("OCT_REPORTS_DIR", str(reports))
    monkeypatch.setenv("OCT_SERVE_FRONTEND", "0")
    monkeypatch.setenv("OCT_SKIP_MODEL_LOAD", "1")
    import backend.settings as settings

    settings.STORAGE_DIR = storage
    settings.DB_PATH = storage / "oct.db"
    settings.REPORTS_DIR = reports
    settings.SERVE_FRONTEND = False
    import backend.db as db
    import backend.services as services

    db.DB_PATH = settings.DB_PATH
    db.STORAGE_DIR = storage
    services.STORAGE_DIR = storage
    services.REPORTS_DIR = reports
    db.init_db()
    return {"storage": storage, "reports": reports}


@pytest.fixture()
def client(tmp_env):
    from fastapi.testclient import TestClient
    from backend.app import app, _STATE

    _STATE["model_loaded"] = True
    _STATE["device"] = "cpu"
    _STATE["model_error"] = None
    with TestClient(app) as c:
        yield c


def gray_png_bytes(size=224, noise=True) -> bytes:
    arr = np.full((size, size, 3), 80, dtype=np.uint8)
    if noise:
        rng = np.random.default_rng(0)
        arr = np.clip(arr + rng.integers(-20, 21, arr.shape), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def color_png_bytes(size=224) -> bytes:
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[:, :, 0] = 255
    arr[:, : size // 2, 1] = 200
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


class FakeResult:
    def __init__(self, status="OK"):
        self.status = status
        self.rejection_reason = None if status == "OK" else "Color check failed: test"
        self.predicted_label = "DME" if status == "OK" else None
        self.confidence = 0.91 if status == "OK" else None
        self.per_class_probs = (
            {"CNV": 0.03, "DME": 0.91, "DRUSEN": 0.03, "NORMAL": 0.03} if status == "OK" else None
        )
        self.uncertainty_std = 0.03 if status == "OK" else None
        self.uncertainty_entropy = 0.4 if status == "OK" else None
        self.uncertainty_level = "LOW" if status == "OK" else None
        self.mc_passes = 20 if status == "OK" else None
        self.review_recommended = False if status == "OK" else None
        img = np.full((224, 224, 3), 90, dtype=np.uint8)
        self.original_image = img if status == "OK" else None
        self.heatmap_overlay = img.copy() if status == "OK" else None
        self.overlay_with_box = img.copy() if status == "OK" else None
        self.cam_array = np.zeros((224, 224), dtype=np.float32) if status == "OK" else None
        self.box_coords = (10, 12, 40, 50) if status == "OK" else None
