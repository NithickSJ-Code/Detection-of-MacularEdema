from __future__ import annotations

import io
import threading
import time
from pathlib import Path

import pytest

from backend.tests.conftest import FakeResult, color_png_bytes, gray_png_bytes


def test_health(client, monkeypatch):
    monkeypatch.setattr("backend.services.ood_stage2_active", lambda: False)
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["model_loaded"] is True
    assert body["ood_stage2_active"] is False
    assert "mc_passes" in body
    assert body["model_name"]


def test_analyze_mocked_ok(client, monkeypatch, tmp_env):
    monkeypatch.setattr("backend.services.analyze_image", lambda _img: FakeResult("OK"))
    monkeypatch.setattr("backend.services.save_pdf", lambda *a, **k: Path(k.get("path", a[1])).write_bytes(b"%PDF-1.4 mock"))
    files = {"image": ("scan.png", gray_png_bytes(), "image/png")}
    res = client.post("/api/analyze", files=files, data={"scan_id": "SCAN001"})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "OK"
    assert body["predicted_label"] == "DME"
    assert body["scan_id"] == "SCAN001"
    assert body["report_url"].endswith("/report.pdf")
    pdf = client.get(body["report_url"])
    assert pdf.status_code == 200
    assert pdf.headers["content-type"].startswith("application/pdf")
    img = client.get(body["images"]["gradcam"])
    assert img.status_code == 200


def test_rejected_image_no_pdf(client, monkeypatch):
    monkeypatch.setattr("backend.services.analyze_image", lambda _img: FakeResult("INVALID_INPUT"))
    files = {"image": ("photo.png", color_png_bytes(), "image/png")}
    res = client.post("/api/analyze", files=files)
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "INVALID_INPUT"
    assert body["rejection_reason"]
    assert body["report_url"] is None
    stats = client.get("/api/stats").json()
    assert stats["rejected_uploads"] == 1
    assert stats["ok_scans"] == 0
    reports = client.get("/api/reports").json()
    assert reports["total"] == 0


def test_non_image_rejected(client):
    files = {"image": ("scan.png", b"this is not an image", "image/png")}
    res = client.post("/api/analyze", files=files)
    assert res.status_code == 400
    assert "valid image" in res.json()["detail"].lower() or "not a valid" in res.json()["detail"].lower()


def test_oversized_rejected(client, monkeypatch):
    monkeypatch.setattr("backend.settings.MAX_UPLOAD_BYTES", 1024)
    monkeypatch.setattr("backend.app.MAX_UPLOAD_BYTES", 1024)
    data = b"x" * 2048
    files = {"image": ("big.png", data, "image/png")}
    res = client.post("/api/analyze", files=files)
    assert res.status_code == 413


def test_list_stats_delete(client, monkeypatch, tmp_env):
    monkeypatch.setattr("backend.services.analyze_image", lambda _img: FakeResult("OK"))
    monkeypatch.setattr("backend.services.save_pdf", lambda *a, **k: Path(a[1]).write_bytes(b"%PDF-1.4"))
    files = {"image": ("scan.png", gray_png_bytes(), "image/png")}
    created = client.post("/api/analyze", files=files, data={"scan_id": "HIST-1"}).json()
    listed = client.get("/api/scans", params={"q": "HIST"}).json()
    assert listed["total"] >= 1
    stats = client.get("/api/stats").json()
    assert stats["total_scans"] >= 1
    assert stats["per_class"]
    deleted = client.delete(f"/api/scans/{created['id']}")
    assert deleted.status_code == 200
    missing = client.get(f"/api/scans/{created['id']}")
    assert missing.status_code == 404


def test_inference_lock_serialises(client, monkeypatch):
    calls = []
    lock_seen = []

    def slow(_img):
        calls.append(time.time())
        time.sleep(0.25)
        return FakeResult("OK")

    monkeypatch.setattr("backend.services.analyze_image", slow)
    monkeypatch.setattr("backend.services.save_pdf", lambda *a, **k: Path(a[1]).write_bytes(b"%PDF-1.4"))

    results = []

    def worker():
        files = {"image": ("scan.png", gray_png_bytes(), "image/png")}
        results.append(client.post("/api/analyze", files=files).status_code)

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert results == [200, 200]
    assert len(calls) == 2
    # Second start should not overlap the first 0.25s window by much
    assert abs(calls[1] - calls[0]) >= 0.2


@pytest.mark.slow
def test_real_colorful_rejection_end_to_end(tmp_path, monkeypatch):
    """Uses the real validation+model path on a colourful synthetic image."""
    monkeypatch.delenv("OCT_SKIP_MODEL_LOAD", raising=False)
    storage = tmp_path / "storage"
    reports = tmp_path / "reports"
    storage.mkdir()
    reports.mkdir()
    monkeypatch.setenv("OCT_STORAGE_DIR", str(storage))
    monkeypatch.setenv("OCT_DB_PATH", str(storage / "oct.db"))
    monkeypatch.setenv("OCT_REPORTS_DIR", str(reports))
    import backend.settings as settings
    import backend.db as db
    import backend.services as services

    settings.STORAGE_DIR = storage
    settings.DB_PATH = storage / "oct.db"
    settings.REPORTS_DIR = reports
    db.DB_PATH = settings.DB_PATH
    db.STORAGE_DIR = storage
    services.STORAGE_DIR = storage
    services.REPORTS_DIR = reports
    db.init_db()

    from fastapi.testclient import TestClient
    import backend.app as appmod

    appmod._STATE["model_loaded"] = False
    with TestClient(appmod.app) as c:
        files = {"image": ("photo.png", color_png_bytes(), "image/png")}
        res = c.post("/api/analyze", files=files)
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "INVALID_INPUT"
        assert body["report_url"] is None
