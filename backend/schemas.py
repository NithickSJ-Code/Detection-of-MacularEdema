"""Pydantic models for the OCT API."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_name: str
    mc_passes: int
    ood_stage2_active: bool
    version: str
    model_error: Optional[str] = None


class ScanImages(BaseModel):
    original: Optional[str] = None
    gradcam: Optional[str] = None
    boxed: Optional[str] = None


class ScanRecord(BaseModel):
    id: str
    scan_id: str
    status: str
    rejection_reason: Optional[str] = None
    predicted_label: Optional[str] = None
    confidence: Optional[float] = None
    per_class_probs: Optional[dict[str, float]] = None
    uncertainty_level: Optional[str] = None
    uncertainty_std: Optional[float] = None
    uncertainty_entropy: Optional[float] = None
    mc_passes: Optional[int] = None
    review_recommended: Optional[bool] = None
    review_reasons: list[str] = Field(default_factory=list)
    box_coords: Optional[list[int]] = None
    source: str = "upload"
    images: ScanImages
    report_url: Optional[str] = None
    created_at: str
    processing_ms: int


class ScanListResponse(BaseModel):
    items: list[ScanRecord]
    total: int
    limit: int
    offset: int


class ClassCount(BaseModel):
    label: str
    count: int
    percent: float


class DayCount(BaseModel):
    date: str
    count: int


class StatsResponse(BaseModel):
    total_scans: int
    ok_scans: int
    rejected_uploads: int
    normal_count: int
    review_count: int
    review_rate: float
    mean_confidence: Optional[float] = None
    per_class: list[ClassCount]
    uncertainty_levels: dict[str, int]
    scans_per_day: list[DayCount]
    recent: list[ScanRecord]


class ReportItem(BaseModel):
    scan_id: str
    id: str
    report_url: str
    size_bytes: int
    created_at: str
    predicted_label: Optional[str] = None


class ReportListResponse(BaseModel):
    items: list[ReportItem]
    total: int


class ErrorDetail(BaseModel):
    detail: str
    code: Optional[str] = None
    extra: Optional[dict[str, Any]] = None
