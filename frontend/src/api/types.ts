export type UncertaintyLevel = 'LOW' | 'MODERATE' | 'HIGH'
export type ScanStatus = 'OK' | 'INVALID_INPUT'
export type ClassLabel = 'CNV' | 'DME' | 'DRUSEN' | 'NORMAL'

export interface HealthResponse {
  status: string
  model_loaded: boolean
  device: string
  model_name: string
  mc_passes: number
  ood_stage2_active: boolean
  version: string
  model_error: string | null
}

export interface ScanImages {
  original: string | null
  gradcam: string | null
  boxed: string | null
}

export interface ScanRecord {
  id: string
  scan_id: string
  status: ScanStatus
  rejection_reason: string | null
  predicted_label: ClassLabel | null
  confidence: number | null
  per_class_probs: Record<string, number> | null
  uncertainty_level: UncertaintyLevel | null
  uncertainty_std: number | null
  uncertainty_entropy: number | null
  mc_passes: number | null
  review_recommended: boolean | null
  review_reasons: string[]
  box_coords: number[] | null
  source: 'upload' | 'webcam' | string
  images: ScanImages
  report_url: string | null
  created_at: string
  processing_ms: number
}

export interface ScanListResponse {
  items: ScanRecord[]
  total: number
  limit: number
  offset: number
}

export interface ClassCount {
  label: string
  count: number
  percent: number
}

export interface DayCount {
  date: string
  count: number
}

export interface StatsResponse {
  total_scans: number
  ok_scans: number
  rejected_uploads: number
  normal_count: number
  review_count: number
  review_rate: number
  mean_confidence: number | null
  per_class: ClassCount[]
  uncertainty_levels: Record<string, number>
  scans_per_day: DayCount[]
  recent: ScanRecord[]
}

export interface ReportItem {
  scan_id: string
  id: string
  report_url: string
  size_bytes: number
  created_at: string
  predicted_label: string | null
}

export interface ReportListResponse {
  items: ReportItem[]
  total: number
}
