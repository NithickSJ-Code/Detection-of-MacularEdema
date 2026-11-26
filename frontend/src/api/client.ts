import type {
  HealthResponse,
  ReportListResponse,
  ScanListResponse,
  ScanRecord,
  StatsResponse,
} from './types'

export const API_BASE = import.meta.env.VITE_API_URL ?? ''

function apiUrl(path: string): string {
  return `${API_BASE}${path}`
}

async function readError(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as { detail?: unknown }
    if (typeof body.detail === 'string') return body.detail
    if (Array.isArray(body.detail)) return JSON.stringify(body.detail)
  } catch {
    /* ignore */
  }
  return res.statusText || 'Request failed'
}

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(apiUrl(path), init)
  if (!res.ok) throw new Error(await readError(res))
  return (await res.json()) as T
}

export function assetUrl(path: string | null | undefined): string {
  if (!path) return ''
  return apiUrl(path)
}

export function fetchHealth(): Promise<HealthResponse> {
  return json('/api/health')
}

export function fetchStats(): Promise<StatsResponse> {
  return json('/api/stats')
}

export function fetchScan(id: string): Promise<ScanRecord> {
  return json(`/api/scans/${id}`)
}

export function fetchScans(params: {
  q?: string
  label?: string
  review?: string
  limit?: number
  offset?: number
  sort?: string
}): Promise<ScanListResponse> {
  const q = new URLSearchParams()
  if (params.q) q.set('q', params.q)
  if (params.label) q.set('label', params.label)
  if (params.review) q.set('review', params.review)
  if (params.limit != null) q.set('limit', String(params.limit))
  if (params.offset != null) q.set('offset', String(params.offset))
  if (params.sort) q.set('sort', params.sort)
  const qs = q.toString()
  return json(`/api/scans${qs ? `?${qs}` : ''}`)
}

export function deleteScan(id: string): Promise<{ ok: boolean }> {
  return json(`/api/scans/${id}`, { method: 'DELETE' })
}

export function fetchReports(): Promise<ReportListResponse> {
  return json('/api/reports')
}

export async function analyzeScan(
  file: Blob,
  filename: string,
  scanId: string,
  source: 'upload' | 'webcam',
  signal?: AbortSignal,
): Promise<ScanRecord> {
  const form = new FormData()
  form.append('image', file, filename)
  if (scanId.trim()) form.append('scan_id', scanId.trim())
  form.append('source', source)
  const res = await fetch(apiUrl('/api/analyze'), { method: 'POST', body: form, signal })
  if (!res.ok) throw new Error(await readError(res))
  return (await res.json()) as ScanRecord
}
