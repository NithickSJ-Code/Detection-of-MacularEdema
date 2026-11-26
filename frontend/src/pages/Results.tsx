import { useQuery } from '@tanstack/react-query'
import { Info } from 'lucide-react'
import { useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { assetUrl, fetchScan } from '../api/client'
import type { ClassLabel } from '../api/types'
import { PageHeader } from '../components/Layout'
import { CLASS_DESCRIPTIONS, CLASS_ORDER, DISCLAIMER, formatPct, uncertaintyCopy } from '../lib/labels'

export function ResultsPage() {
  const { id = '' } = useParams()
  const nav = useNavigate()
  const [zoom, setZoom] = useState<string | null>(null)
  const [heatOpacity, setHeatOpacity] = useState(0.55)
  const q = useQuery({ queryKey: ['scan', id], queryFn: () => fetchScan(id), enabled: Boolean(id) })

  if (q.isLoading) return <p>Loading result…</p>
  if (q.isError) return <p>Could not load this scan.</p>
  const scan = q.data!
  const pdf = scan.report_url ? assetUrl(scan.report_url) : null

  if (scan.status === 'INVALID_INPUT') {
    return (
      <>
        <PageHeader title="Not a valid OCT scan" subtitle="Please retake or choose a different image." />
        <section className="card">
          <div className="banner bad" role="alert">
            {scan.rejection_reason || 'The validation gate rejected this image.'}
          </div>
          {scan.images.original ? (
            <img className="preview-img" src={assetUrl(scan.images.original)} alt="Rejected upload" />
          ) : null}
          <div className="btn-row">
            <button className="btn" type="button" onClick={() => nav('/scan')}>
              Retry
            </button>
            <Link className="btn secondary" to="/history">
              Back to history
            </Link>
          </div>
        </section>
        <p className="clock">{DISCLAIMER}</p>
      </>
    )
  }

  const label = (scan.predicted_label || 'NORMAL') as ClassLabel

  return (
    <>
      <PageHeader
        title={label}
        subtitle={`Scan ${scan.scan_id} · processed in ${scan.processing_ms} ms`}
        actions={
          <div className="btn-row">
            {pdf ? (
              <>
                <a className="btn" href={`${pdf}?download=1`}>
                  Download PDF
                </a>
                <a className="btn secondary" href={pdf} target="_blank" rel="noreferrer">
                  View PDF
                </a>
              </>
            ) : null}
            <button className="btn secondary" type="button" onClick={() => nav('/scan')}>
              New Scan
            </button>
            <button className="btn secondary" type="button" onClick={() => nav('/history')}>
              Back to History
            </button>
          </div>
        }
      />

      <div className="btn-row" style={{ marginBottom: 12 }}>
        <span className={`chip ${label}`}>{label}</span>
        <span className="chip">{formatPct(scan.confidence)} confidence</span>
        {scan.uncertainty_level ? (
          <span className={`chip ${scan.uncertainty_level}`}>{scan.uncertainty_level} uncertainty</span>
        ) : null}
      </div>

      <section className="card">
        <div className="image-panel">
          <figure className="oct-frame" onClick={() => setZoom(assetUrl(scan.images.original))}>
            <img src={assetUrl(scan.images.original)} alt={`Original OCT scan ${scan.scan_id}`} />
            <figcaption>Original</figcaption>
          </figure>
          <figure className="oct-frame" onClick={() => setZoom(assetUrl(scan.images.gradcam))}>
            <div style={{ position: 'relative' }}>
              <img src={assetUrl(scan.images.original)} alt="" />
              <img
                src={assetUrl(scan.images.gradcam)}
                alt={`Grad-CAM overlay for ${scan.scan_id}`}
                style={{ position: 'absolute', inset: 0, opacity: heatOpacity, borderRadius: 8 }}
              />
            </div>
            <figcaption>Grad-CAM</figcaption>
          </figure>
          <figure className="oct-frame" onClick={() => setZoom(assetUrl(scan.images.boxed))}>
            <img src={assetUrl(scan.images.boxed)} alt={`Bounding box overlay for ${scan.scan_id}`} />
            <figcaption>Bounding box</figcaption>
          </figure>
        </div>
        <label className="field">
          Heatmap opacity
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={heatOpacity}
            onChange={(e) => setHeatOpacity(Number(e.target.value))}
          />
        </label>
      </section>

      {zoom ? (
        <div className="lightbox" role="dialog" aria-label="Enlarged scan image" onClick={() => setZoom(null)}>
          <img src={zoom} alt="Enlarged scan" />
        </div>
      ) : null}

      <div className="grid-2" style={{ marginTop: 16 }}>
        <section className="card">
          <h3>Class probabilities</h3>
          {CLASS_ORDER.map((cls) => {
            const p = scan.per_class_probs?.[cls] ?? 0
            return (
              <div key={cls} className={`prob-row ${cls === label ? 'hit' : ''}`}>
                <span>{cls}</span>
                <div className="prob-track">
                  <span style={{ width: `${p * 100}%` }} />
                </div>
                <span>{formatPct(p)}</span>
              </div>
            )
          })}
          <p style={{ marginTop: 16 }}>{CLASS_DESCRIPTIONS[label]}</p>
        </section>
        <section className="card">
          <h3>
            Uncertainty
            <button
              className="tooltip"
              type="button"
              title="MC Dropout is injected at inference on a checkpoint trained without dropout. Scores and LOW/MODERATE/HIGH thresholds are uncalibrated placeholders."
              aria-label="About uncertainty scores"
            >
              <Info size={16} />
            </button>
          </h3>
          <p>Confidence {formatPct(scan.confidence)}</p>
          <p>Level {scan.uncertainty_level}</p>
          <p>Std {scan.uncertainty_std?.toFixed(4)}</p>
          <p>MC passes {scan.mc_passes}</p>
          <p>{uncertaintyCopy(scan.uncertainty_level, scan.uncertainty_std)}</p>
        </section>
      </div>

      {scan.review_recommended ? (
        <div className="banner bad" role="status">
          <strong>Human review recommended.</strong>
          <ul>
            {scan.review_reasons.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
        </div>
      ) : (
        <div className="banner ok" role="status">
          <strong>Low review risk.</strong> Confidence and uncertainty did not trip the research flags. This is not a clinical clearance.
        </div>
      )}
      <p className="clock">{DISCLAIMER}</p>
    </>
  )
}
