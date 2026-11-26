import { useMutation } from '@tanstack/react-query'
import { Camera, ImageUp } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { useNavigate, useOutletContext } from 'react-router-dom'
import { analyzeScan } from '../api/client'
import type { ShellContext } from '../components/Layout'
import { PageHeader } from '../components/Layout'

const STEPS = [
  'Image validation',
  'DenseNet classification',
  'Grad-CAM localisation',
  'Uncertainty estimation',
  'Generating results',
]

export function NewScanPage() {
  const { online } = useOutletContext<ShellContext>()
  const nav = useNavigate()
  const [mode, setMode] = useState<'upload' | 'camera'>('upload')
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [scanId, setScanId] = useState('')
  const [drag, setDrag] = useState(false)
  const [progress, setProgress] = useState(0)
  const [step, setStep] = useState(0)
  const [camError, setCamError] = useState<string | null>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const mutation = useMutation({
    mutationFn: async (payload: { blob: Blob; name: string; source: 'upload' | 'webcam' }) => {
      abortRef.current?.abort()
      const ctl = new AbortController()
      abortRef.current = ctl
      return analyzeScan(payload.blob, payload.name, scanId, payload.source, ctl.signal)
    },
    onSuccess: (rec) => {
      setProgress(100)
      nav(`/results/${rec.id}`)
    },
    onSettled: () => {
      abortRef.current = null
    },
  })

  useEffect(() => {
    if (!file) {
      setPreview(null)
      return
    }
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  useEffect(() => {
    if (!mutation.isPending) {
      setProgress(0)
      setStep(0)
      return
    }
    const started = Date.now()
    const id = window.setInterval(() => {
      const elapsed = Date.now() - started
      const pct = Math.min(95, 8 + elapsed / 80)
      setProgress(pct)
      setStep(Math.min(STEPS.length - 1, Math.floor(pct / 20)))
    }, 200)
    return () => window.clearInterval(id)
  }, [mutation.isPending])

  useEffect(() => {
    return () => abortRef.current?.abort()
  }, [])

  useEffect(() => {
    if (mode !== 'camera') {
      streamRef.current?.getTracks().forEach((t) => t.stop())
      streamRef.current = null
      return
    }
    let cancelled = false
    setCamError(null)
    navigator.mediaDevices
      .getUserMedia({ video: { facingMode: 'environment' }, audio: false })
      .catch(() => navigator.mediaDevices.getUserMedia({ video: true, audio: false }))
      .then((stream) => {
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop())
          return
        }
        streamRef.current = stream
        if (videoRef.current) videoRef.current.srcObject = stream
      })
      .catch((err: unknown) => {
        const name = err instanceof DOMException ? err.name : ''
        if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
          setCamError('Camera permission was blocked. Allow camera access in the browser, or switch to file upload.')
        } else if (name === 'NotFoundError') {
          setCamError('No camera was found on this device. Use file upload instead.')
        } else {
          setCamError('Could not start the camera. Use file upload instead.')
        }
      })
    return () => {
      cancelled = true
      streamRef.current?.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
  }, [mode])

  function onFiles(list: FileList | null) {
    const next = list?.[0]
    if (next) setFile(next)
  }

  function capture() {
    const video = videoRef.current
    if (!video) return
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth || 640
    canvas.height = video.videoHeight || 480
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.drawImage(video, 0, 0)
    canvas.toBlob((blob) => {
      if (!blob) return
      mutation.mutate({ blob, name: 'webcam.png', source: 'webcam' })
    }, 'image/png')
  }

  function analyzeUpload() {
    if (!file) return
    mutation.mutate({ blob: file, name: file.name, source: 'upload' })
  }

  return (
    <>
      <PageHeader
        title="New Scan"
        subtitle="Upload an OCT image or capture from the workstation camera. The model runs on this machine."
      />
      {!online ? (
        <div className="banner bad" role="alert">
          The AI model is offline. Analysis is disabled until the DenseNet checkpoint loads.
        </div>
      ) : null}

      <div className="tabs" role="tablist">
        <button type="button" className={mode === 'upload' ? 'active' : ''} onClick={() => setMode('upload')}>
          <ImageUp size={16} aria-hidden /> Upload
        </button>
        <button type="button" className={mode === 'camera' ? 'active' : ''} onClick={() => setMode('camera')}>
          <Camera size={16} aria-hidden /> Webcam
        </button>
      </div>

      {mutation.isPending ? (
        <section className="card">
          <h3>Analysing scan</h3>
          <p>One server call runs the full pipeline. These steps are a timed indicator, not live stage telemetry.</p>
          <div className="bar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(progress)}>
            <span style={{ width: `${progress}%` }} />
          </div>
          <div className="progress-steps">
            {STEPS.map((label, i) => (
              <div key={label} className={`step ${i <= step ? 'on' : ''}`}>
                {i < step ? '✓' : i === step ? '●' : '○'} {label}
              </div>
            ))}
          </div>
        </section>
      ) : (
        <section className="card">
          <label className="field">
            Scan ID (optional)
            <input
              value={scanId}
              onChange={(e) => setScanId(e.target.value)}
              placeholder="Leave blank to auto-assign SCAN-YYYYMMDD-NNN"
            />
          </label>

          {mode === 'upload' ? (
            <>
              <div
                className={`dropzone ${drag ? 'drag' : ''}`}
                onDragOver={(e) => {
                  e.preventDefault()
                  setDrag(true)
                }}
                onDragLeave={() => setDrag(false)}
                onDrop={(e) => {
                  e.preventDefault()
                  setDrag(false)
                  onFiles(e.dataTransfer.files)
                }}
              >
                <p>Drag and drop a PNG, JPEG, or BMP OCT scan, or browse.</p>
                <input
                  type="file"
                  accept=".png,.jpg,.jpeg,.bmp,image/png,image/jpeg,image/bmp"
                  onChange={(e) => onFiles(e.target.files)}
                />
              </div>
              {preview ? <img className="preview-img" src={preview} alt="Selected OCT scan preview" /> : null}
              {mutation.isError ? <div className="banner bad">{(mutation.error as Error).message}</div> : null}
              <button className="btn" type="button" disabled={!file || !online} onClick={analyzeUpload}>
                Analyze Scan
              </button>
            </>
          ) : (
            <>
              {camError ? (
                <div className="banner bad">{camError}</div>
              ) : (
                <div className="camera-wrap">
                  <video ref={videoRef} autoPlay playsInline muted aria-label="Live camera preview" />
                  <div className="guide" aria-hidden />
                </div>
              )}
              {mutation.isError ? <div className="banner bad">{(mutation.error as Error).message}</div> : null}
              <div className="btn-row" style={{ marginTop: 12 }}>
                <button className="btn" type="button" disabled={Boolean(camError) || !online} onClick={capture}>
                  Capture & Analyze
                </button>
              </div>
            </>
          )}
        </section>
      )}
    </>
  )
}
