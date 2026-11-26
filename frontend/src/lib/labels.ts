import type { ClassLabel, UncertaintyLevel } from '../api/types'

export const DISCLAIMER =
  'Research prototype. Not a medical device. AI output must not be used as a standalone clinical diagnosis.'

export const CLASS_ORDER: ClassLabel[] = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

export const CLASS_DESCRIPTIONS: Record<ClassLabel, string> = {
  CNV: 'Choroidal Neovascularisation (CNV) — Abnormal blood vessels growing beneath the retina. Associated with wet Age-related Macular Degeneration.',
  DME: 'Diabetic Macular Edema (DME) — Fluid accumulation in the macula caused by diabetic damage to retinal blood vessels. A leading cause of vision loss in working-age adults.',
  DRUSEN:
    'Drusen — Small yellow deposits under the retina. A hallmark of early dry Age-related Macular Degeneration. Monitoring is advised.',
  NORMAL:
    'Normal Macula — No pathological fluid or drusen deposits detected in this scan. Regular follow-up recommended as per clinical protocol.',
}

export function formatPct(value: number | null | undefined, digits = 1): string {
  if (value == null || Number.isNaN(value)) return '—'
  return `${(value * 100).toFixed(digits)}%`
}

export function formatWhen(iso: string): string {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleString()
}

export function uncertaintyCopy(level: UncertaintyLevel | null, std: number | null): string {
  if (!level) return 'Uncertainty was not computed for this scan.'
  const stdText = std != null ? ` (std ${std.toFixed(3)})` : ''
  if (level === 'LOW') {
    return `The model’s repeated stochastic passes agreed closely${stdText}. This is an uncalibrated signal, not a clinical certainty score.`
  }
  if (level === 'MODERATE') {
    return `Repeated passes showed some spread${stdText}. Treat the label as a research indication and review the scan.`
  }
  return `Repeated passes disagreed substantially${stdText}. Human review is recommended before any interpretation.`
}
