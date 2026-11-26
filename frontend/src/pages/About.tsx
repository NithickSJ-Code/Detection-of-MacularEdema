import { useOutletContext } from 'react-router-dom'
import type { ShellContext } from '../components/Layout'
import { PageHeader } from '../components/Layout'
import { DISCLAIMER } from '../lib/labels'

const STEPS = [
  'Image',
  'Validation',
  'DenseNet-121',
  'Grad-CAM',
  'MC Dropout',
  'Uncertainty',
  'Final result',
]

export function AboutPage() {
  const { health } = useOutletContext<ShellContext>()
  const ood = health?.ood_stage2_active

  return (
    <>
      <PageHeader title="About this system" subtitle="Research prototype wrapping a trained DenseNet-121 OCT classifier." />
      <section className="card">
        <h3>Pipeline</h3>
        <div className="flow" aria-label="Analysis pipeline">
          {STEPS.map((s, i) => (
            <span key={s} style={{ display: 'contents' }}>
              <span className="node">{s}</span>
              {i < STEPS.length - 1 ? <span className="arrow">→</span> : null}
            </span>
          ))}
        </div>
      </section>
      <div className="grid-2" style={{ marginTop: 16 }}>
        <section className="card">
          <h3>Model facts</h3>
          <ul>
            <li>Architecture: DenseNet-121, 4-class linear head</li>
            <li>Classes: CNV, DME, DRUSEN, NORMAL</li>
            <li>Training data: Kermany 2018 retinal OCT set</li>
            <li>Input size: 224 × 224, ImageNet normalisation</li>
            <li>Device: {health?.device ?? 'unknown'}</li>
            <li>MC passes: {health?.mc_passes ?? 20}</li>
            <li>Checkpoint: {health?.model_name ?? 'DenseNet-121 (Hardened — Pathway A)'}</li>
          </ul>
        </section>
        <section className="card">
          <h3>Team & guide</h3>
          <ul>
            <li>Nishant J S — AI architecture & MC Dropout</li>
            <li>Nishant J R — Results, evaluation & testing</li>
            <li>Nithin S B — System integration & documentation</li>
            <li>Nithick S J — Data pipeline & flow</li>
            <li>Guide: Mrs. Deepthi Nair, AP/CSE</li>
            <li>IV CSE-A, Team 05 · Aravind Eye Hospital research collaboration</li>
          </ul>
        </section>
      </div>
      <section className="card" style={{ marginTop: 16 }}>
        <h3>Known limitations</h3>
        <ul>
          <li>
            Monte Carlo Dropout is an approximation. The checkpoint was trained without dropout; p=0.3 is injected at
            inference. Uncertainty values and LOW/MODERATE/HIGH thresholds are uncalibrated placeholders.
          </li>
          <li>
            OOD Stage 2 (Mahalanobis feature-space gate) is {ood ? 'active on this machine.' : 'inactive because models/ood_stats.npz is not present. Only the colour/saturation gate runs.'}
          </li>
          <li>
            Published “100% test accuracy” on Kermany 2018 is not shown here as clinical performance. That split is known
            to contain redistributed duplicates.
          </li>
          <li>{DISCLAIMER}</li>
        </ul>
      </section>
    </>
  )
}
