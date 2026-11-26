import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { deleteScan, fetchScans } from '../api/client'
import { PageHeader } from '../components/Layout'
import { CLASS_ORDER, formatPct, formatWhen } from '../lib/labels'

export function HistoryPage() {
  const nav = useNavigate()
  const qc = useQueryClient()
  const [q, setQ] = useState('')
  const [label, setLabel] = useState('')
  const [review, setReview] = useState('')
  const [sort, setSort] = useState('created_at_desc')
  const [offset, setOffset] = useState(0)
  const limit = 10
  const list = useQuery({
    queryKey: ['scans', q, label, review, sort, offset],
    queryFn: () =>
      fetchScans({
        q: q || undefined,
        label: label || undefined,
        review: review || undefined,
        sort,
        limit,
        offset,
      }),
  })
  const del = useMutation({
    mutationFn: deleteScan,
    onSuccess: () => qc.invalidateQueries({ queryKey: ['scans'] }),
  })

  const rows = list.data?.items ?? []
  const total = list.data?.total ?? 0

  return (
    <>
      <PageHeader title="Scan History" subtitle="Search, filter, and open any stored analysis." />
      <div className="filters">
        <input placeholder="Search Scan ID" value={q} onChange={(e) => { setQ(e.target.value); setOffset(0) }} />
        <select value={label} onChange={(e) => { setLabel(e.target.value); setOffset(0) }}>
          <option value="">All classes</option>
          {CLASS_ORDER.map((c) => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
        <select value={review} onChange={(e) => { setReview(e.target.value); setOffset(0) }}>
          <option value="">All review states</option>
          <option value="true">Needs review</option>
          <option value="false">No review flag</option>
        </select>
        <select value={sort} onChange={(e) => setSort(e.target.value)}>
          <option value="created_at_desc">Newest</option>
          <option value="created_at_asc">Oldest</option>
          <option value="confidence_desc">Confidence high</option>
          <option value="confidence_asc">Confidence low</option>
          <option value="scan_id_asc">Scan ID A–Z</option>
        </select>
      </div>
      <section className="card">
        {list.isLoading ? (
          <p>Loading history…</p>
        ) : total === 0 ? (
          <div className="empty">
            <h3>No matching scans</h3>
            <p>Run a new scan, or clear the filters.</p>
          </div>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>Scan ID</th>
                <th>Result</th>
                <th>Confidence</th>
                <th>Uncertainty</th>
                <th>Date</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.id} onClick={() => nav(`/results/${row.id}`)}>
                  <td>{row.scan_id}</td>
                  <td>
                    {row.status === 'INVALID_INPUT' ? (
                      <span className="chip HIGH">Rejected</span>
                    ) : (
                      <span className={`chip ${row.predicted_label ?? ''}`}>{row.predicted_label}</span>
                    )}
                  </td>
                  <td>{formatPct(row.confidence)}</td>
                  <td>{row.uncertainty_level ?? '—'}</td>
                  <td>{formatWhen(row.created_at)}</td>
                  <td>
                    <button
                      className="btn danger"
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation()
                        if (window.confirm(`Delete ${row.scan_id} and its files?`)) del.mutate(row.id)
                      }}
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <div className="pager">
          <span>
            {total} records · page {Math.floor(offset / limit) + 1}
          </span>
          <div className="btn-row">
            <button className="btn secondary" type="button" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - limit))}>
              Previous
            </button>
            <button className="btn secondary" type="button" disabled={offset + limit >= total} onClick={() => setOffset(offset + limit)}>
              Next
            </button>
          </div>
        </div>
      </section>
    </>
  )
}
