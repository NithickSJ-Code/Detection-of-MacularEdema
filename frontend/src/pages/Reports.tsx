import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { assetUrl, fetchReports } from '../api/client'
import { PageHeader } from '../components/Layout'
import { formatWhen } from '../lib/labels'

function prettySize(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(1)} MB`
}

export function ReportsPage() {
  const q = useQuery({ queryKey: ['reports'], queryFn: fetchReports })
  if (q.isLoading) return <p>Loading reports…</p>
  if (q.isError) return <p>Could not load reports.</p>
  const items = q.data!.items

  return (
    <>
      <PageHeader title="Reports" subtitle="PDFs generated only for scans that passed validation." />
      <section className="card">
        {items.length === 0 ? (
          <div className="empty">
            <h3>No PDF reports yet</h3>
            <p>Analyse a valid OCT scan to generate a one-page report.</p>
          </div>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>Scan ID</th>
                <th>Class</th>
                <th>Generated</th>
                <th>Size</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {items.map((row) => (
                <tr key={row.id}>
                  <td>
                    <Link to={`/results/${row.id}`}>{row.scan_id}</Link>
                  </td>
                  <td>{row.predicted_label ?? '—'}</td>
                  <td>{formatWhen(row.created_at)}</td>
                  <td>{prettySize(row.size_bytes)}</td>
                  <td className="btn-row">
                    <a className="btn secondary" href={assetUrl(row.report_url)} target="_blank" rel="noreferrer">
                      View
                    </a>
                    <a className="btn" href={`${assetUrl(row.report_url)}?download=1`}>
                      Download
                    </a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </>
  )
}
