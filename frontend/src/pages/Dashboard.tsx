import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { fetchStats } from '../api/client'
import { PageHeader } from '../components/Layout'
import { formatPct, formatWhen } from '../lib/labels'

export function DashboardPage() {
  const nav = useNavigate()
  const stats = useQuery({ queryKey: ['stats'], queryFn: fetchStats })

  if (stats.isLoading) return <p>Loading dashboard…</p>
  if (stats.isError) return <p>Could not load dashboard statistics.</p>
  const data = stats.data!
  const empty = data.total_scans === 0

  return (
    <>
      <PageHeader title="Dashboard" subtitle="Live counts from this workstation’s scan database." />
      <div className="grid-4">
        <article className="card">
          <div className="stat-label">Total scans</div>
          <div className="stat-value">{data.total_scans}</div>
        </article>
        <article className="card">
          <div className="stat-label">Normal</div>
          <div className="stat-value">{data.normal_count}</div>
        </article>
        <article className="card">
          <div className="stat-label">Review required</div>
          <div className="stat-value">{data.review_count}</div>
        </article>
        <article className="card">
          <div className="stat-label">Rejected uploads</div>
          <div className="stat-value">{data.rejected_uploads}</div>
        </article>
      </div>
      <div className="grid-2" style={{ marginTop: 16 }}>
        <section className="card">
          <h3>Recent scans</h3>
          {empty ? (
            <div className="empty">
              <h3>No scans yet</h3>
              <p>Open New Scan, upload an OCT image or capture from the camera, then return here.</p>
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
                </tr>
              </thead>
              <tbody>
                {data.recent.map((row) => (
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
                    <td>
                      {row.uncertainty_level ? (
                        <span className={`chip ${row.uncertainty_level}`}>{row.uncertainty_level}</span>
                      ) : (
                        '—'
                      )}
                    </td>
                    <td>{formatWhen(row.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>
        <section className="card">
          <h3>Class mix</h3>
          {data.ok_scans === 0 ? (
            <div className="empty">
              <p>Class distribution appears after the first accepted OCT scan.</p>
            </div>
          ) : (
            <div style={{ height: 240 }}>
              <ResponsiveContainer>
                <BarChart data={data.per_class}>
                  <XAxis dataKey="label" />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#0e7c7b" radius={6} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </section>
      </div>
    </>
  )
}
