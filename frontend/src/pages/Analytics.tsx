import { useQuery } from '@tanstack/react-query'
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { fetchStats } from '../api/client'
import { PageHeader } from '../components/Layout'
import { formatPct } from '../lib/labels'

export function AnalyticsPage() {
  const stats = useQuery({ queryKey: ['stats'], queryFn: fetchStats })
  if (stats.isLoading) return <p>Loading analytics…</p>
  if (stats.isError) return <p>Could not load analytics.</p>
  const data = stats.data!
  const empty = data.ok_scans === 0

  return (
    <>
      <PageHeader title="Analytics" subtitle="All figures are computed from SQLite on this machine. Nothing is simulated." />
      <div className="grid-4">
        <article className="card">
          <div className="stat-label">Accepted scans</div>
          <div className="stat-value">{data.ok_scans}</div>
        </article>
        <article className="card">
          <div className="stat-label">Mean confidence</div>
          <div className="stat-value">{formatPct(data.mean_confidence)}</div>
        </article>
        <article className="card">
          <div className="stat-label">Review rate</div>
          <div className="stat-value">{formatPct(data.review_rate)}</div>
        </article>
        <article className="card">
          <div className="stat-label">Rejected uploads</div>
          <div className="stat-value">{data.rejected_uploads}</div>
        </article>
      </div>
      {empty ? (
        <div className="card empty" style={{ marginTop: 16 }}>
          <h3>No accepted analyses yet</h3>
          <p>Charts populate after at least one OCT scan passes validation.</p>
        </div>
      ) : (
        <div className="grid-2" style={{ marginTop: 16 }}>
          <section className="card">
            <h3>Class distribution</h3>
            <div style={{ height: 260 }}>
              <ResponsiveContainer>
                <BarChart data={data.per_class}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="label" />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#0f2c4c" radius={6} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>
          <section className="card">
            <h3>Uncertainty levels</h3>
            <div style={{ height: 260 }}>
              <ResponsiveContainer>
                <BarChart
                  data={Object.entries(data.uncertainty_levels).map(([level, count]) => ({ level, count }))}
                >
                  <XAxis dataKey="level" />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#0e7c7b" radius={6} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>
          <section className="card" style={{ gridColumn: '1 / -1' }}>
            <h3>Scans per day (14 days)</h3>
            <div style={{ height: 260 }}>
              <ResponsiveContainer>
                <BarChart data={data.scans_per_day}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="count" name="Scans" fill="#14b8a6" radius={6} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>
        </div>
      )}
    </>
  )
}
