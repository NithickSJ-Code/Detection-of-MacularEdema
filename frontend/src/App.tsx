import { Navigate, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { AboutPage } from './pages/About'
import { AnalyticsPage } from './pages/Analytics'
import { DashboardPage } from './pages/Dashboard'
import { HistoryPage } from './pages/History'
import { NewScanPage } from './pages/NewScan'
import { ReportsPage } from './pages/Reports'
import { ResultsPage } from './pages/Results'

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/scan" element={<NewScanPage />} />
        <Route path="/results/:id" element={<ResultsPage />} />
        <Route path="/history" element={<HistoryPage />} />
        <Route path="/analytics" element={<AnalyticsPage />} />
        <Route path="/reports" element={<ReportsPage />} />
        <Route path="/about" element={<AboutPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}
