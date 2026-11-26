import { useQuery } from '@tanstack/react-query'
import {
  BarChart3,
  FileText,
  History,
  Info,
  LayoutDashboard,
  Menu,
  Moon,
  ScanLine,
  Sun,
  User,
} from 'lucide-react'
import { useEffect, useState, type ReactNode } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import { fetchHealth } from '../api/client'
import { DISCLAIMER } from '../lib/labels'

const LINKS = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/scan', label: 'New Scan', icon: ScanLine },
  { to: '/history', label: 'Scan History', icon: History },
  { to: '/analytics', label: 'Analytics', icon: BarChart3 },
  { to: '/reports', label: 'Reports', icon: FileText },
  { to: '/about', label: 'About System', icon: Info },
]

export function Layout() {
  const [open, setOpen] = useState(false)
  const [now, setNow] = useState(() => new Date())
  const [theme, setTheme] = useState(() => localStorage.getItem('oct-theme') || 'light')
  const health = useQuery({
    queryKey: ['health'],
    queryFn: fetchHealth,
    refetchInterval: 15000,
  })

  useEffect(() => {
    const id = window.setInterval(() => setNow(new Date()), 1000)
    return () => window.clearInterval(id)
  }, [])

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    localStorage.setItem('oct-theme', theme)
  }, [theme])

  const online = Boolean(health.data?.model_loaded)
  const badge = health.isError ? 'offline' : health.isLoading ? 'checking' : online ? 'online' : 'offline'

  return (
    <div className="app">
      <aside className={`sidebar ${open ? 'open' : ''}`}>
        <div className="brand-kicker">Aravind research collaboration</div>
        <div className="brand-title">Macula OCT Lab</div>
        <nav className="nav" aria-label="Primary">
          {LINKS.map((link) => (
            <NavLink key={link.to} to={link.to} end={link.to === '/'} onClick={() => setOpen(false)}>
              <link.icon size={18} aria-hidden />
              {link.label}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-foot">Team 05 · IV CSE-A</div>
      </aside>
      <div className="main">
        <header className="topbar">
          <button className="icon-btn menu-btn" type="button" aria-label="Open menu" onClick={() => setOpen((v) => !v)}>
            <Menu size={18} />
          </button>
          <h1>Early Detection of Macular Edema in Retinal OCT Scans</h1>
          <span className={`badge ${badge}`}>
            <span className="dot" aria-hidden />
            {badge === 'online' ? 'AI Model Online' : badge === 'checking' ? 'Checking model…' : 'AI Model Offline'}
          </span>
          <span className="clock">{now.toLocaleString()}</span>
          <div className="profile">
            <span className="avatar" aria-hidden>
              <User size={16} />
            </span>
            Research User
          </div>
          <button
            className="icon-btn"
            type="button"
            aria-label="Toggle colour theme"
            onClick={() => setTheme((t) => (t === 'light' ? 'dark' : 'light'))}
          >
            {theme === 'light' ? <Moon size={16} /> : <Sun size={16} />}
          </button>
        </header>
        <main className="content">
          <Outlet context={{ health: health.data, online, healthError: health.isError } satisfies ShellContext} />
        </main>
        <footer className="footer-disclaimer">{DISCLAIMER}</footer>
      </div>
    </div>
  )
}

export interface ShellContext {
  health: Awaited<ReturnType<typeof fetchHealth>> | undefined
  online: boolean
  healthError: boolean
}

export function PageHeader({ title, subtitle, actions }: { title: string; subtitle?: string; actions?: ReactNode }) {
  return (
    <div className="page-head">
      <div>
        <h2>{title}</h2>
        {subtitle ? <p>{subtitle}</p> : null}
      </div>
      {actions}
    </div>
  )
}
