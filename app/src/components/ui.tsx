import type { ReactNode } from 'react'
import { ACTIONS, severity, type Action, type Confidence, type HealthState } from '../pipeline/rules'

export const fmt = (n: number, digits = 1) =>
  Number.isFinite(n) ? n.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits }) : '—'

export const STATE_COLOR: Record<HealthState, string> = {
  HEALTHY: 'var(--good)',
  WATCH: 'var(--warning)',
  PLAN: 'var(--serious)',
  ACTION: 'var(--critical)',
}

export const actionColor = (a: Action) => `var(--act-${severity(a)})`

export const actionLabel = (a: Action) =>
  a.toLowerCase().replace(/_/g, ' ').replace(/^\w/, (c) => c.toUpperCase())

// Status is never colour-alone: every state carries an icon and its name.
function StateIcon({ state }: { state: HealthState }) {
  const common = { width: 14, height: 14, viewBox: '0 0 16 16', fill: 'none', stroke: 'currentColor', strokeWidth: 1.8, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const }
  switch (state) {
    case 'HEALTHY':
      return <svg {...common}><path d="M3.5 8.5l3 3 6-7" /></svg>
    case 'WATCH':
      return <svg {...common}><path d="M1.5 8s2.5-4.5 6.5-4.5S14.5 8 14.5 8 12 12.5 8 12.5 1.5 8 1.5 8z" /><circle cx="8" cy="8" r="1.8" /></svg>
    case 'PLAN':
      return <svg {...common}><rect x="2" y="3" width="12" height="11" rx="1.5" /><path d="M2 6.5h12M5.5 1.5v3M10.5 1.5v3" /></svg>
    case 'ACTION':
      return <svg {...common}><path d="M8 1.8L15 14H1L8 1.8z" /><path d="M8 6.5v3.2M8 11.8v.2" /></svg>
  }
}

export function HealthBadge({ state, size = 'md' }: { state: HealthState; size?: 'md' | 'lg' }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full font-semibold text-[#0b0b0b] ${size === 'lg' ? 'px-3 py-1 text-base' : 'px-2 py-0.5 text-xs'}`}
      style={{ background: STATE_COLOR[state] }}
    >
      <StateIcon state={state} />
      {state}
    </span>
  )
}

export function ActionBadge({ action, size = 'md' }: { action: Action; size?: 'md' | 'lg' }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-md font-semibold ${size === 'lg' ? 'px-3 py-1 text-base' : 'px-2 py-0.5 text-xs'}`}
      style={{ background: actionColor(action), color: `var(--act-on-${severity(action)})` }}
    >
      <span className="tabular opacity-70">{severity(action)}</span>
      {actionLabel(action)}
    </span>
  )
}

export function ConfidenceBadge({ confidence }: { confidence: Confidence }) {
  const bars = { HIGH: 3, MEDIUM: 2, LOW: 1 }[confidence]
  return (
    <span className="inline-flex items-center gap-1.5 rounded-md border border-line px-2 py-0.5 text-xs font-semibold text-ink">
      <span className="flex items-end gap-[2px]" aria-hidden>
        {[1, 2, 3].map((i) => (
          <span key={i} className="w-[3px] rounded-sm" style={{ height: 4 + i * 3, background: i <= bars ? 'var(--ink)' : 'var(--line)' }} />
        ))}
      </span>
      {confidence} confidence
    </span>
  )
}

export function Stage({
  id,
  step,
  title,
  kind,
  summary,
  children,
}: {
  id: string
  step: number
  title: string
  kind: 'data' | 'ml' | 'rules' | 'llm'
  summary: ReactNode
  children: ReactNode
}) {
  const kindLabel = { data: 'Data', ml: 'Machine learning', rules: 'Deterministic rules', llm: 'LLM reasoning' }[kind]
  return (
    <section id={id} className="scroll-mt-40 rounded-xl border border-line bg-surface">
      <header className="flex flex-wrap items-baseline gap-x-3 gap-y-1 border-b border-line px-5 py-3">
        <span className="tabular flex h-6 w-6 shrink-0 items-center justify-center self-center rounded-full bg-ink text-xs font-bold text-page">
          {step}
        </span>
        <h2 className="text-base font-semibold">{title}</h2>
        <span className={`text-xs font-medium uppercase tracking-wide ${kind === 'llm' ? 'text-accent' : 'text-ink-3'}`}>{kindLabel}</span>
      </header>
      <div className="px-5 py-4">
        <p className="mb-4 max-w-3xl text-sm leading-relaxed text-ink-2">{summary}</p>
        {children}
      </div>
    </section>
  )
}

export function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="min-w-0">
      <div className="text-xs text-ink-3">{label}</div>
      <div className="tabular text-xl font-semibold leading-tight">{value}</div>
      {sub && <div className="text-xs text-ink-3">{sub}</div>}
    </div>
  )
}

export function ActionLegend() {
  return (
    <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-ink-2">
      {ACTIONS.map((a) => (
        <span key={a} className="inline-flex items-center gap-1">
          <span className="h-2.5 w-2.5 rounded-sm" style={{ background: actionColor(a) }} />
          {actionLabel(a)}
        </span>
      ))}
    </div>
  )
}

export function StateLegend() {
  return (
    <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-ink-2">
      {(Object.keys(STATE_COLOR) as HealthState[]).map((s) => (
        <span key={s} className="inline-flex items-center gap-1">
          <span className="h-2.5 w-2.5 rounded-sm" style={{ background: STATE_COLOR[s] }} />
          {s}
        </span>
      ))}
    </div>
  )
}
