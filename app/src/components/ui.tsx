import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from 'react'
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

export const stateLabel = (s: HealthState) => s.charAt(0) + s.slice(1).toLowerCase()

export type StageKind = 'data' | 'ml' | 'rules' | 'llm'
export const KIND: Record<StageKind, { label: string; color: string }> = {
  data: { label: 'Data', color: 'var(--kind-data)' },
  ml: { label: 'Machine learning', color: 'var(--kind-ml)' },
  rules: { label: 'Deterministic policy', color: 'var(--kind-rules)' },
  llm: { label: 'Language model', color: 'var(--kind-llm)' },
}

/** Simple mode tells the story; expert mode adds the supporting detail. */
export const ExpertContext = createContext(false)
export const useExpert = () => useContext(ExpertContext)

/** Content shown only in expert mode, visibly marked as extra. */
export function Expert({ children, bare = false }: { children: ReactNode; bare?: boolean }) {
  if (!useExpert()) return null
  if (bare) return <>{children}</>
  return (
    <div className="mt-8 border-t border-dashed border-line-strong pt-6">
      <div className="mb-4 inline-flex items-center gap-1.5 rounded-full bg-surface-2 px-2 py-0.5 text-[11px] font-semibold text-ink-2">
        <svg width="11" height="11" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden>
          <circle cx="7" cy="7" r="4.5" /><path d="M10.5 10.5L14 14" />
        </svg>
        Expert detail
      </div>
      {children}
    </div>
  )
}

/** Width of an element, tracked with ResizeObserver (0 until mounted). */
export function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null)
  const [width, setWidth] = useState(0)
  useEffect(() => {
    if (!ref.current) return
    const ro = new ResizeObserver(([e]) => setWidth(e.contentRect.width))
    ro.observe(ref.current)
    return () => ro.disconnect()
  }, [])
  return [ref, width] as const
}

// Status is never colour-alone: every state carries an icon and its name.
export function StateIcon({ state, size = 14 }: { state: HealthState; size?: number }) {
  const common = { width: size, height: size, viewBox: '0 0 16 16', fill: 'none', stroke: 'currentColor', strokeWidth: 1.8, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const, 'aria-hidden': true }
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
  const c = STATE_COLOR[state]
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full font-semibold text-ink ${size === 'lg' ? 'px-3 py-1 text-sm' : 'px-2 py-0.5 text-xs'}`}
      style={{ background: `color-mix(in srgb, ${c} 18%, var(--surface))`, boxShadow: `inset 0 0 0 1px color-mix(in srgb, ${c} 45%, transparent)` }}
    >
      <span style={{ color: c }} className="flex"><StateIcon state={state} size={size === 'lg' ? 15 : 13} /></span>
      {stateLabel(state)}
    </span>
  )
}

/** Urgency meter: five ticks filled up to the action's severity. */
function SeverityTicks({ level, on }: { level: number; on: string }) {
  return (
    <span className="flex items-end gap-[2px]" aria-hidden>
      {[1, 2, 3, 4, 5].map((k) => (
        <span key={k} className="w-[3px] rounded-[1px]" style={{ height: 3 + k * 1.6, background: on, opacity: k <= level ? 0.95 : 0.28 }} />
      ))}
    </span>
  )
}

export function ActionBadge({ action, size = 'md' }: { action: Action; size?: 'md' | 'lg' }) {
  const s = severity(action)
  const on = `var(--act-on-${s})`
  return (
    <span
      className={`inline-flex items-center gap-2 rounded-md font-semibold ${size === 'lg' ? 'px-3 py-1.5 text-sm' : 'px-2 py-0.5 text-xs'}`}
      style={{ background: actionColor(action), color: on }}
      title={`Urgency ${s} of 5`}
    >
      <SeverityTicks level={s} on={on} />
      {actionLabel(action)}
    </span>
  )
}

export function ConfidenceBadge({ confidence }: { confidence: Confidence }) {
  const bars = { HIGH: 3, MEDIUM: 2, LOW: 1 }[confidence]
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full bg-surface-2 px-2 py-0.5 text-xs font-semibold text-ink">
      <span className="flex items-end gap-[2px]" aria-hidden>
        {[1, 2, 3].map((i) => (
          <span key={i} className="w-[3px] rounded-[1px]" style={{ height: 3 + i * 2.5, background: i <= bars ? 'var(--ink)' : 'var(--line-strong)' }} />
        ))}
      </span>
      {confidence.charAt(0) + confidence.slice(1).toLowerCase()} confidence
    </span>
  )
}

export function ReviewFlag() {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs font-semibold text-ink" style={{ boxShadow: 'inset 0 0 0 1px var(--ink)' }}>
      <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden>
        <circle cx="8" cy="5" r="2.8" /><path d="M2.5 14.5c.8-3 3-4.5 5.5-4.5s4.7 1.5 5.5 4.5" />
      </svg>
      Human review
    </span>
  )
}

function Chevron({ open }: { open: boolean }) {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"
      className="transition-transform" style={{ transform: open ? 'rotate(90deg)' : 'none' }} aria-hidden>
      <path d="M4.5 2.5L8 6l-3.5 3.5" />
    </svg>
  )
}

/**
 * A pipeline stage: a heading block (step, kind, title, one-sentence takeaway,
 * collapsible method note) above a card holding the stage's visuals.
 */
export function Stage({
  id,
  step,
  title,
  kind,
  takeaway,
  method,
  children,
}: {
  id: string
  step: number
  title: string
  kind: StageKind
  takeaway: ReactNode
  method: ReactNode
  children: ReactNode
}) {
  const [open, setOpen] = useState(false)
  const k = KIND[kind]
  return (
    <section id={id} className="scroll-mt-32" aria-labelledby={`${id}-title`}>
      <div className="mb-4 max-w-3xl">
        <div className="mb-2 flex items-center gap-2 text-xs font-medium text-ink-3">
          <span className="tabular">Step {step}</span>
          <span className="h-3 w-px bg-line-strong" />
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full" style={{ background: k.color }} />
            {k.label}
          </span>
        </div>
        <h2 id={`${id}-title`} className="text-xl font-semibold tracking-tight">{title}</h2>
        <p className="mt-1.5 text-[15px] leading-relaxed text-ink-2">{takeaway}</p>
        <button
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          aria-controls={`${id}-method`}
          className="mt-2 inline-flex items-center gap-1 text-xs font-semibold text-ink-2 hover:text-ink"
        >
          <Chevron open={open} /> {open ? 'Hide method' : 'Method'}
        </button>
        {open && (
          <div id={`${id}-method`} className="prose-method mt-3 border-l-2 border-line pl-4 text-sm leading-relaxed text-ink-2">
            {method}
          </div>
        )}
      </div>
      <div className="card p-5 sm:p-6">{children}</div>
    </section>
  )
}

export function Stat({ label, value, sub, emphasis = false }: { label: string; value: ReactNode; sub?: ReactNode; emphasis?: boolean }) {
  return (
    <div className="min-w-0">
      <div className="text-xs font-medium text-ink-3">{label}</div>
      <div className={`mt-0.5 font-semibold leading-tight tracking-tight ${emphasis ? 'text-3xl' : 'text-xl'}`}>{value}</div>
      {sub && <div className="mt-0.5 text-xs text-ink-3">{sub}</div>}
    </div>
  )
}

export function PanelLabel({ children, right }: { children: ReactNode; right?: ReactNode }) {
  return (
    <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
      <div className="text-xs font-semibold text-ink-2">{children}</div>
      {right && <div className="text-xs text-ink-3">{right}</div>}
    </div>
  )
}

export function Caption({ children }: { children: ReactNode }) {
  return <p className="mt-2 text-xs leading-relaxed text-ink-3">{children}</p>
}

export function Swatch({ color, shape = 'square' }: { color: string; shape?: 'square' | 'line' | 'dash' | 'band' }) {
  if (shape === 'line') return <span className="inline-block h-[2px] w-4 rounded-full" style={{ background: color }} />
  if (shape === 'dash') return <span className="inline-block w-4 border-t-2 border-dashed" style={{ borderColor: color }} />
  if (shape === 'band') return <span className="inline-block h-2.5 w-4 rounded-sm" style={{ background: color }} />
  return <span className="inline-block h-2.5 w-2.5 rounded-[3px]" style={{ background: color }} />
}

export function ActionLegend() {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2">
      {ACTIONS.map((a) => (
        <span key={a} className="inline-flex items-center gap-1.5"><Swatch color={actionColor(a)} />{actionLabel(a)}</span>
      ))}
    </div>
  )
}

export function StateLegend() {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2">
      {(Object.keys(STATE_COLOR) as HealthState[]).map((s) => (
        <span key={s} className="inline-flex items-center gap-1.5">
          <span className="flex" style={{ color: STATE_COLOR[s] }}><StateIcon state={s} size={12} /></span>
          {stateLabel(s)}
        </span>
      ))}
    </div>
  )
}

/* Shared Recharts styling: solid hairline grid, muted axis text, one tooltip look. */
export const axisTick = { fill: 'var(--axis)', fontSize: 11, fontFamily: 'var(--font-sans)' }
export const tooltipStyle = {
  background: 'var(--surface)',
  border: 'none',
  borderRadius: 8,
  boxShadow: '0 0 0 1px var(--ring), 0 8px 24px -6px rgb(0 0 0 / 0.18)',
  fontSize: 12,
  padding: '8px 10px',
}
export const tooltipLabelStyle = { color: 'var(--ink)', fontWeight: 600, marginBottom: 2 }
export const tooltipItemStyle = { color: 'var(--ink-2)', padding: 0 }
