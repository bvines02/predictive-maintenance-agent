import type { ReactNode } from 'react'
import type { HealthThresholds } from '../pipeline/rules'
import { STATE_COLOR } from './ui'

export const RUL_AXIS_MAX = 130
const pct = (v: number) => `${(Math.min(Math.max(v, 0), RUL_AXIS_MAX) / RUL_AXIS_MAX) * 100}%`

export interface ScaleZone {
  from: number
  to: number
  color: string
  label: string
  textColor?: string
}

export const healthZones = (t: HealthThresholds): ScaleZone[] => [
  { from: 0, to: t.action, color: STATE_COLOR.ACTION, label: 'ACTION' },
  { from: t.action, to: t.plan, color: STATE_COLOR.PLAN, label: 'PLAN' },
  { from: t.plan, to: t.watch, color: STATE_COLOR.WATCH, label: 'WATCH' },
  { from: t.watch, to: RUL_AXIS_MAX, color: STATE_COLOR.HEALTHY, label: 'HEALTHY' },
]

/** A horizontal remaining-useful-life axis (0 = predicted failure) with coloured zones and overlays. */
export function RulScale({
  zones,
  children,
  zoneOpacity = 0.35,
  height = 44,
}: {
  zones: ScaleZone[]
  children?: ReactNode
  zoneOpacity?: number
  height?: number
}) {
  const ticks = [0, 25, 50, 75, 100, 125]
  return (
    <div className="pb-5 pt-7">
      <div className="relative rounded-md bg-surface-2" style={{ height }}>
        {zones.map((z) => (
          <div
            key={z.label + z.from}
            className="absolute inset-y-0 flex items-start overflow-hidden border-r-2 border-surface"
            style={{ left: pct(z.from), width: `calc(${pct(z.to)} - ${pct(z.from)})` }}
          >
            <div className="absolute inset-0" style={{ background: z.color, opacity: zoneOpacity }} />
            <span className="relative truncate px-1.5 pt-0.5 text-[10px] font-semibold tracking-wide" style={{ color: z.textColor ?? 'var(--ink)' }}>
              {z.label}
            </span>
          </div>
        ))}
        {children}
        {ticks.map((t) => (
          <span key={t} className="tabular absolute top-full mt-1 -translate-x-1/2 text-[10px] text-ink-3" style={{ left: pct(t) }}>
            {t}
          </span>
        ))}
      </div>
    </div>
  )
}

export function ScaleMarker({ value, label, color = 'var(--ink)', below = false }: { value: number; label: ReactNode; color?: string; below?: boolean }) {
  return (
    <div className="absolute inset-y-0 z-10" style={{ left: pct(value) }}>
      <div className="absolute inset-y-[-4px] w-[3px] -translate-x-1/2 rounded-full" style={{ background: color }} />
      <div
        className={`tabular absolute -translate-x-1/2 whitespace-nowrap rounded bg-ink px-1.5 py-0.5 text-[11px] font-semibold text-page ${below ? 'top-full mt-4' : 'bottom-full mb-1.5'}`}
      >
        {label}
      </div>
    </div>
  )
}

/** Tree-prediction spread drawn on the same axis: whiskers p0-p100, band p10-p90, box p25-p75. */
export function SpreadOverlay({ q }: { q: Record<'p0' | 'p10' | 'p25' | 'p50' | 'p75' | 'p90' | 'p100', number> }) {
  return (
    <>
      <div className="absolute top-1/2 h-px -translate-y-1/2 bg-ink-2" style={{ left: pct(q.p0), width: `calc(${pct(q.p100)} - ${pct(q.p0)})` }} />
      <div className="absolute top-1/2 h-3 -translate-y-1/2 rounded-sm" style={{ left: pct(q.p10), width: `calc(${pct(q.p90)} - ${pct(q.p10)})`, background: 'var(--series-1)', opacity: 0.35 }} />
      <div className="absolute top-1/2 h-5 -translate-y-1/2 rounded-sm border-2" style={{ left: pct(q.p25), width: `calc(${pct(q.p75)} - ${pct(q.p25)})`, borderColor: 'var(--series-1)' }} />
      {[q.p0, q.p100].map((v, i) => (
        <div key={i} className="absolute top-1/2 h-3 w-px -translate-y-1/2 bg-ink-2" style={{ left: pct(v) }} />
      ))}
    </>
  )
}
