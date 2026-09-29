import type { ReactNode } from 'react'
import type { HealthThresholds } from '../pipeline/rules'
import { STATE_COLOR, stateLabel, useWidth } from './ui'

export const RUL_AXIS_MAX = 130
const frac = (v: number) => Math.min(Math.max(v, 0), RUL_AXIS_MAX) / RUL_AXIS_MAX
const pct = (v: number) => `${frac(v) * 100}%`

export interface ScaleZone {
  from: number
  to: number
  color: string
  label: string
  textColor?: string
}

export const healthZones = (t: HealthThresholds): ScaleZone[] => [
  { from: 0, to: t.action, color: STATE_COLOR.ACTION, label: stateLabel('ACTION') },
  { from: t.action, to: t.plan, color: STATE_COLOR.PLAN, label: stateLabel('PLAN') },
  { from: t.plan, to: t.watch, color: STATE_COLOR.WATCH, label: stateLabel('WATCH') },
  { from: t.watch, to: RUL_AXIS_MAX, color: STATE_COLOR.HEALTHY, label: stateLabel('HEALTHY') },
]

/**
 * Horizontal remaining-useful-life axis (0 = predicted failure, left) with zones.
 * Zone labels render only where they fit — never clipped.
 */
export function RulScale({
  zones,
  children,
  zoneOpacity = 0.22,
  height = 40,
  axisLabel = 'Remaining useful life (cycles)',
  labels = true,
}: {
  zones: ScaleZone[]
  children?: ReactNode
  zoneOpacity?: number
  height?: number
  axisLabel?: string | null
  labels?: boolean
}) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const ticks = [0, 25, 50, 75, 100, 125]
  return (
    <div className={labels || children ? 'pt-8' : 'pt-2'}>
      <div ref={ref} className="relative" style={{ height }}>
        {zones.map((z) => {
          const w = (frac(z.to) - frac(z.from)) * width
          const fits = labels && w >= z.label.length * 6.4 + 14
          return (
            <div
              key={z.label + z.from}
              className="absolute inset-y-0"
              style={{ left: `calc(${pct(z.from)} + 1px)`, width: `calc(${pct(z.to)} - ${pct(z.from)} - 2px)` }}
            >
              <div className="absolute inset-0 rounded-[4px]" style={{ background: z.color, opacity: zoneOpacity }} />
              {fits && (
                <span className="relative block whitespace-nowrap px-2 pt-1 text-[10px] font-semibold" style={{ color: z.textColor ?? 'var(--ink-2)' }}>
                  {z.label}
                </span>
              )}
            </div>
          )
        })}
        {children}
      </div>
      <div className="relative mt-1.5 h-4">
        {ticks.map((t) => (
          <span key={t} className="tabular absolute -translate-x-1/2 text-[10px] text-ink-3" style={{ left: pct(t) }}>{t}</span>
        ))}
      </div>
      {axisLabel && <div className="mt-0.5 text-center text-[10px] text-ink-3">{axisLabel}</div>}
    </div>
  )
}

export function ScaleMarker({ value, label, color = 'var(--ink)', below = false }: { value: number; label: ReactNode; color?: string; below?: boolean }) {
  return (
    <div className="absolute inset-y-0 z-10" style={{ left: pct(value) }}>
      <div className="absolute inset-y-[-5px] w-[2px] -translate-x-1/2 rounded-full" style={{ background: color, boxShadow: '0 0 0 2px var(--surface)' }} />
      <div
        className={`tabular absolute -translate-x-1/2 whitespace-nowrap rounded-md px-1.5 py-0.5 text-[11px] font-semibold ${below ? 'top-full mt-5' : 'bottom-full mb-2'}`}
        style={{ background: color, color: 'var(--surface)' }}
      >
        {label}
      </div>
    </div>
  )
}
