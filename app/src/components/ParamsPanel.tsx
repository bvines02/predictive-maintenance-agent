import type { HealthThresholds } from '../pipeline/rules'
import { RUL_AXIS_MAX } from './RulScale'
import { STATE_COLOR } from './ui'

interface Props {
  thresholds: HealthThresholds
  onThresholds: (t: HealthThresholds) => void
  leadTime: number
  onLeadTime: (v: number) => void
  buffer: number
  defaults: { thresholds: HealthThresholds; leadTime: number }
}

function Slider({
  id,
  label,
  value,
  min,
  max,
  onChange,
  swatch,
  hint,
}: {
  id: string
  label: string
  value: number
  min: number
  max: number
  onChange: (v: number) => void
  swatch?: string
  hint: string
}) {
  return (
    <div>
      <div className="mb-0.5 flex items-center justify-between gap-2">
        <label htmlFor={id} className="flex items-center gap-1.5 text-sm font-medium">
          {swatch && <span className="h-2.5 w-2.5 rounded-sm" style={{ background: swatch }} />}
          {label}
        </label>
        <input
          type="number"
          aria-label={`${label} value`}
          value={value}
          min={min}
          max={max}
          onChange={(e) => {
            const v = Math.round(Number(e.target.value))
            if (Number.isFinite(v)) onChange(Math.min(max, Math.max(min, v)))
          }}
          className="tabular w-16 rounded border border-line bg-surface px-1.5 py-0.5 text-right text-sm"
        />
      </div>
      <input id={id} type="range" min={min} max={max} value={value} onChange={(e) => onChange(Number(e.target.value))} className="w-full" />
      <div className="text-[11px] text-ink-3">{hint}</div>
    </div>
  )
}

export function ParamsPanel({ thresholds: t, onThresholds, leadTime, onLeadTime, buffer, defaults }: Props) {
  // The sliders' ranges enforce action < plan < watch, so an invalid set can't be entered.
  return (
    <div className="space-y-5">
      <div>
        <h3 className="mb-0.5 text-sm font-semibold">Health thresholds</h3>
        <p className="mb-3 text-xs text-ink-2">Predicted RUL at or below each value enters that state.</p>
        <div className="space-y-3">
          <Slider id="t-watch" label="WATCH ≤" swatch={STATE_COLOR.WATCH} value={t.watch} min={t.plan + 1} max={RUL_AXIS_MAX - 5}
            onChange={(v) => onThresholds({ ...t, watch: v })} hint={`repo default ${defaults.thresholds.watch}`} />
          <Slider id="t-plan" label="PLAN ≤" swatch={STATE_COLOR.PLAN} value={t.plan} min={t.action + 1} max={t.watch - 1}
            onChange={(v) => onThresholds({ ...t, plan: v })} hint={`repo default ${defaults.thresholds.plan}`} />
          <Slider id="t-action" label="ACTION ≤" swatch={STATE_COLOR.ACTION} value={t.action} min={0} max={t.plan - 1}
            onChange={(v) => onThresholds({ ...t, action: v })} hint={`repo default ${defaults.thresholds.action}`} />
        </div>
      </div>

      <div>
        <h3 className="mb-0.5 text-sm font-semibold">Maintenance lead time</h3>
        <p className="mb-3 text-xs text-ink-2">Cycles from deciding to act until the work is complete.</p>
        <Slider id="lead" label="Lead time" value={leadTime} min={0} max={60} onChange={onLeadTime}
          hint={`repo default ${defaults.leadTime} · buffer fixed at ${buffer}`} />
      </div>

      <div className="rounded-md bg-surface-2 px-3 py-2 text-[11px] leading-relaxed text-ink-2">
        Criticality and redundancy are excluded. Confidence comes from the model. Lead time is synthetic — NASA C-MAPSS has none.
      </div>
    </div>
  )
}
