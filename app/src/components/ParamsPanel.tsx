import type { HealthThresholds } from '../pipeline/rules'
import { RUL_AXIS_MAX, RulScale, healthZones } from './RulScale'
import { STATE_COLOR, StateIcon } from './ui'

interface Props {
  idPrefix: string
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
  icon,
  value,
  min,
  max,
  onChange,
  tint = 'var(--ink)',
  isDefault,
}: {
  id: string
  label: string
  icon?: React.ReactNode
  value: number
  min: number
  max: number
  onChange: (v: number) => void
  tint?: string
  isDefault: boolean
}) {
  const fill = max > min ? ((value - min) / (max - min)) * 100 : 0
  return (
    <div>
      <div className="flex items-center justify-between gap-2">
        <label htmlFor={id} className="flex items-center gap-1.5 text-[13px] font-medium">
          {icon}
          {label}
          {!isDefault && <span className="h-1.5 w-1.5 rounded-full bg-series-1" title="Changed from default" />}
        </label>
        <div className="flex items-baseline gap-1">
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
            className="num tabular w-11 rounded-md bg-surface-2 px-1.5 py-0.5 text-right text-[13px] font-semibold focus:bg-surface"
          />
          <span className="text-[11px] text-ink-3">cyc</span>
        </div>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="range mt-1 w-full"
        style={{ '--fill': `${fill}%`, '--tint': tint } as React.CSSProperties}
      />
    </div>
  )
}

export function ParamsPanel({ idPrefix, thresholds: t, onThresholds, leadTime, onLeadTime, buffer, defaults }: Props) {
  const d = defaults.thresholds
  const changed = t.watch !== d.watch || t.plan !== d.plan || t.action !== d.action || leadTime !== defaults.leadTime
  const icon = (s: 'WATCH' | 'PLAN' | 'ACTION') => <span className="flex" style={{ color: STATE_COLOR[s] }}><StateIcon state={s} size={13} /></span>

  // The slider ranges enforce action < plan < watch, so an invalid set can't be entered.
  return (
    <div>
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">Decision policy</h3>
        <button
          onClick={() => {
            onThresholds(d)
            onLeadTime(defaults.leadTime)
          }}
          disabled={!changed}
          className="rounded-md px-2 py-0.5 text-xs font-medium text-ink-2 hover:bg-surface-2 hover:text-ink disabled:pointer-events-none disabled:opacity-40"
        >
          Reset
        </button>
      </div>
      <p className="mt-0.5 text-xs leading-relaxed text-ink-3">Organisational choices, not model outputs. Every decision updates live.</p>

      <div className="-mx-1 mt-1">
        <RulScale zones={healthZones(t)} height={8} zoneOpacity={0.7} axisLabel={null} labels={false} />
      </div>

      <div className="mt-1 space-y-3">
        <div className="text-xs font-medium text-ink-3">Health thresholds · RUL at or below</div>
        <Slider id={`${idPrefix}-watch`} label="Watch" icon={icon('WATCH')} tint={STATE_COLOR.WATCH} value={t.watch} min={t.plan + 1} max={RUL_AXIS_MAX - 5}
          onChange={(v) => onThresholds({ ...t, watch: v })} isDefault={t.watch === d.watch} />
        <Slider id={`${idPrefix}-plan`} label="Plan" icon={icon('PLAN')} tint={STATE_COLOR.PLAN} value={t.plan} min={t.action + 1} max={t.watch - 1}
          onChange={(v) => onThresholds({ ...t, plan: v })} isDefault={t.plan === d.plan} />
        <Slider id={`${idPrefix}-action`} label="Action" icon={icon('ACTION')} tint={STATE_COLOR.ACTION} value={t.action} min={0} max={t.plan - 1}
          onChange={(v) => onThresholds({ ...t, action: v })} isDefault={t.action === d.action} />
      </div>

      <div className="mt-5 space-y-3 border-t border-line pt-4">
        <div className="text-xs font-medium text-ink-3">Execution</div>
        <Slider id={`${idPrefix}-lead`} label="Maintenance lead time" value={leadTime} min={0} max={60} onChange={onLeadTime}
          isDefault={leadTime === defaults.leadTime} />
        <p className="text-[11px] leading-relaxed text-ink-3">
          Cycles from committing to work until it is complete, plus a fixed {buffer}-cycle planning buffer. Synthetic: C-MAPSS records no lead time.
        </p>
      </div>
    </div>
  )
}
