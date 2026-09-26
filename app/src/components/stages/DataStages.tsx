import { useState } from 'react'
import { CartesianGrid, Line, LineChart, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { EngineData, FixtureIndex } from '../../pipeline/data'
import { Stage, fmt } from '../ui'

interface Props {
  index: FixtureIndex
  engine: EngineData
  i: number
  setI: (i: number) => void
}

const TRANSFORM_LABEL: Record<string, string> = {
  raw: 'current value',
  roll_mean_5: 'rolling mean, last 5',
  roll_mean_10: 'rolling mean, last 10',
  roll_std_5: 'rolling std, last 5',
  roll_std_10: 'rolling std, last 10',
  delta_1: 'change since last cycle',
  trend_5: 'trend slope, last 5',
}

export function featureLabel(transform: string) {
  return TRANSFORM_LABEL[transform] ?? transform.replace(/_/g, ' ')
}

export function TelemetryStage({ index, engine, i, setI }: Props) {
  const sensorsByColumn = Object.fromEntries(index.sensors.map((s) => [s.column, s]))
  const [sensor, setSensor] = useState(index.model.top_features.find((f) => f.sensor)?.sensor ?? index.kept_sensors[0])
  const meta = sensorsByColumn[sensor]
  const values = engine.sensors[sensor]
  const data = engine.cycle.map((c, k) => ({ cycle: c, value: values[k] }))
  const cycle = engine.cycle[i]
  const window = Math.max(...index.model.rolling_windows)
  const dropped = index.sensors.filter((s) => s.status !== 'keep')

  return (
    <Stage
      id="stage-telemetry"
      step={1}
      kind="data"
      title="Telemetry in"
      summary={
        <>
          Each flight cycle, the engine reports 21 sensors and 3 operating settings. Step 4's screening dropped{' '}
          {dropped.length} sensors that never meaningfully change in FD001, so {index.kept_sensors.length} flow on. The
          model sees only cycles up to and including the current one — never the future.
        </>
      }
    >
      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_280px]">
        <div className="min-w-0">
          <div className="mb-2 flex flex-wrap items-center gap-2 text-sm">
            <label htmlFor="sensor-pick" className="text-ink-2">Sensor</label>
            <select
              id="sensor-pick"
              value={sensor}
              onChange={(e) => setSensor(e.target.value)}
              className="rounded-md border border-line bg-surface px-2 py-1 text-sm"
            >
              {index.kept_sensors.map((s) => (
                <option key={s} value={s}>
                  {sensorsByColumn[s].symbol} — {sensorsByColumn[s].name}
                </option>
              ))}
            </select>
            <span className="text-xs text-ink-3">
              {meta.unit} · near failure it <strong className="font-semibold text-ink-2">{meta.expected_trend}</strong>
            </span>
          </div>
          <div className="h-56">
            <ResponsiveContainer>
              <LineChart
                data={data}
                margin={{ top: 8, right: 12, bottom: 0, left: 0 }}
                onClick={(e) => typeof e?.activeTooltipIndex === 'number' && setI(e.activeTooltipIndex)}
              >
                <CartesianGrid stroke="var(--line)" vertical={false} />
                <XAxis dataKey="cycle" type="number" domain={['dataMin', 'dataMax']} tick={{ fill: 'var(--ink-3)', fontSize: 11 }} stroke="var(--line)" />
                <YAxis domain={['auto', 'auto']} tick={{ fill: 'var(--ink-3)', fontSize: 11 }} stroke="var(--line)" width={56} tickFormatter={(v) => fmt(v, 1)} />
                <ReferenceArea x1={Math.max(engine.cycle[0], cycle - window + 1)} x2={cycle} fill="var(--series-1)" fillOpacity={0.12} ifOverflow="visible" />
                <ReferenceArea x1={cycle} x2={engine.cycle[engine.cycle.length - 1]} fill="var(--surface-2)" fillOpacity={0.7} />
                <ReferenceLine x={cycle} stroke="var(--ink)" strokeWidth={1.5} />
                <Tooltip
                  contentStyle={{ background: 'var(--surface)', border: '1px solid var(--line)', borderRadius: 8, fontSize: 12 }}
                  labelFormatter={(c) => `Cycle ${c}`}
                  formatter={(v) => [fmt(Number(v), 2), meta.symbol]}
                />
                <Line dataKey="value" stroke="var(--series-1)" strokeWidth={1.5} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-1 text-xs text-ink-3">
            Shaded blue: the last {window} cycles the rolling features read. Greyed: the future — not available to the model at cycle {cycle}.
          </p>
        </div>

        <div className="min-w-0">
          <div className="mb-2 text-xs font-medium text-ink-2">Readings at cycle {cycle}</div>
          <div className="grid grid-cols-2 gap-x-3 gap-y-1">
            {index.kept_sensors.map((s) => (
              <button
                key={s}
                onClick={() => setSensor(s)}
                className={`flex items-baseline justify-between gap-2 rounded px-1.5 py-0.5 text-left text-xs ${s === sensor ? 'bg-surface-2 ring-1 ring-line' : 'hover:bg-surface-2'}`}
              >
                <span className="font-medium">{sensorsByColumn[s].symbol}</span>
                <span className="tabular font-mono text-ink-2">{fmt(engine.sensors[s][i], 2)}</span>
              </button>
            ))}
          </div>
          <div className="mt-3 text-xs text-ink-3">
            Dropped at screening: {dropped.map((s) => s.symbol).join(', ')}
          </div>
        </div>
      </div>
    </Stage>
  )
}

export function FeatureStage({ index, engine, i }: Props) {
  const cats = index.model.feature_categories
  const catLabel: Record<string, string> = { raw: 'Raw current-cycle', rolling_mean: 'Rolling mean', rolling_std: 'Rolling std', delta: 'Delta', trend: 'Trend slope' }
  const cycle = engine.cycle[i]

  return (
    <Stage
      id="stage-features"
      step={2}
      kind="data"
      title="Feature engineering"
      summary={
        <>
          Each kept sensor becomes 7 numbers: its current value plus rolling means and standard deviations over the last{' '}
          {index.model.rolling_windows.join(' and ')} cycles, a one-cycle change, and a {index.model.trend_window}-cycle trend
          slope. With the cycle count and operating settings that makes {index.model.n_features} features per row. Rolling
          features smooth out sensor noise so a slow drift shows up clearly.
        </>
      }
    >
      <div className="mb-4 flex flex-wrap gap-2">
        {Object.entries(cats).map(([c, v]) => (
          <span key={c} className="rounded-md border border-line px-2 py-1 text-xs">
            <span className="text-ink-2">{catLabel[c] ?? c}</span> <span className="tabular font-semibold">{v.count}</span>
          </span>
        ))}
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[520px] text-sm">
          <thead>
            <tr className="border-b border-line text-left text-xs text-ink-3">
              <th className="py-1.5 pr-3 font-medium">Feature (top {index.model.top_features.length} by model importance)</th>
              <th className="py-1.5 pr-3 font-medium">Built from</th>
              <th className="py-1.5 pr-3 text-right font-medium">Value at cycle {cycle}</th>
              <th className="py-1.5 text-right font-medium">Raw sensor now</th>
            </tr>
          </thead>
          <tbody>
            {index.model.top_features.map((f) => (
              <tr key={f.feature} className="border-b border-line/60 last:border-0">
                <td className="py-1.5 pr-3 font-mono text-xs">{f.feature}</td>
                <td className="py-1.5 pr-3 text-xs text-ink-2">
                  {f.sensor ? <><strong className="font-semibold text-ink">{f.symbol}</strong> · {featureLabel(f.transform)}</> : 'Engine age in cycles'}
                </td>
                <td className="tabular py-1.5 pr-3 text-right font-mono text-xs">{fmt(engine.features[f.feature][i], 2)}</td>
                <td className="tabular py-1.5 text-right font-mono text-xs text-ink-3">
                  {f.sensor ? fmt(engine.sensors[f.sensor][i], 2) : cycle}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Stage>
  )
}
