import { useState } from 'react'
import { CartesianGrid, Line, LineChart, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { EngineData, FixtureIndex } from '../../pipeline/data'
import { Caption, Expert, PanelLabel, Stage, axisTick, fmt, tooltipItemStyle, tooltipLabelStyle, tooltipStyle, useWidth } from '../ui'

interface Props {
  index: FixtureIndex
  engine: EngineData
  i: number
  setI: (i: number) => void
}

const TRANSFORM_LABEL: Record<string, string> = {
  raw: 'current value',
  roll_mean_5: 'rolling mean, 5 cycles',
  roll_mean_10: 'rolling mean, 10 cycles',
  roll_std_5: 'rolling std, 5 cycles',
  roll_std_10: 'rolling std, 10 cycles',
  delta_1: 'change since last cycle',
  trend_5: 'trend slope, 5 cycles',
}

export function featureLabel(transform: string) {
  return TRANSFORM_LABEL[transform] ?? transform.replace(/_/g, ' ')
}

/** Step 1: the raw material — one engine's sensors drifting towards failure. */
export function EngineStage({ index, engine, i, setI }: Props) {
  const sensorsByColumn = Object.fromEntries(index.sensors.map((s) => [s.column, s]))
  const [sensor, setSensor] = useState(index.model.top_features.find((f) => f.sensor)?.sensor ?? index.kept_sensors[0])
  const meta = sensorsByColumn[sensor]
  const values = engine.sensors[sensor]
  const data = engine.cycle.map((c, k) => ({ cycle: c, value: values[k] }))
  const cycle = engine.cycle[i]
  const last = engine.cycle[engine.cycle.length - 1]
  const window = Math.max(...index.model.rolling_windows)
  const dropped = index.sensors.filter((s) => s.status !== 'keep')
  const [chartRef, chartWidth] = useWidth<HTMLDivElement>()
  // Plot area excludes the 56px y-axis and 8px right margin.
  const futurePx = ((last - cycle) / Math.max(1, last - engine.cycle[0])) * Math.max(0, chartWidth - 64)

  return (
    <Stage
      id="stage-engine"
      step={1}
      kind="data"
      title="An engine wears out"
      takeaway={
        <>Each flight cycle, the engine reports {index.kept_sensors.length} useful sensor readings. As it degrades they drift, slowly and
        noisily. The model may only ever look backwards: at cycle <em>t</em> it sees cycles ≤ <em>t</em>, never the future.</>
      }
      method={
        <>
          <p>
            The data are run-to-failure trajectories from NASA's C-MAPSS turbofan simulation: each engine is observed once per flight
            cycle, recording 21 sensor channels (temperatures, pressures, shaft speeds, flow ratios) and 3 operating settings, until a
            fault develops and it fails. In the FD001 subset, {dropped.length} channels have near-zero variance and are removed, leaving{' '}
            {index.kept_sensors.length}.
          </p>
          <p>
            Each retained channel is summarised by causal window statistics — rolling means and standard deviations over the last{' '}
            {index.model.rolling_windows.join(' and ')} cycles, the change since the previous cycle and a {index.model.trend_window}-cycle
            least-squares slope — which expose slow degradation beneath cycle-to-cycle noise. With the cycle count and operating settings,
            each cycle becomes {index.model.n_features} features. Using any later cycle would be data leakage: optimistic accuracy that
            could never be achieved in service.
          </p>
        </>
      }
    >
      <div className="mb-4 flex flex-wrap items-center gap-x-3 gap-y-2">
        <label htmlFor="sensor-pick" className="sr-only">Sensor</label>
        <select
          id="sensor-pick"
          value={sensor}
          onChange={(e) => setSensor(e.target.value)}
          className="select h-8 max-w-full rounded-lg bg-surface-2 pl-3 text-sm font-medium"
        >
          {index.kept_sensors.map((s) => (
            <option key={s} value={s}>{sensorsByColumn[s].symbol} — {sensorsByColumn[s].name}</option>
          ))}
        </select>
        <span className="text-xs text-ink-3">
          {meta.unit} · expected to <strong className="font-semibold text-ink-2">{meta.expected_trend.replace(/s$/, '')}</strong> as the engine wears
        </span>
      </div>
      <div ref={chartRef} className="h-64">
        <ResponsiveContainer>
          <LineChart
            data={data}
            margin={{ top: 18, right: 8, bottom: 0, left: 0 }}
            onClick={(e) => typeof e?.activeTooltipIndex === 'number' && setI(e.activeTooltipIndex)}
            style={{ cursor: 'crosshair' }}
          >
            <CartesianGrid stroke="var(--grid)" vertical={false} />
            <XAxis dataKey="cycle" type="number" domain={['dataMin', 'dataMax']} tick={axisTick} stroke="var(--grid)" tickLine={false} />
            <YAxis domain={['auto', 'auto']} tick={axisTick} axisLine={false} tickLine={false} width={56} tickFormatter={(v) => fmt(v, 1)} />
            <ReferenceArea x1={Math.max(engine.cycle[0], cycle - window + 1)} x2={cycle} fill="var(--series-1)" fillOpacity={0.1} ifOverflow="visible" />
            <ReferenceArea
              x1={cycle}
              x2={last}
              fill="var(--surface-2)"
              fillOpacity={0.85}
              label={futurePx >= 170 ? { value: 'Future: hidden from the model', position: 'insideTop', fill: 'var(--ink-3)', fontSize: 10 }
                : futurePx >= 44 ? { value: 'Future', position: 'insideTop', fill: 'var(--ink-3)', fontSize: 10 } : undefined}
            />
            <ReferenceLine x={cycle} stroke="var(--ink)" strokeWidth={1.5} />
            <Tooltip
              contentStyle={tooltipStyle}
              labelStyle={tooltipLabelStyle}
              itemStyle={tooltipItemStyle}
              cursor={{ stroke: 'var(--line-strong)' }}
              labelFormatter={(c) => `Cycle ${c}`}
              formatter={(v) => [fmt(Number(v), 2), meta.symbol]}
            />
            <Line dataKey="value" stroke="var(--series-1)" strokeWidth={1.5} dot={false} isAnimationActive={false}
              activeDot={{ r: 4, stroke: 'var(--surface)', strokeWidth: 2 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <Caption>
        {meta.symbol} ({meta.name}) over engine {engine.unit}'s life. Tinted: the last {window} cycles the model summarises. Grey: cycles
        after {cycle}, which the model cannot see.
      </Caption>

      <Expert>
        <div className="grid gap-8 xl:grid-cols-[260px_minmax(0,1fr)]">
          <div className="min-w-0">
            <PanelLabel right={`cycle ${cycle}`}>All {index.kept_sensors.length} sensors now</PanelLabel>
            <div className="grid grid-cols-2 gap-1">
              {index.kept_sensors.map((s) => {
                const on = s === sensor
                return (
                  <button
                    key={s}
                    onClick={() => setSensor(s)}
                    aria-pressed={on}
                    className={`flex items-baseline justify-between gap-2 rounded-md px-2 py-1 text-left text-xs transition-colors ${on ? 'bg-ink text-page' : 'hover:bg-surface-2'}`}
                  >
                    <span className="font-semibold">{sensorsByColumn[s].symbol}</span>
                    <span className={`tabular font-mono ${on ? '' : 'text-ink-2'}`}>{fmt(engine.sensors[s][i], 2)}</span>
                  </button>
                )
              })}
            </div>
            <div className="mt-4 text-xs leading-relaxed text-ink-3">
              <span className="font-medium text-ink-2">Removed at screening</span> (near-zero variance): {dropped.map((s) => s.symbol).join(', ')}
            </div>
          </div>
          <FeatureTable index={index} engine={engine} i={i} />
        </div>
      </Expert>
    </Stage>
  )
}

function FeatureTable({ index, engine, i }: Omit<Props, 'setI'>) {
  const cats = index.model.feature_categories
  const catLabel: Record<string, string> = { raw: 'Raw value', rolling_mean: 'Rolling mean', rolling_std: 'Rolling std', delta: 'First difference', trend: 'Trend slope' }
  const top = index.model.top_features
  const maxImp = Math.max(...top.map((f) => f.importance))

  return (
    <div className="min-w-0">
      <PanelLabel right={Object.entries(cats).map(([c, v]) => `${v.count} ${(catLabel[c] ?? c).toLowerCase()}`).join(' · ')}>
        {index.model.n_features} features per cycle — the {top.length} the model relies on most
      </PanelLabel>
      <div className="-mx-1 overflow-x-auto">
        <table className="w-full min-w-[520px] border-collapse text-sm">
          <thead>
            <tr className="text-left text-xs text-ink-3">
              <th className="px-1 pb-2 font-medium">Feature</th>
              <th className="px-1 pb-2 font-medium">Importance</th>
              <th className="px-1 pb-2 text-right font-medium">Value now</th>
            </tr>
          </thead>
          <tbody>
            {top.map((f) => (
              <tr key={f.feature} className="border-t border-line">
                <td className="px-1 py-1.5">
                  <div className="text-[13px] font-medium">
                    {f.sensor ? <>{f.symbol} <span className="font-normal text-ink-2">· {featureLabel(f.transform)}</span></> : 'Engine age'}
                  </div>
                </td>
                <td className="w-[36%] px-1 py-1.5">
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 flex-1 rounded-full bg-surface-2">
                      <div className="h-full rounded-full bg-series-1" style={{ width: `${(f.importance / maxImp) * 100}%` }} />
                    </div>
                    <span className="tabular w-11 text-right text-xs text-ink-2">{fmt(f.importance * 100)}%</span>
                  </div>
                </td>
                <td className="tabular px-1 py-1.5 text-right font-mono text-xs">{fmt(engine.features[f.feature][i], 2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Caption>
        Importance is each feature's share of the forest's squared-error reduction (impurity-based) — a property of the model, not physical
        causation.
      </Caption>
    </div>
  )
}
