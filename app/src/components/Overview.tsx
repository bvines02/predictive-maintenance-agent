import { Area, CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { EngineData, FixtureIndex } from '../pipeline/data'
import type { Decision, HealthThresholds } from '../pipeline/rules'
import { LifeStrip } from './LifeStrip'
import { RUL_AXIS_MAX } from './RulScale'
import {
  ActionBadge,
  ActionLegend,
  ConfidenceBadge,
  Expert,
  HealthBadge,
  ReviewFlag,
  STATE_COLOR,
  Swatch,
  actionColor,
  actionLabel,
  axisTick,
  fmt,
  stateLabel,
  tooltipItemStyle,
  tooltipLabelStyle,
  tooltipStyle,
  useExpert,
} from './ui'

interface Props {
  index: FixtureIndex
  engine: EngineData
  i: number
  setI: (i: number) => void
  decisions: Decision[]
  thresholds: HealthThresholds
}

const Y_AXIS_W = 36
const PLOT_RIGHT = 68

export function Overview({ index, engine, i, setI, decisions, thresholds: t }: Props) {
  const d = decisions[i]
  const expert = useExpert()
  const predicted = engine.predicted_rul[i]
  const actual = engine.actual_rul_capped[i]
  const p10 = engine.tree_quantiles.p10[i]
  const p90 = engine.tree_quantiles.p90[i]
  const error = predicted - actual
  const last = engine.cycle[engine.cycle.length - 1]

  const life = engine.cycle.map((c, k) => ({
    cycle: c,
    predicted: engine.predicted_rul[k],
    actual: engine.actual_rul_capped[k],
    band: [engine.tree_quantiles.p10[k], engine.tree_quantiles.p90[k]] as [number, number],
  }))

  const thresholdLines = [
    { y: t.action, state: 'ACTION' as const },
    { y: t.plan, state: 'PLAN' as const },
    { y: t.watch, state: 'WATCH' as const },
  ]

  return (
    <section id="overview" className="scroll-mt-32" aria-labelledby="overview-title">
      <div className="card overflow-hidden">
        <div className="grid gap-px bg-line lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
          {/* Prediction */}
          <div className="bg-surface p-6">
            <div className="eyebrow mb-3" id="overview-title">Predicted remaining useful life</div>
            <div className="flex items-baseline gap-2">
              <span className="text-[56px] font-semibold leading-none tracking-tight">{fmt(predicted)}</span>
              <span className="text-lg font-medium text-ink-2">cycles</span>
            </div>
            <div className="mt-3 text-sm text-ink-2">
              80% of trees estimate <span className="font-semibold text-ink">{fmt(p10, 0)}–{fmt(p90, 0)}</span> cycles
            </div>
            <div className="mt-4 flex flex-wrap items-center gap-2">
              <ConfidenceBadge confidence={engine.model_confidence[i]} />
            </div>
            <div className="mt-5 flex items-center gap-4 border-t border-line pt-4 text-sm">
              <div>
                <div className="text-xs text-ink-3">Ground truth</div>
                <div className="font-semibold">{actual} cycles</div>
              </div>
              <div className="h-8 w-px bg-line" />
              <div>
                <div className="text-xs text-ink-3">Prediction error</div>
                <div className="font-semibold">{error >= 0 ? '+' : '−'}{fmt(Math.abs(error))} cycles</div>
              </div>
              <div className="ml-auto hidden max-w-[11rem] text-right text-[11px] leading-snug text-ink-3 sm:block">
                Known only because this is a historical run-to-failure record.
              </div>
            </div>
          </div>

          {/* Decision */}
          <div className="bg-surface p-6">
            <div className="eyebrow mb-3">Maintenance decision</div>
            <div className="flex flex-wrap items-center gap-2.5">
              <HealthBadge state={d.healthState} size="lg" />
              <svg width="16" height="10" viewBox="0 0 16 10" fill="none" stroke="var(--ink-3)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                <path d="M1 5h13M10 1l4 4-4 4" />
              </svg>
              <ActionBadge action={d.recommendedAction} size="lg" />
              {d.requiresHumanReview && <ReviewFlag />}
            </div>
            <p className="mt-4 max-w-md text-[15px] leading-relaxed text-ink">{index.action_meanings[d.recommendedAction]}</p>
            {!expert && (
              <p className="mt-5 border-t border-line pt-4 text-sm leading-relaxed text-ink-2">
                Set by fixed rules you can edit, not by the AI.{' '}
                <a href="#stage-decision" className="font-medium text-ink underline decoration-line-strong underline-offset-2 hover:decoration-ink">
                  See why →
                </a>
              </p>
            )}
            <Expert bare>
            <div className="mt-5 border-t border-line pt-4">
              <div className="mb-2 text-xs text-ink-3">Rules applied, in order</div>
              <ol className="flex flex-wrap items-center gap-1.5">
                {d.trace.map((s, k) => (
                  <li key={s.ruleId} className="flex items-center gap-1.5">
                    {k > 0 && <span className="text-ink-3" aria-hidden>→</span>}
                    <a href="#stage-decision" className="rounded-md bg-surface-2 px-2 py-1 font-mono text-xs font-medium hover:bg-sunken">
                      {s.ruleId}
                    </a>
                  </li>
                ))}
              </ol>
              <p className="mt-3 text-xs leading-relaxed text-ink-3">
                Health state from the <strong className="font-semibold text-ink-2">{stateLabel(d.healthState)}</strong> threshold; action from the rule set.
                Both recompute live as you change the policy.
              </p>
            </div>
            </Expert>
          </div>
        </div>

        {/* Engine life */}
        <div className="border-t border-line p-6">
          <div className="mb-3 flex flex-wrap items-baseline justify-between gap-x-6 gap-y-2">
            <div>
              <h3 className="text-sm font-semibold">Engine {engine.unit} over its life</h3>
              <p className="text-xs text-ink-3">Click the chart, or drag along the strip, to move through cycles 1–{last}.</p>
            </div>
            <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2">
              <span className="inline-flex items-center gap-1.5"><Swatch shape="line" color="var(--series-1)" />Predicted</span>
              <span className="inline-flex items-center gap-1.5"><Swatch shape="band" color="color-mix(in srgb, var(--series-1) 24%, transparent)" />Middle 80% of trees</span>
              <span className="inline-flex items-center gap-1.5"><Swatch shape="dash" color="var(--series-2)" />Actual (capped at {index.dataset.rul_cap})</span>
            </div>
          </div>

          <div className="h-[260px]">
            <ResponsiveContainer>
              <ComposedChart
                data={life}
                margin={{ top: 6, right: PLOT_RIGHT, bottom: 0, left: 0 }}
                onClick={(e) => typeof e?.activeTooltipIndex === 'number' && setI(e.activeTooltipIndex)}
                style={{ cursor: 'crosshair' }}
              >
                <CartesianGrid stroke="var(--grid)" vertical={false} />
                {thresholdLines.map((z) => (
                  <ReferenceLine
                    key={z.state}
                    y={z.y}
                    stroke={STATE_COLOR[z.state]}
                    strokeWidth={1.25}
                    ifOverflow="hidden"
                    label={{ value: `${stateLabel(z.state)} ≤ ${z.y}`, position: 'right', fill: 'var(--ink-2)', fontSize: 10, offset: 6 }}
                  />
                ))}
                <XAxis dataKey="cycle" type="number" domain={['dataMin', 'dataMax']} tick={axisTick} stroke="var(--grid)" tickLine={false} />
                <YAxis domain={[0, RUL_AXIS_MAX]} ticks={[0, 25, 50, 75, 100, 125]} tick={axisTick} axisLine={false} tickLine={false} width={Y_AXIS_W} />
                <Tooltip
                  contentStyle={tooltipStyle}
                  labelStyle={tooltipLabelStyle}
                  itemStyle={tooltipItemStyle}
                  cursor={{ stroke: 'var(--line-strong)', strokeWidth: 1 }}
                  labelFormatter={(c) => `Cycle ${c}`}
                  formatter={(v, name) =>
                    Array.isArray(v)
                      ? [`${fmt(v[0])} – ${fmt(v[1])}`, 'Middle 80% of trees']
                      : [fmt(Number(v)), name === 'predicted' ? 'Predicted RUL' : 'Actual RUL']
                  }
                />
                <Area dataKey="band" stroke="none" fill="var(--series-1)" fillOpacity={0.16} isAnimationActive={false} activeDot={false} />
                <Line dataKey="actual" stroke="var(--series-2)" strokeWidth={2} strokeDasharray="5 4" dot={false} isAnimationActive={false} activeDot={false} />
                <Line dataKey="predicted" stroke="var(--series-1)" strokeWidth={2} dot={false} isAnimationActive={false}
                  activeDot={{ r: 4, stroke: 'var(--surface)', strokeWidth: 2, fill: 'var(--series-1)' }} />
                <ReferenceLine x={engine.cycle[i]} stroke="var(--ink)" strokeWidth={1.5} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          <div className="mt-4" style={{ paddingLeft: Y_AXIS_W, paddingRight: PLOT_RIGHT }}>
            <div className="mb-1.5 text-xs font-medium text-ink-2">Recommended action at each cycle</div>
            <LifeStrip
              colors={decisions.map((dec) => actionColor(dec.recommendedAction))}
              cursor={i}
              onSelect={setI}
              height={14}
              tooltip={(k) => (
                <>Cycle {engine.cycle[k]} · <strong>{actionLabel(decisions[k].recommendedAction)}</strong></>
              )}
            />
            <div className="mt-2.5"><ActionLegend /></div>
          </div>
        </div>
      </div>
    </section>
  )
}
