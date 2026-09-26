import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { EngineData, FixtureIndex, TreeQuantile } from '../../pipeline/data'
import type { HealthThresholds } from '../../pipeline/rules'
import { RUL_AXIS_MAX, RulScale, ScaleMarker, SpreadOverlay, healthZones } from '../RulScale'
import { ConfidenceBadge, STATE_COLOR, Stage, Stat, fmt } from '../ui'
import { featureLabel } from './DataStages'

interface Props {
  index: FixtureIndex
  engine: EngineData
  i: number
  setI: (i: number) => void
  thresholds: HealthThresholds
}

const tooltipStyle = { background: 'var(--surface)', border: '1px solid var(--line)', borderRadius: 8, fontSize: 12 }

export function ModelStage({ index, engine, i, setI, thresholds }: Props) {
  const q = Object.fromEntries(
    Object.entries(engine.tree_quantiles).map(([k, v]) => [k, v[i]]),
  ) as Record<TreeQuantile, number>
  const predicted = engine.predicted_rul[i]
  const actual = engine.actual_rul_capped[i]
  const heldOut = engine.split === 'validation'

  const importance = index.model.top_features.map((f) => ({
    name: f.sensor ? `${f.symbol} · ${featureLabel(f.transform)}` : 'Engine age (cycles)',
    importance: f.importance * 100,
  }))

  const life = engine.cycle.map((c, k) => ({
    cycle: c,
    predicted: engine.predicted_rul[k],
    actual: engine.actual_rul_capped[k],
    band: [engine.tree_quantiles.p10[k], engine.tree_quantiles.p90[k]] as [number, number],
  }))

  return (
    <Stage
      id="stage-model"
      step={3}
      kind="ml"
      title="Random Forest predicts remaining useful life"
      summary={
        <>
          {index.model.n_trees} decision trees each read the {index.model.n_features} features and give their own estimate of
          cycles left. The prediction is simply their average. The model was trained on {index.model.target.replace('_', ' ')} —
          remaining life capped at {index.dataset.rul_cap}, because early in life there is nothing to distinguish 200 cycles from 300.
          Validation error: MAE {index.model.validation_mae}, RMSE {index.model.validation_rmse} cycles.
        </>
      }
    >
      {!heldOut && (
        <div className="mb-4 rounded-md border px-3 py-2 text-xs" style={{ borderColor: 'var(--serious)', background: 'color-mix(in srgb, var(--serious) 10%, transparent)' }}>
          <strong>In-sample engine.</strong> Engine {engine.unit} was one of the 80 the model trained on, so these predictions
          are fitted, not forecast, and look better than the model really is. Pick a held-out engine for honest error.
        </div>
      )}

      <div className="mb-5 flex flex-wrap gap-x-8 gap-y-3">
        <Stat label="Predicted RUL" value={<>{fmt(predicted)} <span className="text-sm font-normal text-ink-3">cycles</span></>} />
        <Stat label="Actual RUL (capped)" value={actual} sub={`error ${predicted - actual >= 0 ? '+' : ''}${fmt(predicted - actual)} cycles`} />
        <Stat label="Middle 80% of trees" value={`${fmt(q.p10, 0)}–${fmt(q.p90, 0)}`} sub="p10 to p90 tree estimate" />
      </div>

      <div className="mb-1 text-xs font-medium text-ink-2">
        What the {index.model.n_trees} trees said at cycle {engine.cycle[i]}, against today's health bands
      </div>
      <RulScale zones={healthZones(thresholds)} zoneOpacity={0.22} height={52}>
        <SpreadOverlay q={q} />
        <ScaleMarker value={predicted} label={`mean ${fmt(predicted)}`} />
        <ScaleMarker value={actual} label={`actual ${actual}`} color="var(--series-2)" below />
      </RulScale>
      <p className="mb-5 mt-4 text-xs text-ink-3">
        Line: full range of tree estimates. Filled bar: middle 80%. Outlined box: middle 50%. When the spread straddles a
        band edge, trees disagree about which health state the engine is in.
      </p>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <div className="min-w-0">
          <div className="mb-1 flex flex-wrap items-center justify-between gap-2">
            <div className="text-xs font-medium text-ink-2">Predicted vs actual RUL over engine {engine.unit}'s life</div>
            <div className="flex gap-3 text-xs text-ink-2">
              <span className="inline-flex items-center gap-1"><span className="h-0.5 w-4" style={{ background: 'var(--series-1)' }} />Predicted</span>
              <span className="inline-flex items-center gap-1"><span className="h-2.5 w-4 rounded-sm" style={{ background: 'var(--band)' }} />Trees p10–p90</span>
              <span className="inline-flex items-center gap-1"><span className="h-0.5 w-4 border-t-2 border-dashed" style={{ borderColor: 'var(--series-2)' }} />Actual</span>
            </div>
          </div>
          <div className="h-64">
            <ResponsiveContainer>
              <ComposedChart
                data={life}
                margin={{ top: 8, right: 44, bottom: 0, left: 0 }}
                onClick={(e) => typeof e?.activeTooltipIndex === 'number' && setI(e.activeTooltipIndex)}
              >
                <CartesianGrid stroke="var(--line)" vertical={false} />
                <XAxis dataKey="cycle" type="number" domain={['dataMin', 'dataMax']} tick={{ fill: 'var(--ink-3)', fontSize: 11 }} stroke="var(--line)" />
                <YAxis domain={[0, RUL_AXIS_MAX]} ticks={[0, 25, 50, 75, 100, 125]} tick={{ fill: 'var(--ink-3)', fontSize: 11 }} stroke="var(--line)" width={36} />
                {(['action', 'plan', 'watch'] as const).map((k) => (
                  <ReferenceLine
                    key={k}
                    y={thresholds[k]}
                    stroke={STATE_COLOR[k === 'action' ? 'ACTION' : k === 'plan' ? 'PLAN' : 'WATCH']}
                    strokeDasharray="4 3"
                    strokeWidth={1.5}
                    label={{ value: k.toUpperCase(), position: 'right', fill: 'var(--ink-2)', fontSize: 10 }}
                  />
                ))}
                <ReferenceLine x={engine.cycle[i]} stroke="var(--ink)" strokeWidth={1.5} />
                <Tooltip
                  contentStyle={tooltipStyle}
                  labelFormatter={(c) => `Cycle ${c}`}
                  formatter={(v, name) =>
                    Array.isArray(v) ? [`${fmt(v[0])} – ${fmt(v[1])}`, 'Trees p10–p90'] : [fmt(Number(v)), name === 'predicted' ? 'Predicted' : 'Actual']
                  }
                />
                <Area dataKey="band" stroke="none" fill="var(--band)" fillOpacity={1} isAnimationActive={false} activeDot={false} />
                <Line dataKey="actual" stroke="var(--series-2)" strokeWidth={2} strokeDasharray="5 4" dot={false} isAnimationActive={false} />
                <Line dataKey="predicted" stroke="var(--series-1)" strokeWidth={2} dot={false} isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-1 text-xs text-ink-3">Click the chart to jump to a cycle. Dashed horizontals are your current health thresholds.</p>
        </div>

        <div className="min-w-0">
          <div className="mb-1 text-xs font-medium text-ink-2">What the forest relies on (feature importance, % of total)</div>
          <div className="h-72">
            <ResponsiveContainer>
              <BarChart data={importance} layout="vertical" margin={{ top: 0, right: 36, bottom: 0, left: 0 }} barCategoryGap={3}>
                <XAxis type="number" hide />
                <YAxis type="category" dataKey="name" width={170} tick={{ fill: 'var(--ink-2)', fontSize: 11 }} stroke="none" interval={0} />
                <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'var(--surface-2)' }} formatter={(v) => [`${fmt(Number(v))}%`, 'Importance']} />
                <Bar dataKey="importance" fill="var(--series-1)" radius={[0, 4, 4, 0]} isAnimationActive={false}
                  label={{ position: 'right', fill: 'var(--ink-2)', fontSize: 10, formatter: (v: unknown) => `${fmt(Number(v))}%` }} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-1 text-xs text-ink-3">
            How much each feature reduced error inside this model — not physical causation. Engine age ranks high: the
            model partly learns "older engines have less life left".
          </p>
        </div>
      </div>
    </Stage>
  )
}

export function ConfidenceStage({ index, engine, i }: Omit<Props, 'thresholds' | 'setI'>) {
  const std = engine.prediction_std[i]
  const { prediction_std_low_max: lo, prediction_std_medium_max: mid } = index.uncertainty
  const max = Math.max(mid * 1.8, std * 1.05)
  const pos = (v: number) => `${(Math.min(v, max) / max) * 100}%`
  const conf = engine.model_confidence[i]
  const zones = [
    { from: 0, to: lo, label: 'HIGH confidence', shade: 0.1 },
    { from: lo, to: mid, label: 'MEDIUM', shade: 0.22 },
    { from: mid, to: max, label: 'LOW', shade: 0.36 },
  ]

  return (
    <Stage
      id="stage-confidence"
      step={4}
      kind="ml"
      title="Tree disagreement becomes model confidence"
      summary={
        <>
          The spread of the {index.model.n_trees} tree estimates (their standard deviation) is turned into a confidence label
          using fixed cut points taken from training data: the third of rows where trees agreed most are HIGH, the third where
          they disagreed most are LOW. It measures agreement between trees, not a calibrated probability of being right.
        </>
      }
    >
      <div className="mb-4 flex flex-wrap items-center gap-x-8 gap-y-3">
        <Stat label="Tree std deviation" value={<>{fmt(std, 2)} <span className="text-sm font-normal text-ink-3">cycles</span></>} />
        <ConfidenceBadge confidence={conf} />
      </div>
      <div className="pb-5">
        <div className="relative h-9 rounded-md bg-surface-2">
          {zones.map((z) => (
            <div key={z.label} className="absolute inset-y-0 overflow-hidden border-r-2 border-surface" style={{ left: pos(z.from), width: `calc(${pos(z.to)} - ${pos(z.from)})` }}>
              <div className="absolute inset-0" style={{ background: 'var(--ink)', opacity: z.shade }} />
              <span className="relative truncate px-1.5 text-[10px] font-semibold">{z.label}</span>
            </div>
          ))}
          <div className="absolute inset-y-[-4px] w-[3px] -translate-x-1/2 rounded-full bg-series-1" style={{ left: pos(std) }} />
          {[lo, mid].map((v) => (
            <span key={v} className="tabular absolute top-full mt-1 -translate-x-1/2 text-[10px] text-ink-3" style={{ left: pos(v) }}>
              {fmt(v, 2)}
            </span>
          ))}
        </div>
      </div>
      <p className="text-xs text-ink-3">
        This label feeds rule CONF_01 in the decision engine. In Step 10, higher disagreement correlated with larger error (r ≈ 0.64 on validation).
      </p>
    </Stage>
  )
}
