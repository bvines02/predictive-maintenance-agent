import type { EngineData, FixtureIndex } from '../../pipeline/data'
import type { HealthThresholds } from '../../pipeline/rules'
import { ForestView } from '../ForestView'
import { Caption, ConfidenceBadge, Expert, PanelLabel, Stage, Stat, fmt, useWidth } from '../ui'

interface Props {
  index: FixtureIndex
  engine: EngineData
  i: number
  thresholds: HealthThresholds
}

/** Step 2: the Random Forest's estimate, and how agreement between its trees becomes confidence. */
export function ModelStage({ index, engine, i, thresholds }: Props) {
  const predicted = engine.predicted_rul[i]

  return (
    <Stage
      id="stage-model"
      step={2}
      kind="ml"
      title="A model predicts the life left"
      takeaway={
        <>A Random Forest of {index.model.n_trees} decision trees estimates how many cycles remain. Its answer is the average of the
        trees; how much they disagree tells you how far to trust it. On engines it never trained on, it is off by{' '}
        {index.model.validation_mae} cycles on average.</>
      }
      method={
        <>
          <p>
            A regression tree recursively partitions feature space with axis-aligned threshold splits, chosen to minimise squared error,
            and predicts the mean target within each leaf. Deep trees have low bias but high variance, so a Random Forest averages{' '}
            {index.model.n_trees} trees, each fitted to a bootstrap resample of the training rows (bagging).
          </p>
          <p>
            The target is remaining useful life (RUL): the unit's final cycle minus the current cycle, clipped at {index.dataset.rul_cap}{' '}
            because early degradation is not observable. Engines are split into training and validation sets by unit, never by row; every
            engine shown here was held out. Validation MAE is {index.model.validation_mae} cycles and RMSE {index.model.validation_rmse}.
          </p>
          <p>
            The standard deviation of the tree estimates is discretised into HIGH / MEDIUM / LOW confidence using tercile cut-points fixed
            on the training data. It is uncalibrated: a measure of agreement between trees, not a probability of being correct.
          </p>
        </>
      }
    >
      <div className="flex flex-wrap items-end gap-x-10 gap-y-4">
        <Stat emphasis label="Predicted remaining life" value={<>{fmt(predicted)} <span className="text-base font-medium text-ink-3">cycles</span></>} />
        <Stat label="Middle 80% of trees" value={`${fmt(engine.tree_quantiles.p10[i], 0)}–${fmt(engine.tree_quantiles.p90[i], 0)}`} />
        <div className="pb-1"><ConfidenceBadge confidence={engine.model_confidence[i]} /></div>
      </div>

      <div className="mt-8 border-t border-line pt-6">
        <ForestView index={index} engine={engine} i={i} thresholds={thresholds} />
      </div>

      <Expert>
        <ConfidenceMeter index={index} engine={engine} i={i} />
      </Expert>
    </Stage>
  )
}

function ConfidenceMeter({ index, engine, i }: Omit<Props, 'thresholds'>) {
  const std = engine.prediction_std[i]
  const { prediction_std_low_max: lo, prediction_std_medium_max: mid } = index.uncertainty
  const max = Math.max(mid * 1.8, std * 1.05)
  const f = (v: number) => Math.min(v, max) / max
  const conf = engine.model_confidence[i]
  const [ref, width] = useWidth<HTMLDivElement>()
  const zones = [
    { from: 0, to: lo, label: 'High confidence', level: 'HIGH', shade: 0.08 },
    { from: lo, to: mid, label: 'Medium', level: 'MEDIUM', shade: 0.16 },
    { from: mid, to: max, label: 'Low', level: 'LOW', shade: 0.28 },
  ]

  return (
    <div>
      <PanelLabel right={<>tree standard deviation <span className="tabular font-semibold text-ink">{fmt(std, 2)}</span> cycles</>}>
        From tree disagreement to a confidence label
      </PanelLabel>
      <div ref={ref} className="relative mt-3 h-10">
        {zones.map((z) => {
          const w = (f(z.to) - f(z.from)) * width
          const on = z.level === conf
          return (
            <div key={z.label} className="absolute inset-y-0" style={{ left: `calc(${f(z.from) * 100}% + 1px)`, width: `calc(${(f(z.to) - f(z.from)) * 100}% - 2px)` }}>
              <div className="absolute inset-0 rounded-[4px]" style={{ background: 'var(--ink)', opacity: z.shade }} />
              {on && <div className="absolute inset-0 rounded-[4px]" style={{ boxShadow: 'inset 0 0 0 1.5px var(--ink)' }} />}
              {w >= z.label.length * 6.4 + 14 && (
                <span className={`relative block whitespace-nowrap px-2 pt-1 text-[10px] ${on ? 'font-bold text-ink' : 'font-semibold text-ink-2'}`}>{z.label}</span>
              )}
            </div>
          )
        })}
        <div className="absolute inset-y-[-5px] w-[2px] -translate-x-1/2 rounded-full bg-series-1" style={{ left: `${f(std) * 100}%`, boxShadow: '0 0 0 2px var(--surface)' }} />
      </div>
      <div className="relative mt-1.5 h-4">
        {[lo, mid].map((v) => (
          <span key={v} className="tabular absolute -translate-x-1/2 text-[10px] text-ink-3" style={{ left: `${f(v) * 100}%` }}>{fmt(v, 2)}</span>
        ))}
      </div>
      <Caption>
        Cut-points are the terciles of tree standard deviation on the training rows. The label feeds rule CONF_01; on the validation
        engines, tree standard deviation correlated with absolute error (r ≈ 0.64).
      </Caption>
    </div>
  )
}
