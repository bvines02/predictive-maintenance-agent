import type { EngineData, FixtureIndex } from '../../pipeline/data'
import { assignHealthState, severity, type Decision, type DecisionParams } from '../../pipeline/rules'
import { LifeStrip } from '../LifeStrip'
import { RUL_AXIS_MAX, RulScale, ScaleMarker, healthZones } from '../RulScale'
import { ActionBadge, ActionLegend, ConfidenceBadge, HealthBadge, STATE_COLOR, Stage, StateLegend, actionColor, actionLabel, fmt } from '../ui'

interface Props {
  index: FixtureIndex
  engine: EngineData
  i: number
  setI: (i: number) => void
  decisions: Decision[]
  params: DecisionParams
}

export function HealthStage({ engine, i, setI, decisions, params }: Props) {
  const t = params.thresholds
  const d = decisions[i]
  const actualStates = engine.actual_rul_capped.map((r) => assignHealthState(r, t))
  const agree = decisions.filter((dec, k) => dec.healthState === actualStates[k]).length

  return (
    <Stage
      id="stage-health"
      step={5}
      kind="rules"
      title="Health state: a fixed lookup, not a model"
      summary={
        <>
          The predicted RUL is mapped to one of four states by three thresholds you control in the panel. This is a policy
          choice about planning horizons, so it is a plain rule anyone can audit, not something learned. Applying the same
          thresholds to the <em>actual</em> RUL shows where the model's state would have been wrong.
        </>
      }
    >
      <div className="mb-4 flex flex-wrap items-center gap-3">
        <HealthBadge state={d.healthState} size="lg" />
        <span className="text-sm text-ink-2">
          predicted {fmt(d.predictedRul)} cycles{' '}
          {d.healthState === 'HEALTHY' ? `> ${t.watch}` : `≤ ${d.healthState === 'ACTION' ? t.action : d.healthState === 'PLAN' ? t.plan : t.watch}`}
        </span>
      </div>
      <RulScale zones={healthZones(t)} height={40}>
        <ScaleMarker value={Math.min(d.predictedRul, RUL_AXIS_MAX)} label={fmt(d.predictedRul)} />
      </RulScale>

      <div className="mt-4 space-y-2">
        <LifeStrip
          label="Predicted"
          colors={decisions.map((dec) => STATE_COLOR[dec.healthState])}
          cursor={i}
          onSelect={setI}
          tooltip={(k) => <>Cycle {engine.cycle[k]}: <strong>{decisions[k].healthState}</strong> (RUL {fmt(decisions[k].predictedRul)})</>}
        />
        <LifeStrip
          label="Actual"
          colors={actualStates.map((s) => STATE_COLOR[s])}
          cursor={i}
          onSelect={setI}
          tooltip={(k) => <>Cycle {engine.cycle[k]}: <strong>{actualStates[k]}</strong> (actual RUL {engine.actual_rul_capped[k]})</>}
        />
        <div className="flex flex-wrap items-center justify-between gap-2 pl-[92px]">
          <StateLegend />
          <span className="tabular text-xs text-ink-3">
            Predicted state matches actual on {Math.round((agree / decisions.length) * 100)}% of this engine's cycles
          </span>
        </div>
      </div>
    </Stage>
  )
}

export function DecisionStage({ index, engine, i, setI, decisions, params }: Props) {
  const d = decisions[i]
  const lead = params.leadTimeCycles
  const buffer = params.leadTimeBufferCycles
  const firstReview = decisions.findIndex((dec) => dec.requiresHumanReview)
  const firstIntervene = decisions.findIndex((dec) => dec.recommendedAction === 'INTERVENE_NOW')

  return (
    <Stage
      id="stage-decision"
      step={6}
      kind="rules"
      title="Decision engine: health state + lead time → one action"
      summary={
        <>
          The health state sets a starting action (BASE_01). The maintenance lead time then escalates it: if predicted RUL is
          within the lead time, work started now may not finish before failure (LEAD_01, intervene now); within the lead time
          plus a {buffer}-cycle buffer, the work must be committed now (LEAD_02). Low model confidence steps an intrusive
          action back to inspection first (CONF_01). Criticality and redundancy are excluded here.
        </>
      }
    >
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <div className="min-w-0">
          <div className="mb-2 text-xs font-medium text-ink-2">Inputs</div>
          <div className="mb-4 flex flex-wrap items-center gap-2 text-sm">
            <HealthBadge state={d.healthState} />
            <span className="tabular rounded-md border border-line px-2 py-0.5 text-xs font-semibold">RUL {fmt(d.predictedRul)}</span>
            <span className="tabular rounded-md border border-line px-2 py-0.5 text-xs font-semibold">lead time {lead}</span>
            <ConfidenceBadge confidence={engine.model_confidence[i]} />
          </div>

          <div className="mb-2 text-xs font-medium text-ink-2">Rule trace</div>
          <ol className="space-y-2">
            {d.trace.map((step) => (
              <li key={step.ruleId} className="rounded-md border border-line bg-page/40 px-3 py-2">
                <div className="mb-1 flex flex-wrap items-center gap-2">
                  <span className="font-mono text-xs font-semibold">{step.ruleId}</span>
                  <span className="text-[10px] uppercase tracking-wide text-ink-3">
                    {step.kind === 'base' ? 'starting point' : step.kind === 'escalate' ? 'escalates' : 'steps down'}
                  </span>
                  <span className="ml-auto flex items-center gap-1.5">
                    {step.actionBefore && (<><ActionBadge action={step.actionBefore} /><span className="text-ink-3">→</span></>)}
                    <ActionBadge action={step.actionAfter} />
                  </span>
                </div>
                <p className="text-xs leading-relaxed text-ink-2">{step.reason}</p>
              </li>
            ))}
          </ol>
          {d.trace.length === 1 && (
            <p className="mt-2 text-xs text-ink-3">No lead-time or confidence rule changed the starting action.</p>
          )}
        </div>

        <div className="min-w-0">
          <div className="mb-2 text-xs font-medium text-ink-2">Recommended action</div>
          <div className="mb-1 flex flex-wrap items-center gap-2">
            <ActionBadge action={d.recommendedAction} size="lg" />
            {d.requiresHumanReview && (
              <span className="rounded-full border border-ink px-2 py-0.5 text-xs font-semibold">Human review required</span>
            )}
          </div>
          <p className="mb-5 text-sm text-ink-2">{index.action_meanings[d.recommendedAction]}</p>

          <div className="mb-1 text-xs font-medium text-ink-2">Lead-time zones on the RUL axis</div>
          <RulScale
            height={36}
            zoneOpacity={1}
            zones={[
              { from: 0, to: lead, color: actionColor('INTERVENE_NOW'), textColor: 'var(--act-on-5)', label: 'LEAD_01' },
              { from: lead, to: lead + buffer, color: actionColor('SCHEDULE_MAINTENANCE'), textColor: 'var(--act-on-4)', label: 'LEAD_02' },
              { from: lead + buffer, to: RUL_AXIS_MAX, color: 'var(--surface-2)', label: 'health state decides' },
            ]}
          >
            <ScaleMarker value={Math.min(d.predictedRul, RUL_AXIS_MAX)} label={fmt(d.predictedRul)} />
          </RulScale>
        </div>
      </div>

      <div className="mt-5 space-y-2">
        <LifeStrip
          label="Action"
          colors={decisions.map((dec) => actionColor(dec.recommendedAction))}
          cursor={i}
          onSelect={setI}
          height={22}
          tooltip={(k) => (
            <>Cycle {engine.cycle[k]}: <strong>{actionLabel(decisions[k].recommendedAction)}</strong> · {decisions[k].ruleIds.join(' → ')}</>
          )}
        />
        <div className="pl-[92px]"><ActionLegend /></div>
      </div>

      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        <Milestone
          label="First cycle needing human review"
          k={firstReview}
          engine={engine}
          onJump={setI}
        />
        <Milestone label="First INTERVENE NOW" k={firstIntervene} engine={engine} onJump={setI} />
      </div>
      {severity(d.recommendedAction) < severity(d.baseAction) && (
        <p className="mt-3 text-xs text-ink-3">Action is below its health-state baseline only because CONF_01 fired — the one rule allowed to do that.</p>
      )}
    </Stage>
  )
}

function Milestone({ label, k, engine, onJump }: { label: string; k: number; engine: EngineData; onJump: (i: number) => void }) {
  if (k < 0) {
    return (
      <div className="rounded-md border border-line px-3 py-2 text-xs text-ink-3">
        {label}: <strong className="text-ink">never</strong> in this engine's life
      </div>
    )
  }
  const remaining = engine.actual_rul[k]
  return (
    <button onClick={() => onJump(k)} className="rounded-md border border-line px-3 py-2 text-left text-xs hover:bg-surface-2">
      <span className="text-ink-3">{label}: </span>
      <strong className="tabular">cycle {engine.cycle[k]}</strong>
      <span className="text-ink-2"> — actually {remaining} cycles before failure</span>
    </button>
  )
}
