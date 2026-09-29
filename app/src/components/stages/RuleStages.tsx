import type { ReactNode } from 'react'
import type { EngineData, FixtureIndex } from '../../pipeline/data'
import { assignHealthState, severity, type Decision, type DecisionParams } from '../../pipeline/rules'
import { LifeStrip } from '../LifeStrip'
import { RUL_AXIS_MAX, RulScale, ScaleMarker, healthZones } from '../RulScale'
import {
  ActionBadge,
  Expert,
  HealthBadge,
  PanelLabel,
  ReviewFlag,
  STATE_COLOR,
  Stage,
  StateLegend,
  Swatch,
  actionColor,
  fmt,
  stateLabel,
  useExpert,
} from '../ui'

interface Props {
  index: FixtureIndex
  engine: EngineData
  i: number
  setI: (i: number) => void
  decisions: Decision[]
  params: DecisionParams
}

/** Step 3: fixed, editable rules turn the estimate into a health state and one action. */
export function DecisionStage({ index, engine, i, setI, decisions, params }: Props) {
  const d = decisions[i]
  const t = params.thresholds
  const lead = params.leadTimeCycles
  const buffer = params.leadTimeBufferCycles
  const expert = useExpert()
  const rul = Math.min(d.predictedRul, RUL_AXIS_MAX)
  const bound = d.healthState === 'HEALTHY' ? `above ${t.watch}` : `at or below ${d.healthState === 'ACTION' ? t.action : d.healthState === 'PLAN' ? t.plan : t.watch}`

  return (
    <Stage
      id="stage-decision"
      step={3}
      kind="rules"
      title="Rules decide what to do"
      takeaway={
        <>The model's number doesn't decide anything. Plain, auditable rules do: thresholds set the health state, a lead-time check asks
        whether there's still time to plan the work, and low confidence asks for an inspection first. Change the rules in the policy
        panel and every decision updates.</>
      }
      method={
        <>
          <p>
            The continuous RUL estimate is discretised into four ordinal health states by three thresholds. The mapping is deterministic
            and deliberately not learned: the thresholds encode organisational policy about planning horizons, so they are explicit,
            auditable and adjustable without retraining. Separating estimation (what the model believes) from policy (what the organisation
            does about it) is the core design principle of this pipeline.
          </p>
          <p>
            An ordered rule set then maps state, lead time and confidence to one action, recording every rule that fired. BASE_01 sets a
            baseline from the health state. LEAD_01 escalates to INTERVENE_NOW when predicted RUL is within the maintenance lead time;
            LEAD_02 escalates to scheduling when it is within the lead time plus a {buffer}-cycle buffer. CONF_01 is the only rule allowed to
            de-escalate: under LOW confidence it replaces an intrusive action with inspection. Asset criticality and redundancy are
            excluded from this view.
          </p>
        </>
      }
    >
      <div className="grid gap-x-10 gap-y-8 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <ol className="min-w-0 space-y-7">
          <SubStep n={1} title="Health state from the thresholds">
            <div className="flex flex-wrap items-center gap-2 text-sm text-ink-2">
              <HealthBadge state={d.healthState} />
              predicted <span className="tabular font-semibold text-ink">{fmt(d.predictedRul)}</span> cycles, {bound}
            </div>
            <RulScale zones={healthZones(t)} height={30} axisLabel={null}>
              <ScaleMarker value={rul} label={fmt(d.predictedRul)} />
            </RulScale>
          </SubStep>

          <SubStep n={2} title="Is there still time to plan the work?">
            <p className="text-[13px] leading-relaxed text-ink-2">
              Maintenance takes {lead} cycles to complete (plus a {buffer}-cycle buffer). If the engine may fail sooner than that, waiting
              is not an option.
            </p>
            <RulScale
              height={26}
              zoneOpacity={1}
              labels={false}
              axisLabel={null}
              zones={[
                { from: 0, to: lead, color: actionColor('INTERVENE_NOW'), label: 'intervene' },
                { from: lead, to: lead + buffer, color: actionColor('SCHEDULE_MAINTENANCE'), label: 'schedule' },
                { from: lead + buffer, to: RUL_AXIS_MAX, color: 'var(--surface-2)', label: 'health state decides' },
              ]}
            >
              <ScaleMarker value={rul} label={fmt(d.predictedRul)} />
            </RulScale>
            <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2">
              <span className="inline-flex items-center gap-1.5"><Swatch color={actionColor('INTERVENE_NOW')} />≤ {lead}: intervene now</span>
              <span className="inline-flex items-center gap-1.5"><Swatch color={actionColor('SCHEDULE_MAINTENANCE')} />≤ {lead + buffer}: schedule now</span>
              <span className="inline-flex items-center gap-1.5"><Swatch color="var(--line-strong)" />otherwise the health state decides</span>
            </div>
          </SubStep>

          <SubStep n={3} title="Rules applied">
            <ul className="space-y-2">
              {d.trace.map((step) => (
                <li key={step.ruleId} className="flex items-start gap-2.5 text-[13px] leading-relaxed">
                  <span className="mt-[3px] flex shrink-0 items-center gap-1.5">
                    {step.actionBefore && (<><ActionBadge action={step.actionBefore} /><span className="text-ink-3" aria-hidden>→</span></>)}
                    <ActionBadge action={step.actionAfter} />
                  </span>
                  <span className="text-ink-2">
                    {expert && <span className="mr-1.5 font-mono text-xs font-semibold text-ink">{step.ruleId}</span>}
                    {step.reason}
                  </span>
                </li>
              ))}
            </ul>
          </SubStep>
        </ol>

        <div className="min-w-0">
          <div className="rounded-xl bg-surface-2 p-5 lg:sticky lg:top-40">
            <div className="eyebrow mb-3">Recommended action</div>
            <div className="flex flex-wrap items-center gap-2">
              <ActionBadge action={d.recommendedAction} size="lg" />
              {d.requiresHumanReview && <ReviewFlag />}
            </div>
            <p className="mt-3 text-[15px] leading-relaxed">{index.action_meanings[d.recommendedAction]}</p>
            {severity(d.recommendedAction) < severity(d.baseAction) && (
              <p className="mt-2 text-xs text-ink-3">Stepped down from the health-state baseline because confidence is low: inspect before intervening.</p>
            )}
            {d.requiresHumanReview && (
              <p className="mt-2 text-xs text-ink-3">High-urgency actions go to a person for sign-off; the system recommends, it doesn't act.</p>
            )}
          </div>
        </div>
      </div>

      <Expert>
        <ExpertHistory engine={engine} i={i} setI={setI} decisions={decisions} params={params} />
      </Expert>
    </Stage>
  )
}

function SubStep({ n, title, children }: { n: number; title: string; children: ReactNode }) {
  return (
    <li>
      <div className="mb-2 flex items-center gap-2">
        <span className="tabular flex h-5 w-5 items-center justify-center rounded-full bg-ink text-[11px] font-semibold text-page">{n}</span>
        <h3 className="text-sm font-semibold">{title}</h3>
      </div>
      <div className="space-y-1 pl-7">{children}</div>
    </li>
  )
}

function ExpertHistory({ engine, i, setI, decisions, params }: Omit<Props, 'index'>) {
  const actualStates = engine.actual_rul_capped.map((r) => assignHealthState(r, params.thresholds))
  const agree = decisions.filter((dec, k) => dec.healthState === actualStates[k]).length
  const firstReview = decisions.findIndex((dec) => dec.requiresHumanReview)
  const firstIntervene = decisions.findIndex((dec) => dec.recommendedAction === 'INTERVENE_NOW')

  return (
    <div>
      <PanelLabel right={<><span className="tabular font-semibold text-ink">{Math.round((agree / decisions.length) * 100)}%</span> of cycles agree</>}>
        Predicted vs true health state over the engine's life
      </PanelLabel>
      <div className="space-y-2">
        <LifeStrip
          label="Predicted"
          colors={decisions.map((dec) => STATE_COLOR[dec.healthState])}
          cursor={i}
          onSelect={setI}
          tooltip={(k) => <>Cycle {engine.cycle[k]} · <strong>{stateLabel(decisions[k].healthState)}</strong> (RUL {fmt(decisions[k].predictedRul)})</>}
        />
        <LifeStrip
          label="True"
          colors={actualStates.map((s) => STATE_COLOR[s])}
          cursor={i}
          onSelect={setI}
          tooltip={(k) => <>Cycle {engine.cycle[k]} · <strong>{stateLabel(actualStates[k])}</strong> (true RUL {engine.actual_rul_capped[k]})</>}
        />
      </div>
      <div className="mt-3 pl-[76px]"><StateLegend /></div>
      <div className="mt-5 grid gap-2 sm:grid-cols-2">
        <Milestone label="First human review" k={firstReview} engine={engine} onJump={setI} />
        <Milestone label="First intervene now" k={firstIntervene} engine={engine} onJump={setI} />
      </div>
    </div>
  )
}

function Milestone({ label, k, engine, onJump }: { label: string; k: number; engine: EngineData; onJump: (i: number) => void }) {
  if (k < 0) {
    return (
      <div className="rounded-lg px-3 py-2.5 text-xs" style={{ boxShadow: 'inset 0 0 0 1px var(--line)' }}>
        <div className="text-ink-3">{label}</div>
        <div className="mt-0.5 font-semibold">Never reached</div>
      </div>
    )
  }
  return (
    <button onClick={() => onJump(k)} className="group rounded-lg px-3 py-2.5 text-left text-xs hover:bg-surface-2" style={{ boxShadow: 'inset 0 0 0 1px var(--line)' }}>
      <div className="flex items-center justify-between text-ink-3">
        {label}
        <span className="opacity-0 transition-opacity group-hover:opacity-100" aria-hidden>Jump →</span>
      </div>
      <div className="mt-0.5 font-semibold">
        Cycle <span className="tabular">{engine.cycle[k]}</span>
        <span className="font-normal text-ink-2"> · {engine.actual_rul[k]} cycles before failure</span>
      </div>
    </button>
  )
}
