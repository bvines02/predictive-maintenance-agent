/**
 * Deterministic health-state and maintenance-decision rules - a TypeScript
 * port of src/health_state.py + src/decision_engine.py, restricted to the
 * Pipeline Explorer context in src/pipeline_explorer.py (criticality and
 * redundancy excluded, so only BASE_01, LEAD_01, LEAD_02 and CONF_01 exist).
 *
 * Ported, not reinvented: rules.test.ts replays every case in
 * parity_cases.json (generated from the real Python engine) and fails on
 * any disagreement. Change the Python first, regenerate, then this file.
 */

export const HEALTH_STATES = ['HEALTHY', 'WATCH', 'PLAN', 'ACTION'] as const
export type HealthState = (typeof HEALTH_STATES)[number]

// Ordered by urgency: the index IS the severity (MaintenanceAction IntEnum).
export const ACTIONS = [
  'NO_ACTION',
  'MONITOR',
  'INSPECT',
  'PLAN_MAINTENANCE',
  'SCHEDULE_MAINTENANCE',
  'INTERVENE_NOW',
] as const
export type Action = (typeof ACTIONS)[number]

export type Confidence = 'HIGH' | 'MEDIUM' | 'LOW'

export interface HealthThresholds {
  action: number
  plan: number
  watch: number
}

export interface RuleStep {
  ruleId: 'BASE_01' | 'LEAD_01' | 'LEAD_02' | 'CONF_01'
  kind: 'base' | 'escalate' | 'de_escalate'
  actionBefore: Action | null
  actionAfter: Action
  reason: string
}

export interface Decision {
  predictedRul: number
  healthState: HealthState
  baseAction: Action
  recommendedAction: Action
  requiresHumanReview: boolean
  ruleIds: RuleStep['ruleId'][]
  trace: RuleStep[]
}

export interface DecisionParams {
  thresholds: HealthThresholds
  leadTimeCycles: number
  leadTimeBufferCycles: number
}

const BASE_ACTIONS: Record<HealthState, Action> = {
  HEALTHY: 'NO_ACTION',
  WATCH: 'MONITOR',
  PLAN: 'PLAN_MAINTENANCE',
  ACTION: 'SCHEDULE_MAINTENANCE',
}

export const severity = (action: Action): number => ACTIONS.indexOf(action)

/** Python's f"{x:g}" for the magnitudes that occur here. */
const g = (x: number): string => String(Number(x.toPrecision(6)))

/** Returns an error message, or null if valid. Mirrors validate_thresholds(). */
export function thresholdError(t: HealthThresholds): string | null {
  if (![t.action, t.plan, t.watch].every(Number.isFinite)) return 'Thresholds must be numbers.'
  if (!(t.action < t.plan && t.plan < t.watch)) {
    return `Thresholds must satisfy action < plan < watch (got ${g(t.action)}, ${g(t.plan)}, ${g(t.watch)}).`
  }
  return null
}

/** ACTION: rul <= action, PLAN: <= plan, WATCH: <= watch, else HEALTHY. Negative RUL clamps to 0. */
export function assignHealthState(rul: number, t: HealthThresholds): HealthState {
  const error = thresholdError(t)
  if (error) throw new Error(error)
  if (Number.isNaN(rul)) throw new Error('Cannot assign a health state to a NaN RUL.')
  const r = Math.max(rul, 0)
  if (r <= t.action) return 'ACTION'
  if (r <= t.plan) return 'PLAN'
  if (r <= t.watch) return 'WATCH'
  return 'HEALTHY'
}

export function decide(predictedRul: number, confidence: Confidence, p: DecisionParams): Decision {
  const healthState = assignHealthState(predictedRul, p.thresholds)
  const baseAction = BASE_ACTIONS[healthState]
  let action: Action = baseAction
  const trace: RuleStep[] = [
    { ruleId: 'BASE_01', kind: 'base', actionBefore: null, actionAfter: baseAction,
      reason: `Health state ${healthState} maps to ${baseAction}.` },
  ]
  const escalateTo = (floor: Action, ruleId: RuleStep['ruleId'], reason: string) => {
    if (severity(floor) > severity(action)) {
      trace.push({ ruleId, kind: 'escalate', actionBefore: action, actionAfter: floor, reason })
      action = floor
    }
  }

  // LEAD_01 / LEAD_02: the decision must precede the time needed to do the work.
  const rul = Math.max(predictedRul, 0)
  const lead = p.leadTimeCycles
  const buffer = p.leadTimeBufferCycles
  if (rul <= lead) {
    escalateTo('INTERVENE_NOW', 'LEAD_01',
      `Predicted RUL ${g(rul)} <= maintenance lead time ${g(lead)}: work started now may not finish before predicted failure.`)
  } else if (rul <= lead + buffer) {
    escalateTo('SCHEDULE_MAINTENANCE', 'LEAD_02',
      `Predicted RUL ${g(rul)} is within ${g(buffer)} cycles of maintenance lead time ${g(lead)}: commit the work now to keep a margin.`)
  }

  // CONF_01: low confidence means "seek confirming evidence", not "ignore".
  // (CONF_02's override needs HIGH criticality, which the explorer excludes.)
  if (confidence === 'LOW' && (healthState === 'PLAN' || healthState === 'ACTION') && severity(action) > severity('INSPECT')) {
    trace.push({ ruleId: 'CONF_01', kind: 'de_escalate', actionBefore: action, actionAfter: 'INSPECT',
      reason: 'Confidence is LOW: seek confirming evidence (inspection) before an intrusive action.' })
    action = 'INSPECT'
  }

  return {
    predictedRul,
    healthState,
    baseAction,
    recommendedAction: action,
    requiresHumanReview: severity(action) >= severity('SCHEDULE_MAINTENANCE'),
    ruleIds: trace.map((s) => s.ruleId),
    trace,
  }
}
