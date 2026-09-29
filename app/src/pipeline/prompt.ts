/**
 * TypeScript port of build_rul_explanation_prompt() in src/explainer.py, used
 * by the Vercel function (api/explain.ts). prompt.test.ts rebuilds every case
 * in prompt_cases.json (generated from Python) and requires byte equality.
 */
import type { Decision, HealthThresholds } from './rules.js'

export interface PromptPrediction {
  unit_number: number
  time_cycles: number
  prediction_p10: number
  prediction_p90: number
}

// Mirrors ACTION_MEANINGS and DecisionConfig.lead_time_buffer_cycles in
// src/decision_engine.py; prompt.test.ts checks both against the Python fixture.
export const ACTION_MEANINGS: Record<string, string> = {
  NO_ACTION: 'Continue normal operation.',
  MONITOR: 'Increase observation or review the trend at normal planning cadence.',
  INSPECT: 'Perform targeted inspection or diagnostic verification.',
  PLAN_MAINTENANCE: 'Start defining scope, spares and execution requirements.',
  SCHEDULE_MAINTENANCE: 'Commit the intervention into an executable maintenance window.',
  INTERVENE_NOW: 'Immediate operational / maintenance response is required.',
}
export const LEAD_TIME_BUFFER_CYCLES = 5

/** Python's f"{x:.1f}": exact binary value, ties (only .x25/.x75 are exact in binary) round half to even. */
export function pyFixed1(x: number): string {
  const tie = Number.isInteger(x * 4) && !Number.isInteger(x * 2)
  if (!tie) return x.toFixed(1)
  const down = Math.floor(x * 10) / 10
  const up = Math.ceil(x * 10) / 10
  const evenDigit = (v: number) => Math.round(Math.abs(v) * 10) % 2 === 0
  return (evenDigit(down) ? down : up).toFixed(1)
}

/** Python's f"{x:g}" for the magnitudes that occur here. */
export const pyG = (x: number): string => String(Number(x.toPrecision(6)))

export function buildRulPrompt(
  decision: Decision,
  confidence: string,
  leadTimeCycles: number,
  prediction: PromptPrediction,
  t: HealthThresholds,
  leadTimeBufferCycles: number,
  actionMeanings: Record<string, string>,
): string {
  const trace = decision.trace.map((s) => `- ${s.ruleId}: ${s.reason}`).join('\n')
  return `You are explaining a turbofan engine maintenance decision to a university
student in engineering or computer science. They are numerate and know basic
statistics, but are new to machine learning and maintenance engineering. The
decision has ALREADY been made by deterministic rules - do not change it,
second-guess it, or suggest a different action. Only explain it.

Engine ${prediction.unit_number}, flight cycle ${prediction.time_cycles} (NASA C-MAPSS FD001, simulated data)

Model prediction (Random Forest: 200 regression trees, each fitted to a bootstrap
resample of the training rows and considering every feature at each split; the
point estimate is the mean of the tree outputs):
Predicted remaining useful life: ${pyFixed1(decision.predictedRul)} cycles
Middle 80% of individual tree predictions: ${pyFixed1(prediction.prediction_p10)} to ${pyFixed1(prediction.prediction_p90)} cycles
Model confidence (from how closely the trees agree, not a calibrated probability): ${confidence}

Health state: ${decision.healthState}
(ACTION if RUL <= ${pyG(t.action)}, PLAN if <= ${pyG(t.plan)}, WATCH if <= ${pyG(t.watch)}, otherwise HEALTHY)

Maintenance lead time: ${pyG(leadTimeCycles)} cycles (planning buffer ${pyG(leadTimeBufferCycles)} cycles)

Decision (already made, do not change):
Recommended action: ${decision.recommendedAction} - ${actionMeanings[decision.recommendedAction]}
Human review required: ${decision.requiresHumanReview ? 'True' : 'False'}
Rules applied, in order:
${trace}

Write one paragraph of 150-200 words, in a precise academic register,
explaining why this prediction led to this action. Cover: how the Random
Forest's ensemble of trees produces the point estimate and the spread of
estimates; how the deterministic thresholds map the estimate to the health
state; what each rule that fired changed and why; and what the model
confidence implies about the reliability of the estimate, noting that it
reflects agreement between trees rather than a calibrated probability. Use
correct technical terms, defining each briefly on first use. Do not invent
numbers or facts not given above - including details of how the model was
trained - and do not discuss asset criticality or redundancy - they are
deliberately not part of this decision.`
}
