import { describe, expect, it } from 'vitest'
import cases from './prompt_cases.json'
import { ACTION_MEANINGS, LEAD_TIME_BUFFER_CYCLES, buildRulPrompt, pyFixed1 } from './prompt'
import { decide, type Confidence } from './rules'

describe('prompt parity (prompt_cases.json, from src/explainer.py)', () => {
  it.each(cases.cases.map((c) => [`engine ${c.input.unit_number}`, c] as const))('%s', (_, c) => {
    const i = c.input
    const decision = decide(i.predicted_rul, i.model_confidence as Confidence, {
      thresholds: i.health_thresholds,
      leadTimeCycles: i.maintenance_lead_time_cycles,
      leadTimeBufferCycles: cases.lead_time_buffer_cycles,
    })
    const prompt = buildRulPrompt(decision, i.model_confidence, i.maintenance_lead_time_cycles, i,
      i.health_thresholds, cases.lead_time_buffer_cycles, cases.action_meanings)
    expect(prompt).toBe(c.prompt)
  })
})

describe('pyFixed1', () => {
  it('rounds exact ties half to even like Python', () => {
    expect(pyFixed1(35.25)).toBe('35.2')
    expect(pyFixed1(12.75)).toBe('12.8')
    expect(pyFixed1(0.25)).toBe('0.2')
    expect(pyFixed1(-2.5)).toBe('-2.5')
    expect(pyFixed1(30.05)).toBe('30.1') // not a true tie in binary
  })
})

describe('constants mirrored from Python', () => {
  it('action meanings match', () => expect(ACTION_MEANINGS).toEqual(cases.action_meanings))
  it('lead-time buffer matches', () => expect(LEAD_TIME_BUFFER_CYCLES).toBe(cases.lead_time_buffer_cycles))
})
