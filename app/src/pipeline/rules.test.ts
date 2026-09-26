import { describe, expect, it } from 'vitest'
import parity from './parity_cases.json'
import { assignHealthState, decide, thresholdError, type Confidence } from './rules'

const DEFAULTS = { action: 15, plan: 30, watch: 60 }

describe('Python parity (parity_cases.json, from src/pipeline_explorer.py)', () => {
  it('has cases', () => expect(parity.cases.length).toBeGreaterThan(500))

  it('reproduces every Python decision exactly', () => {
    const mismatches = parity.cases.flatMap(({ input, expected }) => {
      const d = decide(input.predicted_rul, input.model_confidence as Confidence, {
        thresholds: input.health_thresholds,
        leadTimeCycles: input.maintenance_lead_time_cycles,
        leadTimeBufferCycles: parity.lead_time_buffer_cycles,
      })
      const actual = {
        health_state: d.healthState,
        base_action: d.baseAction,
        recommended_action: d.recommendedAction,
        rule_ids: d.ruleIds,
        requires_human_review: d.requiresHumanReview,
      }
      return JSON.stringify(actual) === JSON.stringify(expected) ? [] : [{ input, expected, actual }]
    })
    expect(mismatches).toEqual([])
  })
})

describe('health state', () => {
  it('uses inclusive upper bounds', () => {
    expect(assignHealthState(15, DEFAULTS)).toBe('ACTION')
    expect(assignHealthState(15.01, DEFAULTS)).toBe('PLAN')
    expect(assignHealthState(60, DEFAULTS)).toBe('WATCH')
    expect(assignHealthState(60.01, DEFAULTS)).toBe('HEALTHY')
  })
  it('clamps negative RUL into ACTION', () => expect(assignHealthState(-4, DEFAULTS)).toBe('ACTION'))
  it('rejects NaN', () => expect(() => assignHealthState(NaN, DEFAULTS)).toThrow())
  it('rejects unordered thresholds', () => {
    expect(thresholdError({ action: 30, plan: 30, watch: 60 })).not.toBeNull()
    expect(() => assignHealthState(10, { action: 40, plan: 30, watch: 60 })).toThrow()
  })
})

describe('decision', () => {
  const params = { thresholds: DEFAULTS, leadTimeCycles: 15, leadTimeBufferCycles: 5 }
  it('never gives less than INSPECT for PLAN/ACTION', () => {
    for (let rul = 0; rul <= 30; rul += 0.5) {
      for (const c of ['HIGH', 'MEDIUM', 'LOW'] as const) {
        expect(['INSPECT', 'PLAN_MAINTENANCE', 'SCHEDULE_MAINTENANCE', 'INTERVENE_NOW'])
          .toContain(decide(rul, c, params).recommendedAction)
      }
    }
  })
  it('longer lead time escalates earlier', () => {
    expect(decide(40, 'HIGH', params).recommendedAction).toBe('MONITOR')
    expect(decide(40, 'HIGH', { ...params, leadTimeCycles: 40 }).recommendedAction).toBe('INTERVENE_NOW')
  })
})
