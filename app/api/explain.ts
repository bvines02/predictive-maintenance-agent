/**
 * Vercel Function: POST /api/explain - the deployed equivalent of POST
 * /explain in src/api.py, with the same request and response shapes.
 *
 * Same guarantee as the Python endpoint: it takes the prediction and the
 * parameters, RE-DERIVES the decision itself (with the rules port that
 * app/src/pipeline/rules.test.ts checks against the Python engine), and only
 * then asks the LLM to explain it. The browser cannot hand it a decision.
 * ANTHROPIC_API_KEY lives in the Vercel project's environment variables.
 *
 * Imports use .js extensions: Vercel compiles each file to ESM, where Node
 * needs them. TypeScript maps them back to the .ts sources.
 */
import { ACTION_MEANINGS, LEAD_TIME_BUFFER_CYCLES, buildRulPrompt } from '../src/pipeline/prompt.js'
import { decide, thresholdError, type Confidence, type HealthThresholds } from '../src/pipeline/rules.js'

// Same model and length as generate_rul_explanation() in src/explainer.py.
export const MODEL = 'claude-sonnet-5'
const MAX_TOKENS = 400

const json = (body: unknown, status = 200) => Response.json(body, { status })
const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)

interface Body {
  unit_number: number
  time_cycles: number
  predicted_rul: number
  prediction_p10: number
  prediction_p90: number
  model_confidence: Confidence
  health_thresholds: HealthThresholds
  maintenance_lead_time_cycles: number
}

function validate(b: Partial<Body> | null): string | null {
  if (!b || typeof b !== 'object') return 'Body must be a JSON object.'
  for (const k of ['unit_number', 'time_cycles', 'predicted_rul', 'prediction_p10', 'prediction_p90', 'maintenance_lead_time_cycles'] as const) {
    if (!isNum(b[k])) return `${k} must be a number.`
  }
  if (!Number.isInteger(b.unit_number) || !Number.isInteger(b.time_cycles)) return 'unit_number and time_cycles must be integers.'
  if (b.maintenance_lead_time_cycles! < 0 || b.maintenance_lead_time_cycles! > 1000) return 'maintenance_lead_time_cycles must be between 0 and 1000.'
  if (!['HIGH', 'MEDIUM', 'LOW'].includes(b.model_confidence as string)) return 'model_confidence must be HIGH, MEDIUM or LOW.'
  const t = b.health_thresholds
  if (!t || !isNum(t.action) || !isNum(t.plan) || !isNum(t.watch)) return 'health_thresholds needs numeric action, plan and watch.'
  return thresholdError(t)
}

export async function POST(request: Request): Promise<Response> {
  const body = (await request.json().catch(() => null)) as Partial<Body> | null
  const problem = validate(body)
  if (problem) return json({ detail: problem }, 422)
  const b = body as Body

  const decision = decide(b.predicted_rul, b.model_confidence, {
    thresholds: b.health_thresholds,
    leadTimeCycles: b.maintenance_lead_time_cycles,
    leadTimeBufferCycles: LEAD_TIME_BUFFER_CYCLES,
  })
  const prompt = buildRulPrompt(decision, b.model_confidence, b.maintenance_lead_time_cycles, b,
    b.health_thresholds, LEAD_TIME_BUFFER_CYCLES, ACTION_MEANINGS)
  const summary = {
    health_state: decision.healthState,
    base_action: decision.baseAction,
    recommended_action: decision.recommendedAction,
    rule_ids: decision.ruleIds,
    requires_human_review: decision.requiresHumanReview,
  }

  const key = process.env.ANTHROPIC_API_KEY
  if (!key) return json({ decision: summary, prompt, explanation: null })

  const upstream = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: { 'x-api-key': key, 'anthropic-version': '2023-06-01', 'content-type': 'application/json' },
    body: JSON.stringify({ model: MODEL, max_tokens: MAX_TOKENS, messages: [{ role: 'user', content: prompt }] }),
  })
  if (!upstream.ok) {
    return json({ detail: `LLM request failed (HTTP ${upstream.status}).` }, 502)
  }
  const message = (await upstream.json()) as { content?: { type: string; text?: string }[] }
  const explanation = message.content?.find((c) => c.type === 'text')?.text ?? null
  return json({ decision: summary, prompt, explanation })
}
