/** Client for POST /explain (src/api.py) - the only call to a backend. */
import type { Confidence, HealthThresholds } from './rules'

export interface ExplainRequest {
  unit_number: number
  time_cycles: number
  predicted_rul: number
  prediction_p10: number
  prediction_p90: number
  model_confidence: Confidence
  health_thresholds: HealthThresholds
  maintenance_lead_time_cycles: number
}

export interface ExplainResponse {
  decision: {
    health_state: string
    base_action: string
    recommended_action: string
    rule_ids: string[]
    requires_human_review: boolean
  }
  prompt: string
  explanation: string | null
}

export class BackendUnavailable extends Error {}

export async function requestExplanation(body: ExplainRequest): Promise<ExplainResponse> {
  let response: Response
  try {
    response = await fetch('./api/explain', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
  } catch {
    throw new BackendUnavailable('Could not reach the API.')
  }
  // Vite's proxy answers 500/502/504 when uvicorn isn't running.
  if ([404, 500, 502, 503, 504].includes(response.status) && !response.headers.get('content-type')?.includes('json')) {
    throw new BackendUnavailable(`API not reachable (HTTP ${response.status}).`)
  }
  if (!response.ok) {
    const detail = await response.json().catch(() => null)
    throw new Error(detail?.detail ? JSON.stringify(detail.detail) : `HTTP ${response.status}`)
  }
  return response.json()
}
