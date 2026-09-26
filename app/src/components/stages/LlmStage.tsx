import { useState } from 'react'
import type { EngineData } from '../../pipeline/data'
import { BackendUnavailable, requestExplanation, type ExplainRequest, type ExplainResponse } from '../../pipeline/explain'
import type { Decision, DecisionParams } from '../../pipeline/rules'
import { ActionBadge, Stage } from '../ui'

interface Props {
  engine: EngineData
  i: number
  decision: Decision
  params: DecisionParams
}

type State =
  | { status: 'idle' }
  | { status: 'loading'; key: string }
  | { status: 'done'; key: string; request: ExplainRequest; response: ExplainResponse }
  | { status: 'offline'; key: string; message: string }
  | { status: 'error'; key: string; message: string }

export function LlmStage({ engine, i, decision, params }: Props) {
  const [state, setState] = useState<State>({ status: 'idle' })

  const request: ExplainRequest = {
    unit_number: engine.unit,
    time_cycles: engine.cycle[i],
    predicted_rul: engine.predicted_rul[i],
    prediction_p10: engine.tree_quantiles.p10[i],
    prediction_p90: engine.tree_quantiles.p90[i],
    model_confidence: engine.model_confidence[i],
    health_thresholds: params.thresholds,
    maintenance_lead_time_cycles: params.leadTimeCycles,
  }
  const key = JSON.stringify(request)
  const stale = state.status !== 'idle' && state.key !== key

  async function run() {
    setState({ status: 'loading', key })
    try {
      const response = await requestExplanation(request)
      setState({ status: 'done', key, request, response })
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      setState(err instanceof BackendUnavailable ? { status: 'offline', key, message } : { status: 'error', key, message })
    }
  }

  return (
    <Stage
      id="stage-llm"
      step={7}
      kind="llm"
      title="LLM explains the decision — and cannot change it"
      summary={
        <>
          Only now does the LLM get involved. It receives the finished decision, the rule trace and the prediction summary, and
          writes a plain-language explanation for the reviewer. The API re-runs the decision in the Python engine first, so the
          LLM only ever explains what deterministic code decided. Remove this step and every decision above is unchanged.
        </>
      }
    >
      <div className="grid gap-5 lg:grid-cols-2">
        <div className="min-w-0 rounded-lg border border-line p-4">
          <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-ink-3">Sent to the LLM</div>
          <ul className="space-y-1 text-sm">
            <li>Engine {engine.unit}, cycle {engine.cycle[i]}</li>
            <li>Predicted RUL and the p10–p90 tree range</li>
            <li>Model confidence ({engine.model_confidence[i]})</li>
            <li>Health state {decision.healthState} and the thresholds in force</li>
            <li>Lead time {params.leadTimeCycles} + buffer {params.leadTimeBufferCycles}</li>
            <li className="flex flex-wrap items-center gap-1.5">Final action <ActionBadge action={decision.recommendedAction} /> and the rule trace ({decision.ruleIds.join(' → ')})</li>
          </ul>
          <div className="mb-2 mt-4 text-xs font-semibold uppercase tracking-wide text-ink-3">Never sent</div>
          <ul className="space-y-1 text-sm text-ink-2">
            <li>Raw sensor readings or engineered features</li>
            <li>Any way to change the action, the state or the review flag</li>
            <li>Asset criticality or redundancy (excluded from this view)</li>
          </ul>
        </div>

        <div className="min-w-0 rounded-lg border p-4" style={{ borderColor: 'var(--accent)', background: 'var(--accent-soft)' }}>
          <div className="mb-3 flex flex-wrap items-center gap-3">
            <button
              onClick={run}
              disabled={state.status === 'loading'}
              className="rounded-md bg-accent px-3 py-1.5 text-sm font-semibold text-page disabled:opacity-60"
            >
              {state.status === 'loading' ? 'Asking the LLM…' : state.status === 'idle' || stale ? 'Explain this decision' : 'Regenerate'}
            </button>
            <span className="text-xs text-ink-3">Live call via POST /explain</span>
          </div>

          {stale && (
            <p className="mb-2 text-xs font-medium text-ink-2">Out of date: the cycle or parameters changed since this was generated.</p>
          )}

          {state.status === 'idle' && (
            <p className="text-sm text-ink-2">Generate an explanation for the decision at cycle {engine.cycle[i]}.</p>
          )}

          {state.status === 'offline' && (
            <div className="text-sm text-ink-2">
              <p className="mb-2">The API isn't running, so there is no LLM step. Everything above still works: it doesn't depend on the LLM.</p>
              <p className="mb-1 text-xs">From the repo root:</p>
              <pre className="overflow-x-auto rounded bg-surface px-2 py-1.5 font-mono text-xs">uvicorn src.api:app --port 8000</pre>
            </div>
          )}

          {state.status === 'error' && <p className="text-sm" style={{ color: 'var(--critical)' }}>API error: {state.message}</p>}

          {state.status === 'done' && <Result request={state.request} response={state.response} browserDecision={stale ? null : decision} />}
        </div>
      </div>
    </Stage>
  )
}

function Result({ request, response, browserDecision }: { request: ExplainRequest; response: ExplainResponse; browserDecision: Decision | null }) {
  const py = response.decision
  const agrees =
    browserDecision &&
    py.recommended_action === browserDecision.recommendedAction &&
    py.health_state === browserDecision.healthState &&
    JSON.stringify(py.rule_ids) === JSON.stringify(browserDecision.ruleIds)

  return (
    <div className="space-y-3">
      {browserDecision && (
        <div className="text-xs font-medium" style={{ color: agrees ? 'var(--ink-2)' : 'var(--critical)' }}>
          {agrees
            ? `✓ Python engine agrees with the browser: ${py.recommended_action} via ${py.rule_ids.join(' → ')}`
            : `✗ Python engine disagrees: ${py.recommended_action} via ${py.rule_ids.join(' → ')}. The TypeScript rules have drifted.`}
        </div>
      )}
      {response.explanation ? (
        <blockquote className="whitespace-pre-wrap border-l-2 pl-3 text-sm leading-relaxed" style={{ borderColor: 'var(--accent)' }}>
          {response.explanation}
        </blockquote>
      ) : (
        <p className="text-sm text-ink-2">
          No <code className="font-mono text-xs">ANTHROPIC_API_KEY</code> on the API server, so the LLM wasn't called. The prompt it
          would receive is below — add the key to <code className="font-mono text-xs">.env</code> and restart the API.
        </p>
      )}
      <details className="text-xs">
        <summary className="cursor-pointer font-medium text-ink-2">Exact prompt (engine {request.unit_number}, cycle {request.time_cycles})</summary>
        <pre className="mt-2 max-h-80 overflow-auto whitespace-pre-wrap rounded bg-surface p-2 font-mono text-[11px] leading-relaxed">{response.prompt}</pre>
      </details>
    </div>
  )
}
