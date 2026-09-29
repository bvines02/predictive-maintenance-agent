import { useState, type ReactNode } from 'react'
import type { EngineData } from '../../pipeline/data'
import { BackendUnavailable, requestExplanation, type ExplainRequest, type ExplainResponse } from '../../pipeline/explain'
import type { Decision, DecisionParams } from '../../pipeline/rules'
import { ActionBadge, Expert, PanelLabel, Stage, stateLabel, useExpert } from '../ui'

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
      step={4}
      kind="llm"
      title="AI explains the decision"
      takeaway={
        <>Only now does a language model get involved. It receives the finished decision and writes it up in plain English for the person
        reviewing it. It never sees the raw sensor data and has no way to change the decision: delete this step and every action above
        stays the same.</>
      }
      method={
        <p>
          The large language model (LLM) is positioned at the end of the pipeline, downstream of every decision. It receives the completed
          decision, the rule trace and a summary of the prediction, and generates a natural-language explanation for the human reviewer. The
          server re-derives the decision itself before calling the LLM, so a client cannot supply a decision for it to endorse. This is a
          deliberate architectural constraint: LLM output is non-deterministic and can contain confabulated content, so it is confined to a
          presentational role with no path back into the decision. Removing this stage leaves every action above unchanged; the LLM improves
          interpretability, not the decision.
        </p>
      }
    >
      <div className="rounded-xl p-5" style={{ background: 'var(--accent-soft)', boxShadow: 'inset 0 0 0 1px color-mix(in srgb, var(--accent) 25%, transparent)' }}>
        <div className="mb-4 flex flex-wrap items-center gap-3">
          <button
            onClick={run}
            disabled={state.status === 'loading'}
            className="inline-flex h-8 items-center gap-2 rounded-lg px-3.5 text-sm font-semibold transition-opacity disabled:opacity-60"
            style={{ background: 'var(--kind-llm)', color: 'var(--surface)' }}
          >
            {state.status === 'loading' && <Spinner />}
            {state.status === 'loading' ? 'Generating…' : state.status === 'idle' || stale ? 'Generate explanation' : 'Regenerate'}
          </button>
          <Expert bare><span className="text-xs text-ink-3">Live call to <code className="font-mono">POST /api/explain</code></span></Expert>
        </div>

        {stale && (
          <p className="mb-3 inline-flex items-center gap-1.5 rounded-md bg-surface px-2 py-1 text-xs font-medium text-ink-2">
            <span className="h-1.5 w-1.5 rounded-full" style={{ background: 'var(--warning)' }} />
            Out of date — the cycle or policy has changed since this was generated.
          </p>
        )}

        {state.status === 'idle' && (
          <p className="text-sm leading-relaxed text-ink-2">
            Ask the AI to explain the decision at cycle {engine.cycle[i]}. The server re-checks the decision itself before the model sees it.
          </p>
        )}

        {state.status === 'loading' && (
          <div className="max-w-3xl space-y-2" aria-hidden>
            {[92, 100, 84, 96, 60].map((w, k) => (
              <div key={k} className="h-3 animate-pulse rounded bg-surface" style={{ width: `${w}%` }} />
            ))}
          </div>
        )}

        {state.status === 'offline' && (
          <div className="text-sm text-ink-2">
            <p className="mb-2">The explanation service isn't reachable from this page. Everything above still works: none of it depends on the AI.</p>
            <p className="mb-1 text-xs">Deployed, it is the <code className="font-mono">api/explain</code> function. Locally, start the API from the repo root:</p>
            <pre className="overflow-x-auto rounded-md bg-surface px-2.5 py-2 font-mono text-xs">uvicorn src.api:app --port 8000</pre>
          </div>
        )}

        {state.status === 'error' && <p className="text-sm" style={{ color: 'var(--critical)' }}>Request failed: {state.message}</p>}

        {state.status === 'done' && <Result request={state.request} response={state.response} browserDecision={stale ? null : decision} />}
      </div>

      <Expert>
        <div className="grid gap-8 sm:grid-cols-2">
          <div>
            <PanelLabel>Provided to the model</PanelLabel>
            <ul className="space-y-1.5 text-sm">
              <Li mark="in">Engine {engine.unit}, cycle {engine.cycle[i]}</Li>
              <Li mark="in">Predicted RUL and the 10th–90th percentile tree range</Li>
              <Li mark="in">Model confidence ({engine.model_confidence[i].toLowerCase()})</Li>
              <Li mark="in">Health state ({stateLabel(decision.healthState)}) and thresholds in force</Li>
              <Li mark="in">Lead time {params.leadTimeCycles} + {params.leadTimeBufferCycles}-cycle buffer</Li>
              <Li mark="in">
                <span className="inline-flex flex-wrap items-center gap-1.5">
                  Final action <ActionBadge action={decision.recommendedAction} /> and trace
                  <span className="font-mono text-xs text-ink-2">{decision.ruleIds.join(' → ')}</span>
                </span>
              </Li>
            </ul>
          </div>
          <div>
            <PanelLabel>Withheld</PanelLabel>
            <ul className="space-y-1.5 text-sm text-ink-2">
              <Li mark="out">Raw sensor readings and engineered features</Li>
              <Li mark="out">Any means to alter the action, state or review flag</Li>
              <Li mark="out">Asset criticality and redundancy</Li>
            </ul>
          </div>
        </div>
      </Expert>
    </Stage>
  )
}

function Li({ mark, children }: { mark: 'in' | 'out'; children: ReactNode }) {
  return (
    <li className="flex gap-2.5">
      <span className="mt-[3px] flex h-4 w-4 shrink-0 items-center justify-center rounded-full" style={{ background: mark === 'in' ? 'var(--surface-2)' : 'transparent', boxShadow: mark === 'out' ? 'inset 0 0 0 1px var(--line-strong)' : undefined }} aria-hidden>
        <svg width="9" height="9" viewBox="0 0 10 10" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" className={mark === 'in' ? 'text-ink' : 'text-ink-3'}>
          {mark === 'in' ? <path d="M2 5.2l2 2 4-4.4" /> : <path d="M2.5 5h5" />}
        </svg>
      </span>
      <span className="min-w-0">{children}</span>
    </li>
  )
}

function Result({ request, response, browserDecision }: { request: ExplainRequest; response: ExplainResponse; browserDecision: Decision | null }) {
  const [showPrompt, setShowPrompt] = useState(false)
  const expert = useExpert()
  const py = response.decision
  const agrees =
    browserDecision &&
    py.recommended_action === browserDecision.recommendedAction &&
    py.health_state === browserDecision.healthState &&
    JSON.stringify(py.rule_ids) === JSON.stringify(browserDecision.ruleIds)

  return (
    <div className="space-y-4">
      {response.explanation ? (
        <p className="max-w-3xl whitespace-pre-wrap text-[15px] leading-[1.7] text-ink">{response.explanation}</p>
      ) : (
        <p className="text-sm leading-relaxed text-ink-2">
          No <code className="font-mono text-xs">ANTHROPIC_API_KEY</code> is configured on the server, so the model wasn't called. The
          prompt it would receive is below.
        </p>
      )}
      <div className="flex flex-wrap items-center justify-between gap-2 border-t pt-3 text-xs" style={{ borderColor: 'color-mix(in srgb, var(--accent) 20%, transparent)' }}>
        {expert && browserDecision ? (
          <span className="inline-flex items-center gap-1.5 font-medium" style={{ color: agrees ? 'var(--ink-2)' : 'var(--critical)' }}>
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
              {agrees ? <path d="M2.5 6.2l2.3 2.3L9.5 3.5" /> : <path d="M3 3l6 6M9 3l-6 6" />}
            </svg>
            {agrees
              ? `Server decision matches the browser: ${py.rule_ids.join(' → ')}`
              : `Server disagrees (${py.recommended_action} via ${py.rule_ids.join(' → ')}): the TypeScript rules have drifted`}
          </span>
        ) : <span />}
        <button onClick={() => setShowPrompt(true)} className="font-medium text-ink-2 underline decoration-line-strong underline-offset-2 hover:text-ink">
          View exact prompt
        </button>
      </div>
      {showPrompt && (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/40 p-4" onClick={() => setShowPrompt(false)}>
          <div
            role="dialog"
            aria-modal="true"
            aria-label="Exact prompt sent to the LLM"
            className="card max-h-[80vh] w-full max-w-2xl overflow-hidden"
            onClick={(e) => e.stopPropagation()}
            onKeyDown={(e) => e.key === 'Escape' && setShowPrompt(false)}
          >
            <div className="flex items-center justify-between border-b border-line px-4 py-3">
              <div className="text-sm font-semibold">Prompt · engine {request.unit_number}, cycle {request.time_cycles}</div>
              <button autoFocus className="rounded-md px-2 py-0.5 text-xs text-ink-2 hover:bg-surface-2" onClick={() => setShowPrompt(false)}>
                Close
              </button>
            </div>
            <pre className="max-h-[calc(80vh-52px)] overflow-auto whitespace-pre-wrap p-4 font-mono text-[11.5px] leading-relaxed text-ink-2">{response.prompt}</pre>
          </div>
        </div>
      )}
    </div>
  )
}

function Spinner() {
  return (
    <svg className="animate-spin" width="13" height="13" viewBox="0 0 16 16" fill="none" aria-hidden>
      <circle cx="8" cy="8" r="6" stroke="currentColor" strokeWidth="2" opacity="0.3" />
      <path d="M14 8a6 6 0 00-6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  )
}
