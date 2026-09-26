import type { ReactNode } from 'react'
import type { EngineData, FixtureIndex } from '../pipeline/data'
import type { Decision } from '../pipeline/rules'
import { ActionBadge, HealthBadge, fmt } from './ui'

/** The whole pipeline in one row: each stage's live output at the current cycle. Click to jump. */
export function FlowOverview({ index, engine, i, decision }: { index: FixtureIndex; engine: EngineData; i: number; decision: Decision }) {
  const nodes: { id: string; step: number; label: string; kind: 'data' | 'ml' | 'rules' | 'llm'; value: ReactNode }[] = [
    { id: 'stage-telemetry', step: 1, label: 'Telemetry', kind: 'data', value: `${index.kept_sensors.length} sensors` },
    { id: 'stage-features', step: 2, label: 'Features', kind: 'data', value: `${index.model.n_features} per cycle` },
    { id: 'stage-model', step: 3, label: 'Random Forest', kind: 'ml', value: <span className="tabular">RUL {fmt(engine.predicted_rul[i])}</span> },
    { id: 'stage-confidence', step: 4, label: 'Confidence', kind: 'ml', value: engine.model_confidence[i] },
    { id: 'stage-health', step: 5, label: 'Health state', kind: 'rules', value: <HealthBadge state={decision.healthState} /> },
    { id: 'stage-decision', step: 6, label: 'Decision', kind: 'rules', value: <ActionBadge action={decision.recommendedAction} /> },
    { id: 'stage-llm', step: 7, label: 'LLM', kind: 'llm', value: 'explains only' },
  ]
  const kindColor = { data: 'var(--line)', ml: 'var(--series-1)', rules: 'var(--ink-2)', llm: 'var(--accent)' }
  return (
    <nav aria-label="Pipeline stages" className="-mx-1 overflow-x-auto px-1">
      <ol className="flex min-w-max items-stretch gap-1.5">
        {nodes.map((n, k) => (
          <li key={n.id} className="flex items-center gap-1.5">
            <a
              href={`#${n.id}`}
              className="flex min-w-[112px] flex-col gap-1 rounded-lg border border-t-[3px] border-line bg-surface px-2.5 py-1.5 hover:bg-surface-2"
              style={{ borderTopColor: kindColor[n.kind] }}
            >
              <span className="text-[10px] font-medium uppercase tracking-wide text-ink-3">{n.step} · {n.label}</span>
              <span className="text-xs font-semibold">{n.value}</span>
            </a>
            {k < nodes.length - 1 && <span aria-hidden className="text-ink-3">→</span>}
          </li>
        ))}
      </ol>
      <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-ink-3">
        <span className="inline-flex items-center gap-1"><span className="h-[3px] w-4 bg-line" />data</span>
        <span className="inline-flex items-center gap-1"><span className="h-[3px] w-4 bg-series-1" />machine learning</span>
        <span className="inline-flex items-center gap-1"><span className="h-[3px] w-4 bg-ink-2" />deterministic rules (you edit these)</span>
        <span className="inline-flex items-center gap-1"><span className="h-[3px] w-4 bg-accent" />LLM (read-only)</span>
      </div>
    </nav>
  )
}
