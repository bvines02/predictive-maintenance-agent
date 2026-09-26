import { useEffect, useMemo, useState } from 'react'
import { FlowOverview } from './components/FlowOverview'
import { ParamsPanel } from './components/ParamsPanel'
import { FeatureStage, TelemetryStage } from './components/stages/DataStages'
import { LlmStage } from './components/stages/LlmStage'
import { ConfidenceStage, ModelStage } from './components/stages/ModelStages'
import { DecisionStage, HealthStage } from './components/stages/RuleStages'
import { loadEngine, loadIndex, type EngineData, type FixtureIndex } from './pipeline/data'
import { decide, type DecisionParams, type HealthThresholds } from './pipeline/rules'

export default function App() {
  const [index, setIndex] = useState<FixtureIndex | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [unit, setUnit] = useState<number | null>(null)
  const [engine, setEngine] = useState<EngineData | null>(null)
  const [i, setI] = useState(0)
  const [thresholds, setThresholds] = useState<HealthThresholds | null>(null)
  const [leadTime, setLeadTime] = useState(0)
  const [playing, setPlaying] = useState(false)

  useEffect(() => {
    loadIndex()
      .then((idx) => {
        setIndex(idx)
        setThresholds(idx.defaults.health_thresholds)
        setLeadTime(idx.defaults.maintenance_lead_time_cycles)
        setUnit(idx.engines.find((e) => e.split === 'validation')!.unit)
      })
      .catch((e) => setError(String(e)))
  }, [])

  useEffect(() => {
    if (unit === null) return
    let live = true
    loadEngine(unit)
      .then((e) => {
        if (!live) return
        setEngine(e)
        // Open near end of life, where the rules have something to do.
        setI(Math.max(0, e.cycle.length - 30))
      })
      .catch((e) => setError(String(e)))
    return () => {
      live = false
    }
  }, [unit])

  useEffect(() => {
    if (!playing || !engine) return
    const timer = setInterval(() => {
      setI((k) => {
        if (k >= engine.cycle.length - 1) {
          setPlaying(false)
          return k
        }
        return k + 1
      })
    }, 60)
    return () => clearInterval(timer)
  }, [playing, engine])

  const params: DecisionParams | null =
    index && thresholds
      ? { thresholds, leadTimeCycles: leadTime, leadTimeBufferCycles: index.defaults.lead_time_buffer_cycles }
      : null

  const decisions = useMemo(
    () => (engine && params ? engine.predicted_rul.map((r, k) => decide(r, engine.model_confidence[k], params)) : []),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [engine, thresholds, leadTime],
  )

  if (error) return <div className="p-6 text-sm">Could not load pipeline data: {error}</div>
  if (!index || !engine || !params || !thresholds || decisions.length !== engine.cycle.length) {
    return <div className="p-6 text-sm text-ink-2">Loading pipeline…</div>
  }

  const decision = decisions[i]
  const validation = index.engines.filter((e) => e.split === 'validation')
  const training = index.engines.filter((e) => e.split === 'train')
  const stageProps = { index, engine, i, setI }

  return (
    <div className="min-h-screen">
      <header style={{ top: 'env(safe-area-inset-top, 0px)' }} className="z-30 sm:sticky border-b border-line bg-page/95 backdrop-blur">
        <div className="mx-auto max-w-[1400px] px-4 pb-3 pt-3 sm:px-6">
          <div className="mb-3 flex flex-wrap items-center gap-x-6 gap-y-2">
            <div className="mr-auto">
              <h1 className="text-lg font-bold leading-tight">Turbofan Pipeline Explorer</h1>
              <p className="text-xs text-ink-3">NASA C-MAPSS FD001 · telemetry → ML → deterministic rules → LLM explanation</p>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <label htmlFor="engine" className="text-ink-2">Engine</label>
              <select
                id="engine"
                value={unit ?? ''}
                onChange={(e) => {
                  setPlaying(false)
                  setUnit(Number(e.target.value))
                }}
                className="rounded-md border border-line bg-surface px-2 py-1 text-sm"
              >
                <optgroup label="Held out from training (honest error)">
                  {validation.map((e) => <option key={e.unit} value={e.unit}>Engine {e.unit} · {e.n_cycles} cycles</option>)}
                </optgroup>
                <optgroup label="Training engines (in-sample, optimistic)">
                  {training.map((e) => <option key={e.unit} value={e.unit}>Engine {e.unit} · {e.n_cycles} cycles</option>)}
                </optgroup>
              </select>
              <span
                className="rounded-full px-2 py-0.5 text-[11px] font-semibold"
                style={engine.split === 'validation'
                  ? { background: 'var(--surface-2)', color: 'var(--ink-2)' }
                  : { background: 'color-mix(in srgb, var(--serious) 25%, transparent)', color: 'var(--ink)' }}
              >
                {engine.split === 'validation' ? 'held out' : 'in-sample'}
              </span>
            </div>
          </div>

          <div className="mb-3 flex items-center gap-3">
            <button
              onClick={() => {
                if (i >= engine.cycle.length - 1) setI(0)
                setPlaying((p) => !p)
              }}
              className="w-16 shrink-0 rounded-md bg-ink px-2 py-1 text-xs font-semibold text-page"
              aria-label={playing ? 'Pause' : 'Play through the engine life'}
            >
              {playing ? 'Pause' : 'Play'}
            </button>
            <label htmlFor="cycle" className="sr-only">Cycle</label>
            <input
              id="cycle"
              type="range"
              min={0}
              max={engine.cycle.length - 1}
              value={i}
              onChange={(e) => {
                setPlaying(false)
                setI(Number(e.target.value))
              }}
              className="min-w-0 flex-1"
            />
            <div className="tabular w-36 shrink-0 text-right text-sm">
              cycle <strong>{engine.cycle[i]}</strong> <span className="text-ink-3">/ {engine.cycle[engine.cycle.length - 1]}</span>
            </div>
          </div>

          <FlowOverview index={index} engine={engine} i={i} decision={decision} />
        </div>
      </header>

      <div className="mx-auto grid max-w-[1400px] gap-5 px-4 py-5 sm:px-6 lg:grid-cols-[260px_minmax(0,1fr)]">
        <aside className="lg:sticky lg:top-[228px] lg:self-start">
          <div className="rounded-xl border border-line bg-surface p-4">
            <ParamsPanel
              thresholds={thresholds}
              onThresholds={setThresholds}
              leadTime={leadTime}
              onLeadTime={setLeadTime}
              buffer={params.leadTimeBufferCycles}
              defaults={{ thresholds: index.defaults.health_thresholds, leadTime: index.defaults.maintenance_lead_time_cycles }}
            />
          </div>
        </aside>

        <main className="min-w-0 space-y-5">
          <TelemetryStage {...stageProps} />
          <FeatureStage {...stageProps} />
          <ModelStage {...stageProps} thresholds={thresholds} />
          <ConfidenceStage index={index} engine={engine} i={i} />
          <HealthStage {...stageProps} decisions={decisions} params={params} />
          <DecisionStage {...stageProps} decisions={decisions} params={params} />
          <LlmStage engine={engine} i={i} decision={decision} params={params} />
          <p className="pb-6 text-center text-xs text-ink-3">
            Model output is precomputed by <code className="font-mono">python -m src.export_pipeline_fixture</code>. Health states and
            decisions are recomputed live in the browser, checked against the Python engine by <code className="font-mono">app/src/pipeline/rules.test.ts</code>.
          </p>
        </main>
      </div>
    </div>
  )
}
