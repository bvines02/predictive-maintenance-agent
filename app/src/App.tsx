import { useEffect, useMemo, useState } from 'react'
import { Overview } from './components/Overview'
import { ParamsPanel } from './components/ParamsPanel'
import { StageNav } from './components/StageNav'
import { EngineStage } from './components/stages/DataStages'
import { LlmStage } from './components/stages/LlmStage'
import { ModelStage } from './components/stages/ModelStages'
import { DecisionStage } from './components/stages/RuleStages'
import { ExpertContext } from './components/ui'
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
  const [expert, setExpert] = useState(readExpert)

  // Only engines the model never trained on: their errors are honest.
  const engines = useMemo(() => index?.engines.filter((e) => e.split === 'validation') ?? [], [index])

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
    return (
      <div className="flex min-h-screen items-center justify-center text-sm text-ink-3">
        <span className="inline-flex items-center gap-2"><Spinner />Loading pipeline…</span>
      </div>
    )
  }

  const decision = decisions[i]
  const stageProps = { index, engine, i, setI }
  const lastIdx = engine.cycle.length - 1
  const policy = (idPrefix: string) => (
    <ParamsPanel
      idPrefix={idPrefix}
      thresholds={thresholds}
      onThresholds={setThresholds}
      leadTime={leadTime}
      onLeadTime={setLeadTime}
      buffer={params.leadTimeBufferCycles}
      defaults={{ thresholds: index.defaults.health_thresholds, leadTime: index.defaults.maintenance_lead_time_cycles }}
    />
  )

  return (
    <ExpertContext.Provider value={expert}>
    <div className="min-h-screen">
      <header className="z-30 border-b border-line bg-page/85 backdrop-blur-md sm:sticky sm:top-0">
        {/* Brand + engine */}
        <div className="mx-auto flex max-w-[1440px] flex-wrap items-center gap-x-6 gap-y-3 px-4 pt-3 sm:px-6">
          <div className="mr-auto flex items-center gap-3">
            <Logo />
            <div>
              <h1 className="text-[15px] font-semibold leading-tight tracking-tight">Turbofan RUL Explorer</h1>
              <p className="text-xs text-ink-3">NASA C-MAPSS FD001 · predictive maintenance pipeline</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <label htmlFor="engine" className="sr-only">Engine</label>
            <select
              id="engine"
              value={unit ?? ''}
              onChange={(e) => {
                setPlaying(false)
                setUnit(Number(e.target.value))
              }}
              className="select h-8 rounded-lg bg-surface pl-3 text-sm font-medium"
              style={{ boxShadow: '0 0 0 1px var(--ring)' }}
            >
              {engines.map((e) => <option key={e.unit} value={e.unit}>Engine {e.unit} · {e.n_cycles} cycles</option>)}
            </select>
            <ModeSwitch expert={expert} onChange={setExpert} />
            <ThemeToggle />
          </div>
        </div>

        {/* Transport */}
        <div className="mx-auto flex max-w-[1440px] items-center gap-3 px-4 py-3 sm:gap-4 sm:px-6">
          <button
            onClick={() => {
              if (i >= lastIdx) setI(0)
              setPlaying((p) => !p)
            }}
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-ink text-page transition-transform hover:scale-105 active:scale-95"
            aria-label={playing ? 'Pause' : 'Play through the engine life'}
          >
            {playing ? (
              <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor" aria-hidden><rect x="2" y="1.5" width="3" height="9" rx="1" /><rect x="7" y="1.5" width="3" height="9" rx="1" /></svg>
            ) : (
              <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor" aria-hidden><path d="M3 1.8v8.4a.6.6 0 00.9.5l7-4.2a.6.6 0 000-1L3.9 1.3a.6.6 0 00-.9.5z" /></svg>
            )}
          </button>
          <div className="shrink-0 whitespace-nowrap text-sm sm:w-[108px]">
            <span className="text-ink-3">Cycle </span>
            <span className="tabular font-semibold">{engine.cycle[i]}</span>
            <span className="tabular text-ink-3">/{engine.cycle[lastIdx]}</span>
          </div>
          <label htmlFor="cycle" className="sr-only">Cycle</label>
          <input
            id="cycle"
            type="range"
            min={0}
            max={lastIdx}
            value={i}
            onChange={(e) => {
              setPlaying(false)
              setI(Number(e.target.value))
            }}
            className="range min-w-0 flex-1"
            style={{ '--fill': `${(i / lastIdx) * 100}%` } as React.CSSProperties}
          />
        </div>
      </header>

      <div className="mx-auto grid max-w-[1440px] gap-8 px-4 py-8 sm:px-6 lg:grid-cols-[264px_minmax(0,1fr)]">
        <aside className="hidden lg:sticky lg:top-[128px] lg:block lg:max-h-[calc(100vh-148px)] lg:self-start lg:overflow-y-auto">
          <StageNav />
          <div className="card mt-6 p-4">{policy('side')}</div>
        </aside>

        <main className="min-w-0 space-y-16">
          <div className="space-y-6">
            <Overview {...stageProps} decisions={decisions} thresholds={thresholds} />
            <div className="card p-4 lg:hidden">{policy('inline')}</div>
          </div>
          <EngineStage {...stageProps} />
          <ModelStage index={index} engine={engine} i={i} thresholds={thresholds} />
          <DecisionStage {...stageProps} decisions={decisions} params={params} />
          <LlmStage engine={engine} i={i} decision={decision} params={params} />
          <footer className="border-t border-line pt-6 pb-10 text-xs leading-relaxed text-ink-3">
            Data: NASA C-MAPSS turbofan degradation simulation (FD001). All {engines.length} engines shown were held out from training.
            Model outputs are precomputed by <code className="font-mono">python -m src.export_pipeline_fixture</code>; decisions are
            recomputed live in the browser and verified against the Python engine by <code className="font-mono">app/src/pipeline/rules.test.ts</code>.
          </footer>
        </main>
      </div>
    </div>
    </ExpertContext.Provider>
  )
}

function readExpert(): boolean {
  try {
    return localStorage.getItem('detail') === 'expert'
  } catch {
    return false
  }
}

function ModeSwitch({ expert, onChange }: { expert: boolean; onChange: (v: boolean) => void }) {
  const set = (v: boolean) => {
    onChange(v)
    try {
      localStorage.setItem('detail', v ? 'expert' : 'simple')
    } catch {
      /* storage unavailable: the choice still applies for this visit */
    }
  }
  return (
    <div role="radiogroup" aria-label="Level of detail" className="flex h-8 items-center rounded-lg bg-surface-2 p-0.5 text-xs font-semibold">
      {[false, true].map((v) => (
        <button
          key={String(v)}
          role="radio"
          aria-checked={expert === v}
          onClick={() => set(v)}
          className={`h-7 rounded-md px-2.5 transition-colors ${expert === v ? 'bg-surface text-ink shadow-sm' : 'text-ink-3 hover:text-ink'}`}
          title={v ? 'Show the supporting detail: features, confidence cut-points, rule IDs, history' : 'Just the story'}
        >
          {v ? 'Expert' : 'Simple'}
        </button>
      ))}
    </div>
  )
}

function Logo() {
  return (
    <span className="flex h-9 w-9 items-center justify-center rounded-[10px] bg-ink text-page" aria-hidden>
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
        <circle cx="12" cy="12" r="9.2" stroke="currentColor" strokeWidth="1.4" opacity="0.45" />
        {[0, 72, 144, 216, 288].map((a) => (
          <path key={a} d="M12 12c.2-2.6 1.6-5 4.3-6.2-.1 2.8-1.6 5-4.3 6.2z" fill="currentColor" transform={`rotate(${a} 12 12)`} />
        ))}
        <circle cx="12" cy="12" r="1.9" fill="currentColor" />
      </svg>
    </span>
  )
}

type Theme = 'system' | 'light' | 'dark'
function readTheme(): Theme {
  try {
    const t = localStorage.getItem('theme')
    return t === 'light' || t === 'dark' ? t : 'system'
  } catch {
    return 'system'
  }
}

function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(readTheme)
  useEffect(() => {
    const root = document.documentElement
    if (theme === 'system') delete root.dataset.theme
    else root.dataset.theme = theme
    try {
      if (theme === 'system') localStorage.removeItem('theme')
      else localStorage.setItem('theme', theme)
    } catch {
      /* storage unavailable: theme still applies for this visit */
    }
  }, [theme])
  const next: Record<Theme, Theme> = { system: 'light', light: 'dark', dark: 'system' }
  const label = { system: 'System theme', light: 'Light theme', dark: 'Dark theme' }[theme]
  return (
    <button
      onClick={() => setTheme(next[theme])}
      className="flex h-8 w-8 items-center justify-center rounded-lg text-ink-2 hover:bg-surface-2 hover:text-ink"
      aria-label={`${label} — click to change`}
      title={label}
    >
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" aria-hidden>
        {theme === 'light' && (<><circle cx="8" cy="8" r="3" /><path d="M8 1v1.5M8 13.5V15M1 8h1.5M13.5 8H15M3 3l1 1M12 12l1 1M3 13l1-1M12 4l1-1" /></>)}
        {theme === 'dark' && <path d="M13.5 9.5A5.5 5.5 0 016.5 2.5a5.5 5.5 0 107 7z" />}
        {theme === 'system' && (<><rect x="1.5" y="2.5" width="13" height="9" rx="1.5" /><path d="M5.5 14h5" /></>)}
      </svg>
    </button>
  )
}

function Spinner() {
  return (
    <svg className="animate-spin" width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden>
      <circle cx="8" cy="8" r="6" stroke="currentColor" strokeWidth="2" opacity="0.25" />
      <path d="M14 8a6 6 0 00-6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  )
}
