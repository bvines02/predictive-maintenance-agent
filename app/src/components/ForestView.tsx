import { useEffect, useState } from 'react'
import { isLeaf, loadForest, type EngineData, type FixtureIndex, type ForestData } from '../pipeline/data'
import type { HealthThresholds } from '../pipeline/rules'
import { RUL_AXIS_MAX, healthZones } from './RulScale'
import { featureLabel } from './stages/DataStages'
import { fmt, useWidth } from './ui'

interface Props {
  index: FixtureIndex
  engine: EngineData
  i: number
  thresholds: HealthThresholds
}

/** How the Random Forest reaches its number: one real tree's route, then all 200 trees' estimates. */
export function ForestView({ index, engine, i, thresholds }: Props) {
  const [forest, setForest] = useState<ForestData | null>(null)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    let live = true
    setForest(null)
    setFailed(false)
    loadForest(engine.unit)
      .then((f) => live && setForest(f))
      .catch(() => live && setFailed(true))
    return () => {
      live = false
    }
  }, [engine.unit])

  const treeNo = index.model.sample_tree.index + 1
  return (
    <div className="grid gap-x-10 gap-y-8 xl:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
      <div className="min-w-0">
        <Heading n={1} title="One tree is a flowchart">
          Tree #{treeNo} of {index.model.n_trees} answers a chain of learned yes/no questions about this cycle's features. Where it
          stops, it predicts the average RUL of the training rows that ended up there.
        </Heading>
        {forest ? <TreePath index={index} forest={forest} i={i} /> : <Placeholder failed={failed} rows={6} />}
      </div>
      <div className="min-w-0">
        <Heading n={2} title={`The forest averages ${index.model.n_trees} such trees`}>
          Each tree learned from a different bootstrap resample, so each routes the same cycle differently and lands on a
          different estimate. One dot per tree; their mean is the model's prediction.
        </Heading>
        {forest ? (
          <TreeDotPlot forest={forest} engine={engine} i={i} thresholds={thresholds} treeNo={treeNo} />
        ) : (
          <Placeholder failed={failed} rows={4} />
        )}
      </div>
    </div>
  )
}

function Heading({ n, title, children }: { n: number; title: string; children: React.ReactNode }) {
  return (
    <div className="mb-4">
      <div className="flex items-center gap-2">
        <span className="tabular flex h-5 w-5 items-center justify-center rounded-full bg-series-1 text-[11px] font-semibold text-white">{n}</span>
        <h3 className="text-sm font-semibold">{title}</h3>
      </div>
      <p className="mt-1.5 text-[13px] leading-relaxed text-ink-2">{children}</p>
    </div>
  )
}

function Placeholder({ failed, rows }: { failed: boolean; rows: number }) {
  if (failed) return <p className="text-sm text-ink-3">Tree-level data for this engine could not be loaded.</p>
  return (
    <div className="space-y-2" aria-hidden>
      {Array.from({ length: rows }, (_, k) => (
        <div key={k} className="h-9 animate-pulse rounded-md bg-surface-2" />
      ))}
    </div>
  )
}

function describeFeature(index: FixtureIndex, f: number) {
  const col = index.model.feature_columns[f]
  if (col.sensor) return { name: col.symbol, detail: featureLabel(col.transform) }
  if (col.feature === 'time_cycles') return { name: 'Engine age', detail: 'cycles' }
  return { name: col.feature.replace(/_/g, ' ').replace(/^\w/, (c) => c.toUpperCase()), detail: 'operating setting' }
}

const COLLAPSED_SPLITS = 4

function TreePath({ index, forest, i }: { index: FixtureIndex; forest: ForestData; i: number }) {
  const [expanded, setExpanded] = useState(false)
  const { nodes } = forest.sample_tree
  const path = forest.sample_tree.paths[i]
  const values = forest.sample_tree.split_values[i]
  const splits = path.slice(0, -1)
  const leaf = nodes[String(path[path.length - 1])]
  const rootRows = nodes['0'].samples
  const trainRows = index.engines.filter((e) => e.split === 'train').reduce((s, e) => s + e.n_cycles, 0)
  const shown = expanded ? splits : splits.slice(0, COLLAPSED_SPLITS)
  const hidden = splits.length - shown.length

  return (
    <div>
      <ol>
        {shown.map((id, k) => {
          const node = nodes[String(id)]
          if (isLeaf(node)) return null
          const yes = path[k + 1] === node.left
          const f = describeFeature(index, node.feature)
          return (
            <li key={id} className="relative pb-3 pl-8">
              <span className="absolute left-[11px] top-6 bottom-0 w-px bg-line-strong" aria-hidden />
              <span className="tabular absolute left-0 top-0.5 flex h-[23px] w-[23px] items-center justify-center rounded-full bg-surface-2 text-[10px] font-semibold text-ink-2">
                {k + 1}
              </span>
              <div className="rounded-lg px-3 py-2" style={{ boxShadow: 'inset 0 0 0 1px var(--line)' }}>
                <div className="flex items-baseline justify-between gap-3">
                  <div className="min-w-0 text-[13px]">
                    <span className="font-semibold">{f.name}</span> <span className="text-ink-2">· {f.detail}</span>
                  </div>
                  <span className="tabular shrink-0 text-[11px] text-ink-3" title="Distinct training rows that reached this question">
                    {node.samples.toLocaleString()} rows
                  </span>
                </div>
                <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs">
                  <span className="text-ink-3">Is</span>
                  <span className="tabular font-mono font-medium">{fmtVal(values[k])}</span>
                  <span className="text-ink-3">≤</span>
                  <span className="tabular font-mono font-medium">{fmtVal(node.threshold)}</span>
                  <span className="text-ink-3">?</span>
                  <span
                    className="ml-auto rounded-full px-2 py-px text-[11px] font-semibold"
                    style={yes ? { background: 'var(--ink)', color: 'var(--surface)' } : { boxShadow: 'inset 0 0 0 1px var(--ink)', color: 'var(--ink)' }}
                  >
                    {yes ? 'Yes → left' : 'No → right'}
                  </span>
                </div>
                <div className="mt-2 h-[3px] rounded-full bg-surface-2">
                  <div className="h-full rounded-full bg-series-1" style={{ width: `${Math.max(1, (node.samples / rootRows) * 100)}%` }} />
                </div>
              </div>
            </li>
          )
        })}
      </ol>

      {splits.length > COLLAPSED_SPLITS && (
        <div className="relative pb-3 pl-8">
          <span className="absolute left-[11px] top-0 bottom-0 w-px border-l border-dashed border-line-strong" aria-hidden />
          <button onClick={() => setExpanded((e) => !e)} className="rounded-md px-2 py-1 text-xs font-semibold text-ink-2 hover:bg-surface-2 hover:text-ink">
            {expanded ? 'Collapse the path' : `Show ${hidden} more question${hidden === 1 ? '' : 's'}`}
          </button>
        </div>
      )}

      {isLeaf(leaf) && (
        <div className="relative pl-8">
          <span className="absolute left-0 top-2 flex h-[23px] w-[23px] items-center justify-center rounded-full bg-ink" aria-hidden>
            <svg width="11" height="11" viewBox="0 0 12 12" fill="none" stroke="var(--surface)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M2.5 6.2l2.3 2.3L9.5 3.5" />
            </svg>
          </span>
          <div className="rounded-lg bg-surface-2 px-3 py-2.5">
            <div className="text-xs text-ink-3">Leaf after {splits.length} questions · {leaf.samples.toLocaleString()} training row{leaf.samples === 1 ? '' : 's'}</div>
            <div className="mt-0.5 text-sm">
              Tree #{index.model.sample_tree.index + 1} predicts <span className="font-semibold">{fmt(leaf.value)} cycles</span>
            </div>
          </div>
        </div>
      )}

      <p className="mt-3 text-xs leading-relaxed text-ink-3">
        This tree was fitted to {rootRows.toLocaleString()} of the {trainRows.toLocaleString()} training rows (
        {Math.round((rootRows / trainRows) * 100)}%, as expected of a bootstrap resample) and grown to {index.model.sample_tree.depth} levels
        with {index.model.sample_tree.n_leaves.toLocaleString()} leaves. Grown that deep, one tree memorises its sample: accurate on it,
        unreliable on new data.
      </p>
    </div>
  )
}

const fmtVal = (v: number) => {
  const a = Math.abs(v)
  const digits = a >= 1000 ? 1 : a >= 100 ? 2 : a >= 1 ? 3 : 4
  return v.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits })
}

const BIN = 2
const PLOT_H = 150

function TreeDotPlot({ forest, engine, i, thresholds, treeNo }: { forest: ForestData; engine: EngineData; i: number; thresholds: HealthThresholds; treeNo: number }) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const [hover, setHover] = useState<number | null>(null)
  const preds = forest.tree_predictions[i]
  const path = forest.sample_tree.paths[i]
  const leaf = forest.sample_tree.nodes[String(path[path.length - 1])]
  const sampleValue = isLeaf(leaf) ? leaf.value : NaN
  const mean = engine.predicted_rul[i]
  const p10 = engine.tree_quantiles.p10[i]
  const p90 = engine.tree_quantiles.p90[i]

  const nBins = Math.ceil((RUL_AXIS_MAX + 1) / BIN)
  const counts = new Array<number>(nBins).fill(0)
  for (const v of preds) counts[Math.min(nBins - 1, Math.max(0, Math.floor(v / BIN)))]++
  const maxStack = Math.max(...counts)
  const x = (v: number) => (Math.min(Math.max(v, 0), RUL_AXIS_MAX) / RUL_AXIS_MAX) * width
  const binPx = (BIN / RUL_AXIS_MAX) * width
  // Dots stack upward from the baseline; when the tallest stack can't fit at
  // full size, the pitch shrinks so dots overlap into a column whose height
  // stays proportional to its count.
  const fitPitch = (PLOT_H - 6) / maxStack
  const dot = Math.max(2, Math.min(binPx - 1.5, fitPitch - 1))
  const pitch = Math.min(dot + 1, fitPitch)
  const compressed = pitch < dot + 1
  const sampleBin = Math.floor(Math.round(sampleValue) / BIN)

  return (
    <div className="pt-5">
      <div ref={ref} className="relative" style={{ height: PLOT_H + 46 }}>
        {width > 0 && (
          <svg width={width} height={PLOT_H + 46} className="block overflow-visible" role="img"
            aria-label={`Estimates of all ${preds.length} trees at cycle ${engine.cycle[i]}: mean ${fmt(mean)}, middle 80% ${fmt(p10, 0)} to ${fmt(p90, 0)} cycles`}>
            {healthZones(thresholds).map((z) => (
              <rect key={z.label} x={x(z.from) + 0.5} y={0} width={Math.max(0, x(z.to) - x(z.from) - 1)} height={PLOT_H} fill={z.color} opacity={0.07} rx={4} />
            ))}
            <line x1={0} x2={width} y1={PLOT_H} y2={PLOT_H} stroke="var(--line-strong)" />
            {counts.map((c, b) =>
              Array.from({ length: c }, (_, k) => {
                const isSample = b === sampleBin && k === 0
                return (
                  <circle
                    key={`${b}-${k}`}
                    cx={x(b * BIN + BIN / 2)}
                    cy={PLOT_H - 2 - dot / 2 - k * pitch}
                    r={dot / 2}
                    fill={isSample ? 'var(--ink)' : 'var(--series-1)'}
                    opacity={isSample ? 1 : hover === null || hover === b ? 0.8 : 0.45}
                    stroke={isSample ? 'var(--surface)' : 'none'}
                    strokeWidth={isSample ? 1.5 : 0}
                  />
                )
              }),
            )}
            <line x1={x(mean)} x2={x(mean)} y1={-4} y2={PLOT_H} stroke="var(--ink)" strokeWidth={1.5} />
            <g transform={`translate(${Math.min(Math.max(x(mean), 70), width - 70)}, -10)`}>
              <text textAnchor="middle" fontSize={11} fontWeight={600} fill="var(--ink)">Mean {fmt(mean)} = prediction</text>
            </g>
            <g transform={`translate(0, ${PLOT_H + 10})`}>
              <line x1={x(p10)} x2={x(p90)} y1={0} y2={0} stroke="var(--ink-2)" />
              <line x1={x(p10)} x2={x(p10)} y1={-3} y2={3} stroke="var(--ink-2)" />
              <line x1={x(p90)} x2={x(p90)} y1={-3} y2={3} stroke="var(--ink-2)" />
              {[0, 25, 50, 75, 100, 125].map((t) => (
                <text key={t} x={x(t)} y={22} textAnchor="middle" fontSize={10} fill="var(--ink-3)" className="tabular">{t}</text>
              ))}
              <text x={width / 2} y={34} textAnchor="middle" fontSize={10} fill="var(--ink-3)">Tree estimate of remaining useful life (cycles)</text>
            </g>
            {counts.map((c, b) =>
              c > 0 ? (
                <rect key={`hit-${b}`} x={x(b * BIN)} y={0} width={Math.max(binPx, 8)} height={PLOT_H} fill="transparent"
                  onPointerEnter={() => setHover(b)} onPointerLeave={() => setHover(null)} />
              ) : null,
            )}
          </svg>
        )}
        {hover !== null && width > 0 && (
          <div className="card pointer-events-none absolute z-10 -translate-x-1/2 whitespace-nowrap px-2.5 py-1.5 text-xs" style={{ left: x(hover * BIN + BIN / 2), top: 4 }}>
            <strong className="tabular">{counts[hover]}</strong> tree{counts[hover] === 1 ? '' : 's'} estimate {hover * BIN}–{hover * BIN + BIN - 1} cycles
          </div>
        )}
      </div>
      <div className="mt-1 flex flex-wrap gap-x-5 gap-y-1 text-xs text-ink-2">
        <span className="inline-flex items-center gap-1.5"><span className="h-2 w-2 rounded-full bg-series-1" />One tree's estimate</span>
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-ink" style={{ boxShadow: '0 0 0 1.5px var(--surface)' }} />Tree #{treeNo}, traced in the flowchart
        </span>
        <span className="inline-flex items-center gap-1.5"><span className="inline-block h-px w-4 bg-ink-2" />Middle 80%: {fmt(p10, 0)}–{fmt(p90, 0)}</span>
      </div>
      <p className="mt-2 text-xs leading-relaxed text-ink-3">
        Estimates rounded to whole cycles and grouped in {BIN}-cycle bins; faint bands are the current health zones. A tight cluster
        means the trees agree (high confidence); a wide spread means they disagree (low confidence).
        {compressed && <> Here {maxStack} trees share one bin, so dots overlap into a column; its height stays proportional to the count.</>}
      </p>
    </div>
  )
}
