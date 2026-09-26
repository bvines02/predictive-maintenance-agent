import { useState, type ReactNode } from 'react'

/** One coloured cell per cycle across an engine's whole life. Click or drag to move the cursor. */
export function LifeStrip({
  label,
  colors,
  cursor,
  onSelect,
  tooltip,
  height = 18,
}: {
  label: string
  colors: string[]
  cursor: number
  onSelect: (i: number) => void
  tooltip: (i: number) => ReactNode
  height?: number
}) {
  const [hover, setHover] = useState<number | null>(null)
  const n = colors.length
  const indexAt = (e: React.PointerEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    return Math.min(n - 1, Math.max(0, Math.floor(((e.clientX - rect.left) / rect.width) * n)))
  }
  const shown = hover ?? null

  return (
    <div className="flex items-center gap-3">
      <div className="w-20 shrink-0 text-right text-xs text-ink-2">{label}</div>
      <div className="relative min-w-0 flex-1">
        <svg
          viewBox={`0 0 ${n} ${height}`}
          preserveAspectRatio="none"
          className="block w-full cursor-pointer touch-none rounded-sm"
          style={{ height }}
          onPointerMove={(e) => {
            setHover(indexAt(e))
            if (e.buttons === 1) onSelect(indexAt(e))
          }}
          onPointerDown={(e) => onSelect(indexAt(e))}
          onPointerLeave={() => setHover(null)}
          role="img"
          aria-label={`${label} over the engine's life`}
        >
          {/* Merge runs of equal colour into one rect: fewer nodes, no hairline seams. */}
          {runs(colors).map(([start, end, color]) => (
            <rect key={start} x={start} y={0} width={end - start} height={height} fill={color} />
          ))}
          <rect x={cursor - 0.5} y={0} width={Math.max(1, n / 250)} height={height} fill="var(--ink)" />
        </svg>
        {shown !== null && (
          <div
            className="pointer-events-none absolute bottom-full z-20 mb-1 -translate-x-1/2 whitespace-nowrap rounded-md border border-line bg-surface px-2 py-1 text-xs shadow-lg"
            style={{ left: `${((shown + 0.5) / n) * 100}%` }}
          >
            {tooltip(shown)}
          </div>
        )}
      </div>
    </div>
  )
}

function runs(colors: string[]): [number, number, string][] {
  const out: [number, number, string][] = []
  colors.forEach((c, i) => {
    const last = out[out.length - 1]
    if (last && last[2] === c) last[1] = i + 1
    else out.push([i, i + 1, c])
  })
  return out
}
