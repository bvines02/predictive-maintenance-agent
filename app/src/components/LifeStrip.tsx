import { useState, type ReactNode } from 'react'

/** One coloured cell per cycle across an engine's whole life. Click or drag to move the cursor. */
export function LifeStrip({
  label,
  colors,
  cursor,
  onSelect,
  tooltip,
  height = 16,
}: {
  label?: string
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

  const strip = (
    <div className="relative min-w-0 flex-1">
      <svg
        viewBox={`0 0 ${n} ${height}`}
        preserveAspectRatio="none"
        className="block w-full cursor-pointer touch-none overflow-hidden rounded-[4px]"
        style={{ height }}
        onPointerMove={(e) => {
          setHover(indexAt(e))
          if (e.buttons === 1) onSelect(indexAt(e))
        }}
        onPointerDown={(e) => onSelect(indexAt(e))}
        onPointerLeave={() => setHover(null)}
        role="img"
        aria-label={`${label ?? 'State'} over the engine's life`}
      >
        {/* Runs of equal colour merge into one rect: fewer nodes, no hairline seams. */}
        {runs(colors).map(([start, end, color]) => (
          <rect key={start} x={start} y={0} width={end - start} height={height} fill={color} />
        ))}
      </svg>
      <div
        className="pointer-events-none absolute -inset-y-1 w-[2px] -translate-x-1/2 rounded-full bg-ink"
        style={{ left: `${((cursor + 0.5) / n) * 100}%`, boxShadow: '0 0 0 2px var(--surface)' }}
      />
      {hover !== null && (
        <div
          className="card pointer-events-none absolute bottom-full z-20 mb-2 -translate-x-1/2 whitespace-nowrap px-2.5 py-1.5 text-xs"
          style={{ left: `${((hover + 0.5) / n) * 100}%` }}
        >
          {tooltip(hover)}
        </div>
      )}
    </div>
  )

  if (!label) return strip
  return (
    <div className="flex items-center gap-3">
      <div className="w-16 shrink-0 text-right text-xs font-medium text-ink-3">{label}</div>
      {strip}
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
