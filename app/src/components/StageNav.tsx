import { useEffect, useState } from 'react'
import { KIND, type StageKind } from './ui'

const ITEMS: { id: string; step?: number; label: string; kind?: StageKind }[] = [
  { id: 'overview', label: 'The answer' },
  { id: 'stage-engine', step: 1, label: 'An engine wears out', kind: 'data' },
  { id: 'stage-model', step: 2, label: 'A model predicts', kind: 'ml' },
  { id: 'stage-decision', step: 3, label: 'Rules decide', kind: 'rules' },
  { id: 'stage-llm', step: 4, label: 'AI explains', kind: 'llm' },
]

/** The story as navigation: the answer first, then the four steps that produced it. */
export function StageNav() {
  const active = useActiveSection(ITEMS.map((it) => it.id))
  return (
    <nav aria-label="How the answer is produced">
      <ol className="space-y-0.5">
        {ITEMS.map((it) => {
          const on = active === it.id
          return (
            <li key={it.id}>
              <a
                href={`#${it.id}`}
                aria-current={on ? 'location' : undefined}
                className={`flex items-center gap-2.5 rounded-md px-2.5 py-2 text-sm ${on ? 'bg-surface-2 font-medium text-ink' : 'text-ink-2 hover:text-ink'}`}
              >
                {it.step ? (
                  <span
                    className="tabular flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[10px] font-semibold"
                    style={on ? { background: 'var(--ink)', color: 'var(--surface)' } : { boxShadow: 'inset 0 0 0 1px var(--line-strong)' }}
                  >
                    {it.step}
                  </span>
                ) : (
                  <span className="flex h-5 w-5 shrink-0 items-center justify-center" aria-hidden>
                    <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M3.5 8.5l3 3 6-7" />
                    </svg>
                  </span>
                )}
                <span className="min-w-0 flex-1 truncate">{it.label}</span>
                {it.kind && <span className="h-1.5 w-1.5 shrink-0 rounded-full" style={{ background: KIND[it.kind].color }} title={KIND[it.kind].label} />}
              </a>
            </li>
          )
        })}
      </ol>
    </nav>
  )
}

/** The id of the section nearest the top of the viewport (below the sticky header). */
function useActiveSection(ids: string[]) {
  const [active, setActive] = useState(ids[0])
  const key = ids.join(',')
  useEffect(() => {
    const els = key.split(',').map((id) => document.getElementById(id)).filter((e): e is HTMLElement => !!e)
    const update = () => {
      const line = 160
      let current = els[0]?.id
      for (const el of els) if (el.getBoundingClientRect().top <= line) current = el.id
      if (window.innerHeight + window.scrollY >= document.body.scrollHeight - 4) current = els[els.length - 1]?.id
      if (current) setActive(current)
    }
    update()
    window.addEventListener('scroll', update, { passive: true })
    window.addEventListener('resize', update)
    return () => {
      window.removeEventListener('scroll', update)
      window.removeEventListener('resize', update)
    }
  }, [key])
  return active
}
