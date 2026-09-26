import { afterEach, describe, expect, it, vi } from 'vitest'
import { POST } from '../api/explain'

const PAYLOAD = {
  unit_number: 7,
  time_cycles: 180,
  predicted_rul: 12,
  prediction_p10: 8,
  prediction_p90: 19.5,
  model_confidence: 'HIGH',
  health_thresholds: { action: 15, plan: 30, watch: 60 },
  maintenance_lead_time_cycles: 15,
}
const post = (body: unknown) =>
  POST(new Request('http://x/api/explain', { method: 'POST', body: JSON.stringify(body) }))

afterEach(() => {
  vi.unstubAllEnvs()
  vi.unstubAllGlobals()
})

describe('POST /api/explain', () => {
  it('re-derives the decision and returns the prompt without a key', async () => {
    vi.stubEnv('ANTHROPIC_API_KEY', '')
    const res = await post(PAYLOAD)
    const body = await res.json()
    expect(res.status).toBe(200)
    expect(body.decision.recommended_action).toBe('INTERVENE_NOW')
    expect(body.decision.rule_ids).toEqual(['BASE_01', 'LEAD_01'])
    expect(body.prompt).toContain('Engine 7, flight cycle 180')
    expect(body.explanation).toBeNull()
  })

  it('calls Anthropic with the key server-side and returns its text', async () => {
    vi.stubEnv('ANTHROPIC_API_KEY', 'test-key')
    const fetchMock = vi.fn(async () => Response.json({ content: [{ type: 'text', text: 'Act now.' }] }))
    vi.stubGlobal('fetch', fetchMock)
    const body = await (await post(PAYLOAD)).json()
    expect(body.explanation).toBe('Act now.')
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit]
    expect(url).toBe('https://api.anthropic.com/v1/messages')
    expect((init.headers as Record<string, string>)['x-api-key']).toBe('test-key')
    expect(JSON.parse(init.body as string).messages[0].content).toBe(body.prompt)
  })

  it('reports an upstream failure as 502', async () => {
    vi.stubEnv('ANTHROPIC_API_KEY', 'test-key')
    vi.stubGlobal('fetch', vi.fn(async () => new Response('nope', { status: 529 })))
    expect((await post(PAYLOAD)).status).toBe(502)
  })

  it.each([
    ['unordered thresholds', { ...PAYLOAD, health_thresholds: { action: 40, plan: 30, watch: 60 } }],
    ['unknown confidence', { ...PAYLOAD, model_confidence: 'SURE' }],
    ['missing rul', { ...PAYLOAD, predicted_rul: undefined }],
    ['negative lead time', { ...PAYLOAD, maintenance_lead_time_cycles: -1 }],
  ])('rejects %s with 422', async (_, body) => {
    expect((await post(body)).status).toBe(422)
  })
})
