/** Types and loaders for the static fixture written by src/export_pipeline_fixture.py. */
import type { Confidence, HealthThresholds } from './rules'

export interface SensorInfo {
  column: string
  symbol: string
  name: string
  unit: string
  expected_trend: string
  status: 'keep' | 'constant' | 'near_constant'
  unique_values: number
}

export interface TopFeature {
  feature: string
  category: string
  importance: number
  sensor: string | null
  symbol: string
  transform: string
}

export interface FixtureIndex {
  dataset: { name: string; n_engines: number; n_rows: number; rul_cap: number }
  model: {
    type: string
    n_trees: number
    random_state: number
    target: string
    n_features: number
    rolling_windows: number[]
    trend_window: number
    validation_mae: number
    validation_rmse: number
    feature_categories: Record<string, { count: number; importance: number }>
    top_features: TopFeature[]
  }
  sensors: SensorInfo[]
  kept_sensors: string[]
  uncertainty: { prediction_std_low_max: number; prediction_std_medium_max: number }
  defaults: {
    health_thresholds: HealthThresholds
    maintenance_lead_time_cycles: number
    lead_time_buffer_cycles: number
  }
  action_meanings: Record<string, string>
  engines: { unit: number; split: 'train' | 'validation'; n_cycles: number }[]
}

export type TreeQuantile = 'p0' | 'p10' | 'p25' | 'p50' | 'p75' | 'p90' | 'p100'

export interface EngineData {
  unit: number
  split: 'train' | 'validation'
  cycle: number[]
  actual_rul: number[]
  actual_rul_capped: number[]
  predicted_rul: number[]
  prediction_std: number[]
  tree_quantiles: Record<TreeQuantile, number[]>
  model_confidence: Confidence[]
  sensors: Record<string, number[]>
  features: Record<string, number[]>
}

const cache = new Map<string, Promise<unknown>>()

function getJson<T>(path: string): Promise<T> {
  if (!cache.has(path)) {
    cache.set(
      path,
      fetch(path).then((r) => {
        if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`)
        return r.json()
      }),
    )
  }
  return cache.get(path) as Promise<T>
}

export const loadIndex = () => getJson<FixtureIndex>('./data/index.json')
export const loadEngine = (unit: number) => getJson<EngineData>(`./data/engines/${unit}.json`)
