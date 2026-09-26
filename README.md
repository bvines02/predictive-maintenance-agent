# Predictive Maintenance Agent

[![Tests](https://github.com/bvines02/predictive-maintenance-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/bvines02/predictive-maintenance-agent/actions/workflows/tests.yml)

A predictive maintenance decision-support system: it predicts equipment failure risk from telemetry, applies deterministic maintenance rules informed by asset context, and uses an LLM strictly to *explain* the resulting decision in maintenance language — with a human reviewer as the final step for anything high-risk.

This is a learning project built to practice real AI engineering architecture, not a notebook-only ML demo. The emphasis throughout is on keeping the machine-learned, the rule-based, and the generative parts of the system cleanly separated, testable independently, and safe to compose.

**This repo now contains two generations of the same architecture.** V1 (below, through "Future roadmap") is a binary failure classifier on the AI4I 2020 dataset. **[V2](#v2-remaining-useful-life-nasa-c-mapss)** upgrades the ML core to Remaining Useful Life (RUL) regression on NASA's C-MAPSS turbofan dataset, and extends the architecture with model uncertainty and a richer, traceable decision engine. The core discipline — deterministic rules between the model and any decision, human review before anything urgent, an LLM that can explain but never override — carries over unchanged.

## Project purpose

Most "AI predictive maintenance" demos stop at a probability score. That's not a decision a maintenance planner can act on by itself — the same failure probability means something very different for a single, uninstrumented pump with no backup than for one of three redundant fans. This project builds the layer most demos skip: turning a risk score plus the asset's real-world context (criticality, redundancy, known failure modes) into an actual recommended action, with a deterministic, auditable trail from score to decision, and a human always in the loop before anything high-risk is acted on.

Concretely, it answers, for one asset at a time:
1. How likely is this asset to fail soon, based on its current telemetry?
2. Given what this specific asset is and how much redundancy it has, what should happen next?
3. Explained in plain maintenance language: why?

## Architecture

```
Telemetry snapshot (asset_id + sensor readings)
        │
        ▼
┌─────────────────────┐
│ src/features.py      │  feature engineering (physically-motivated
│                      │  ratios/interactions: power, wear×torque, ΔT)
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│ src/predict.py       │  trained RandomForestClassifier
│ (models/*.pkl)       │  → failure_probability, risk_band
└─────────┬────────────┘
          ▼
┌─────────────────────┐    ┌──────────────────────────┐
│ src/decision_rules.py│◀───│ src/asset_context.py     │
│ deterministic rules  │    │ criticality, redundancy,  │
│ → action, urgency,   │    │ single point of failure,  │
│   human_review flag  │    │ known failure modes        │
└─────────┬────────────┘    └──────────────────────────┘
          ▼
┌─────────────────────┐
│ src/explainer.py      │  LLM explains the decision above in
│ (Anthropic API)       │  maintenance language - it cannot change it
└─────────┬────────────┘
          ▼
   Human review (required for High-risk / SPOF cases)
```

**The one rule the whole design protects:** the LLM only ever explains a decision that deterministic code already made. It never sees raw telemetry, never computes a risk score, and never chooses an action. If you removed `src/explainer.py` entirely, the system would still produce complete, correct, actionable maintenance recommendations — the LLM step is additive (better communication), never load-bearing (never a dependency for correctness or safety).

This is enforced by more than convention — every function on the decision path (`predict_failure_risk`, `decide_maintenance_action`) is fully unit tested and has no LLM dependency; `generate_explanation()` is the only function in the codebase that imports `anthropic`, and it takes the finished decision as input, not raw data.

## Dataset

**AI4I 2020 Predictive Maintenance Dataset** (UCI Machine Learning Repository): 10,000 rows of simulated industrial machine telemetry — air/process temperature, rotational speed, torque, tool wear — with a binary `Machine failure` label.

- **Why this dataset:** a single flat CSV with a clear binary target, which keeps the framing to binary classification (see Stage 14 below for what changes with a time-series dataset).
- **Class balance:** 339 failures out of 10,000 rows — a 3.39% failure rate, ~28.5:1 imbalance. This shapes the whole modeling approach: `class_weight="balanced"` at training time, and precision/recall/F1/ROC AUC (not accuracy) at evaluation time.
- **Columns used as model features:** `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`, plus three engineered features (below). `TWF`/`HDF`/`PWF`/`OSF`/`RNF` (which failure mode occurred) are deliberately **excluded** — they're outcomes, and using them as inputs would be label leakage: you'd never know the failure mode before the failure happens.

**Included in this repo:** `data/raw/ai4i2020.csv` is committed directly (it's ~500KB and CC BY 4.0 licensed), so a fresh clone works with no download step. Everything else under `data/raw/` stays git-ignored - this is a deliberate, named exception in `.gitignore`, not a change to the general rule.

Source, for attribution: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

## Setup

Requires Python 3.10+ (3.12 recommended — the repo's `.venv` was created with it; the system Python on macOS is often too old for current numpy/pandas).

```bash
cd predictive-maintenance-agent

# Create and activate a virtual environment
uv venv --python 3.12 .venv          # or: python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt   # or: pip install -r requirements.txt
```

(Optional, for the LLM explanation step) copy `.env.example` to `.env` and add a real `ANTHROPIC_API_KEY`. Everything else works with no key configured — the explanation step is skipped gracefully, not a hard failure.

## How to train the model

```bash
python -m src.train_model
```

Trains **two** models with an identical train/test split (`stratify`d on the target, so the ~3.4% failure rate holds in both halves) — a raw-features baseline and a feature-engineered model — and prints both evaluations side by side, so the effect of feature engineering is visible, not assumed:

```
Baseline (raw features):
  precision: 0.691   recall: 0.691   f1: 0.691   roc_auc: 0.963
  confusion_matrix: [[1911, 21], [21, 47]]

Engineered (raw + engineered features):
  precision: 0.903   recall: 0.824   f1: 0.862   roc_auc: 0.979
  confusion_matrix: [[1926, 6], [12, 56]]
```
Saves `models/failure_risk_model.pkl` (engineered - used by prediction/API/UI) and `models/baseline_model.pkl` (kept for comparison).

**Engineered features** (`src/features.py`), each a physically-motivated combination rather than a raw reading:
- `temperature_difference` = process − air temperature (heat generated by work, not ambient drift)
- `power_proxy` = torque × rotational speed (proportional to mechanical load)
- `wear_torque_interaction` = tool wear × torque (a worn tool under load is worse than either alone)

**Feature importances** from the trained model (`model.feature_importances_`), most to least:

| Feature | Importance |
|---|---|
| Rotational speed [rpm] | 0.213 |
| power_proxy | 0.178 |
| Torque [Nm] | 0.174 |
| Tool wear [min] | 0.145 |
| wear_torque_interaction | 0.126 |
| temperature_difference | 0.088 |
| Air temperature [K] | 0.048 |
| Process temperature [K] | 0.028 |

Speed/torque/power together account for over half the model's decisions — consistent with mechanical load being the dominant failure driver in this dataset.

## How to run predictions

Directly in Python, for one telemetry snapshot:
```bash
python -m src.predict
```
```
Failure probability: 0.11
Risk band: Low
```

Risk bands (`src/predict.py`): **Low** < 0.4, **Medium** 0.4–0.7, **High** > 0.7. These thresholds are provisional starting points, not calibrated against real cost tradeoffs — see Limitations.

## How to run the API

```bash
uvicorn src.api:app --reload
```
Interactive docs (auto-generated from the Pydantic models): `http://127.0.0.1:8000/docs`

**`GET /health`**
```json
{"status": "ok", "model_loaded": true}
```

**`POST /predict`** — takes an `asset_id` and a telemetry snapshot; looks up that asset's context, runs the ML prediction, and returns a full deterministic decision:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "asset_id": "pump-01",
    "air_temperature": 302.0,
    "process_temperature": 312.0,
    "rotational_speed": 1350,
    "torque": 68.0,
    "tool_wear": 210
  }'
```

## Example input and output

The same telemetry snapshot (giving a 0.97 failure probability) sent for two different assets, to show that asset context — not just the risk score — drives the recommendation:

**`pump-01`** — High criticality, single point of failure, no redundancy:
```json
{
  "asset_id": "pump-01",
  "failure_probability": 0.97,
  "risk_band": "High",
  "asset_criticality": "High",
  "single_point_of_failure": true,
  "redundancy": "None",
  "recommended_action": "Immediate engineering review",
  "urgency": "Immediate",
  "human_review_required": true,
  "rationale_code": "HIGH_RISK_CRITICAL_SPOF_NO_REDUNDANCY"
}
```

**`fan-11`** — Low criticality, two redundant fans in the bank — **identical telemetry**:
```json
{
  "asset_id": "fan-11",
  "failure_probability": 0.97,
  "risk_band": "High",
  "asset_criticality": "Low",
  "single_point_of_failure": false,
  "redundancy": "2 additional fans in bank (fan-12, fan-13)",
  "recommended_action": "Inspect within 24-48 hours",
  "urgency": "High",
  "human_review_required": true,
  "rationale_code": "HIGH_RISK_STANDARD"
}
```
Same score, different consequence, different action.

## Streamlit UI

```bash
streamlit run src/ui.py
```
Lets you pick an example asset, enter telemetry, and see the risk score, risk band, recommended action, and (if `ANTHROPIC_API_KEY` is configured) an LLM-generated plain-language explanation. Runs the same `src/predict.py` / `src/decision_rules.py` logic in-process — see `src/ui.py`'s docstring for the tradeoff against calling the API instead, and what would change if this became a separately deployed frontend.

## How to run tests

```bash
pytest -q
```
279 tests total (53 V1 + 226 V2), covering every deterministic stage of both pipelines: data loading validation, feature engineering, evaluation metrics, risk-band/health-state thresholds, every branch of the decision rules (plus invalid-input errors), asset context lookup, the API (including `/health`, `/predict`, and its validation/404 error paths), and V2's engine-level splitting, temporal features, error analysis, uncertainty extraction, and the NASA test-set evaluation. The LLM explainer is tested with a mocked Anthropic client — real requests/response wiring is verified without a live API call (see Limitations).

## V2: Remaining Useful Life (NASA C-MAPSS)

V1 answers "will this fail?" as a yes/no classification from one telemetry snapshot. V2 answers a harder, more useful question: **"how many operating cycles does this engine have left?"** — a regression problem over a real time-series degradation dataset, built up in eleven deliberate steps rather than jumping straight to a finished model.

### Dataset

**NASA C-MAPSS** (Commercial Modular Aero-Propulsion System Simulation) — simulated turbofan jet engine run-to-failure data. This project uses the **FD001** subset only: one operating condition (sea-level flight), one fault mode (High-Pressure Compressor degradation), 100 training engines and 100 test engines. Each row is one flight cycle for one engine; a row has 3 operational settings and 21 raw sensor readings (temperatures, pressures, speeds, fuel flow around the engine core) — see `src/config.py`'s `SENSOR_METADATA` for what each one physically measures. Files are committed under `data/raw/cmapss/FD001/` so a fresh clone needs no download.

`train_FD001.txt` engines run to simulated failure — RUL is calculable directly (`max_cycle − current_cycle`). `test_FD001.txt` engines are deliberately truncated *before* failure; `RUL_FD001.txt` gives NASA's true remaining life measured from each engine's *last observed* test cycle. Step 11 (below) explains why these two files can't be treated the same way.

### Architecture

```
NASA training telemetry (train_FD001, 21 sensors x cycle)
        │
        ▼
Sensor screening (Step 4)         src/sensor_screening.py
  drop constant/near-constant sensors, using training data only
        │
        ▼
Feature engineering (Steps 5-6)   src/cmapss_features.py
  current-cycle features + causal rolling mean/std/delta/trend
  (never a future cycle, never crosses an engine boundary)
        │
        ▼
Random Forest regressor (Steps 5-6)   src/train_rul_temporal.py
  engine-level train/validation split, evaluated with MAE/RMSE (Step 7)
        │
        ▼
   predicted RUL
        │
        ├──────────────────────────────┐
        ▼                              ▼
Health state (Step 8)          Ensemble uncertainty (Step 10)
  HEALTHY / WATCH / PLAN /       src/uncertainty.py
  ACTION                         tree-disagreement -> model_confidence
        │                              │
        └──────────────┬───────────────┘
                        ▼
        Deterministic decision engine (Step 9)
        src/decision_engine.py - traceable, ID'd rules
                        ▲
                        │
        SYNTHETIC asset context: criticality,
        redundancy, maintenance lead time
                        │
                        ▼
        Maintenance action + rule-by-rule trace

Official FD001 test set (Step 11) - final, one-shot held-out evaluation
of everything above
```

**The same discipline as V1, extended:** the model only ever outputs a number; a fixed, testable lookup turns that number into a health state; a separate, fixed, testable rule set turns the health state *plus* asset context into an action. Neither the model nor a future LLM can skip a link in that chain or override what a rule decided.

### How to run each stage

```bash
python -m src.sensor_screening       # Step 4  - which of the 21 sensors carry information
python -m src.train_rul_baseline     # Step 5  - baseline RF, current-cycle features only
python -m src.train_rul_temporal     # Step 6  - + causal rolling/trend features
python -m src.run_error_analysis     # Step 7  - error by RUL band, over/under-prediction
python -m src.run_health_state       # Step 8  - RUL -> HEALTHY/WATCH/PLAN/ACTION
python -m src.run_decision_engine    # Step 9  - + synthetic asset context -> action
python -m src.run_uncertainty_analysis  # Step 10 - Random Forest tree-disagreement confidence
python -m src.run_test_evaluation    # Step 11 - final evaluation on the official NASA test set
```

Each script is also a runnable report: it prints its own explanation, tables and metrics, and — from Step 5 onward — saves outputs under `models/` (git-ignored) or `results/` / `artifacts/` (generated, not committed).

### Key results

**Model comparison** (validation set, 20 held-out engines, 4,070 cycle-level rows):

| Model | Features | Val MAE | Val RMSE |
|---|---|---|---|
| Baseline RF (Step 5) | 18 (current cycle only) | 10.84 | 15.97 |
| Temporal RF (Step 6) | 102 (+ rolling/delta/trend) | 9.95 | 14.94 |

The temporal model won everywhere, but by more near failure (critical-band MAE 2.77 vs. 3.89) than overall — a reminder not to assume an aggregate improvement transfers evenly (Step 7).

**Official NASA test set** (Step 11 — the one held-out, one-shot evaluation in this project):

| Dataset | n | MAE | RMSE |
|---|---|---|---|
| Validation (cycle-level) | 4,070 | 9.95 | 14.94 |
| **Official FD001 test** (engine-level, final cycle only) | 100 | **13.37** | **18.41** |

Test performance is materially worse than validation — the honest result, reported without retuning against it (see "Test-set discipline" below). Health-state accuracy on the test set: 0.82 overall, ACTION precision 1.00 / recall 0.80. Zero cases of "near failure, badly over-predicting, and the model's own trees were confidently agreeing" — the specific dangerous failure mode Step 10 was built to catch.

**Uncertainty** (Step 10): Random Forest tree-prediction disagreement correlates with actual error at 0.64 on validation — a real, checkable relationship, not an assumed one. It is called an "ensemble prediction range", never a "confidence interval": empirical coverage of the trees' own [p10, p90] range came out at 89.9% on validation and 83.0% on test, not a calibrated 80%, because the trees share training data and are not independent.

### Test-set discipline

`src/run_test_evaluation.py` is the *only* script that reads `test_FD001.txt` / `RUL_FD001.txt`, and it runs once, as a final exam, not a scratchpad — its own module docstring says so explicitly. If test performance disappoints, the correct response is to report it plainly, not to quietly retune sensor screening, features, or thresholds until the number improves; doing that turns the test set into just another validation set and the resulting score stops meaning what a held-out score is supposed to mean.

**Training / validation / test, in one line each:** training is revision (how well the model fits engines it studied); validation is the mock exam (used repeatedly, during development, to compare choices); test is the final exam (seen once, after every modelling decision was already locked in).

### What's NASA-validated vs. synthetic

- **NASA-supported** (evaluated against real ground truth): telemetry, sensor screening, RUL regression, prediction error, ensemble uncertainty, and health-state accuracy.
- **Synthetic** (invented for this learning project — NASA C-MAPSS contains none of it): asset criticality, redundancy, maintenance lead time, and therefore the final maintenance *recommendation* itself. `src/synthetic_asset_profiles.py` and every `AssetContext` in the decision-engine examples are clearly marked `source="SYNTHETIC_LEARNING_EXERCISE"` for exactly this reason. A real deployment would source these from an asset register/FMEA, a P&ID or knowledge graph, and a CMMS respectively.

### V2 limitations

- **FD001 only.** One operating condition, one fault mode — the simplest of C-MAPSS's four subsets. FD002–FD004 (multiple operating conditions/fault modes) are untouched.
- **Capped RUL.** The model is trained on `rul_capped` (ceilinged at 125 cycles) and cannot be meaningfully interpreted above that — test/validation evaluation compares against the capped target for this reason (see Step 11).
- **Ensemble disagreement ≠ full predictive uncertainty.** It reflects the trees' agreement with each other, not aleatoric noise, true epistemic gaps, out-of-distribution risk, or sensor-quality issues — see `src/uncertainty.py`'s module docstring.
- **Decision-engine thresholds and asset profiles are placeholders,** not calibrated against real cost/consequence data — same caveat as V1's risk bands, one level higher up the stack.
- **No LSTM/sequence model yet.** The regression model is tabular (Random Forest on engineered features), per CLAUDE.md's principle of establishing a scikit-learn baseline before deep learning.

## Pipeline Explorer (interactive V2 visualisation)

`app/` is a lightweight React app that walks one engine through the whole V2 pipeline, cycle by cycle, so you can see each stage's output and how it feeds the next:

1. **Telemetry in** - the 14 screened sensors over the engine's life, with the window the rolling features read and the future the model can't see
2. **Feature engineering** - the model's top features and their values at the current cycle
3. **Random Forest** - predicted vs actual RUL, the spread of the 200 trees' estimates, and feature importances
4. **Confidence** - tree disagreement mapped to HIGH / MEDIUM / LOW
5. **Health state** - the threshold lookup, predicted vs actual state across the whole life
6. **Decision engine** - the rule-by-rule trace and the recommended action across the whole life
7. **LLM reasoning** - a live explanation of the decision, plus the exact prompt it was given

**Editable:** the three health thresholds (ACTION / PLAN / WATCH) and the maintenance lead time. Every state and decision recomputes instantly. **Excluded:** asset criticality and redundancy - the explorer runs the decision engine with values under which those rules can never fire (`src/pipeline_explorer.py`). Confidence comes from the model, not a dropdown; the lead-time buffer is fixed at the engine default.

```bash
cd app && npm install && npm run dev        # http://localhost:5173 - works with no Python running
uvicorn src.api:app --port 8000             # optional, from the repo root: enables step 7 (needs ANTHROPIC_API_KEY in .env)
```

### Deploy to Vercel

The app deploys as a static Vite site plus one Vercel Function (`app/api/explain.ts`) for the LLM step. No Python runs on Vercel.

1. In Vercel, **Add New → Project** and import this GitHub repository.
2. Set **Root Directory** to `app`. The framework (Vite), build command and output directory come from `app/vercel.json`.
3. Under **Environment Variables**, add `ANTHROPIC_API_KEY`. Without it, step 7 still shows the decision and the exact prompt, but makes no LLM call.
4. Deploy. Every push to the connected branch redeploys.

The function mirrors the Python `POST /explain`: it re-derives the decision with the rules port before prompting, and its prompt is checked byte for byte against the Python prompt (`app/src/pipeline/prompt_cases.json`). Anyone who can open the deployment can spend your Anthropic key through step 7, so keep Vercel's Deployment Protection on, or leave the key unset for a public demo.

How it stays honest:

- **Real model output, frozen.** `python -m src.export_pipeline_fixture` trains the exact Step 6 temporal Random Forest on the Step 5 split and writes every cycle of all 100 `train_FD001` engines to `app/public/data/` (committed, ~5 MB, one file per engine). It reproduces the validation MAE 9.95 / RMSE 14.94 above.
- **In-sample engines are labelled.** 80 of the 100 engines trained the model, so their predictions look far better than the model is. The app opens on a held-out engine and flags in-sample ones.
- **Rules are ported, and checked.** Health states and decisions run in TypeScript (`app/src/pipeline/rules.ts`) so edits are instant. `app/src/pipeline/parity_cases.json` holds 810 boundary cases decided by the real Python engine; the app's tests must reproduce every one, and `tests/test_pipeline_explorer.py` fails if that file is stale. CI runs both.
- **The LLM explains a server-side decision, not the browser's.** Locally, `POST /explain` re-runs the decision in the Python engine; on Vercel, `api/explain.ts` re-runs it with the parity-tested rules port. Only then is the prompt built. The app shows whether the server agreed with the browser. Without an API key both still return the decision and the prompt.
- **One prompt, two implementations.** `app/src/pipeline/prompt.ts` must rebuild every prompt in `prompt_cases.json` (generated from `build_rul_explanation_prompt`) exactly, including Python's half-to-even rounding.

## Project structure

| Path | Purpose |
|---|---|
| `data/raw/` | Original telemetry. `ai4i2020.csv` is committed (see Dataset section); anything else here is git-ignored. |
| `data/processed/` | Reserved for data derived by code from `raw/`. |
| `data/assets/asset_context.json` | Example asset context: criticality, redundancy, failure modes, workarounds. |
| `models/` | Saved trained models (`.pkl`, git-ignored). |
| `notebooks/` | Exploration only - reusable logic lives in `src/`. |
| `src/config.py` | File paths and settings in one place. |
| `src/data_loader.py` | Load and validate raw telemetry. |
| `src/features.py` | Feature engineering, shared by training and prediction. |
| `src/train_model.py` | Train and compare the baseline vs. feature-engineered model. |
| `src/evaluate.py` | Precision/recall/F1/ROC AUC/confusion matrix. |
| `src/predict.py` | Load a saved model, score a telemetry snapshot, assign a risk band. |
| `src/decision_rules.py` | Deterministic maintenance actions, urgency, and escalation. |
| `src/asset_context.py` | Look up per-asset criticality, redundancy, failure modes. |
| `src/explainer.py` | LLM explanation - the only module that calls an LLM. |
| `src/agent.py` | Orchestrates the full pipeline for one asset. |
| `src/api.py` | FastAPI endpoints (`/health`, `/predict`). |
| `src/ui.py` | Streamlit UI. |
| `tests/` | 291 automated Python tests. |
| **Pipeline Explorer** | |
| `app/` | Interactive React + TypeScript visualisation of the V2 pipeline (see [Pipeline Explorer](#pipeline-explorer-interactive-v2-visualisation)). |
| `app/public/data/` | Committed model output for all 100 FD001 training engines. |
| `app/src/pipeline/rules.ts` | TypeScript port of the health-state and decision rules, checked against `parity_cases.json`. |
| `src/pipeline_explorer.py` | The explorer's decision context (criticality/redundancy excluded) and the parity-case generator. |
| `src/export_pipeline_fixture.py` | Trains Model B and exports the app's data. |
| **V2 (NASA C-MAPSS)** | |
| `data/raw/cmapss/FD001/` | NASA C-MAPSS FD001 train/test/RUL files, committed for a no-download clone. |
| `src/cmapss_loader.py` | Load and validate FD001's raw, header-less text files. |
| `src/rul.py` | Compute raw and capped RUL labels from run-to-failure training data. |
| `src/sensor_screening.py` | Step 4: flag constant/near-constant sensors from training data only. |
| `src/engine_split.py` | Leakage-safe train/validation split by `unit_number`, never by row. |
| `src/train_rul_baseline.py` | Step 5: current-cycle-only baseline Random Forest. |
| `src/cmapss_features.py` | Step 6: causal rolling mean/std/delta/trend features, per engine. |
| `src/train_rul_temporal.py` | Step 6: baseline + temporal-feature model comparison. |
| `src/evaluate_rul.py` | MAE/RMSE for RUL regression (version-agnostic RMSE). |
| `src/error_analysis.py` / `error_analysis_plots.py` | Step 7: error by RUL band, over/under-prediction, threshold diagnostics. |
| `src/health_state.py` / `health_state_plots.py` | Step 8: RUL → HEALTHY/WATCH/PLAN/ACTION, and evaluation against actual RUL. |
| `src/decision_engine.py` | Step 9: deterministic, traceable maintenance decision rules + asset context. |
| `src/synthetic_asset_profiles.py` | SYNTHETIC example asset contexts (criticality/redundancy/lead time) - NASA has none of this. |
| `src/uncertainty.py` / `uncertainty_plots.py` | Step 10: Random Forest tree-disagreement → `model_confidence`. |
| `src/evaluate_test_set.py` / `test_evaluation_plots.py` | Step 11: official FD001 test-set evaluation (loads saved model, never refits). |
| `src/nasa_score.py` | The PHM08/C-MAPSS asymmetric scoring function (secondary metric). |
| `src/run_*.py` | One runnable report per step (`run_error_analysis.py`, `run_health_state.py`, `run_decision_engine.py`, `run_uncertainty_analysis.py`, `run_test_evaluation.py`). |
| `results/` | Generated CSV/JSON outputs from Steps 7-11 (not committed - regenerate via the `run_*.py` scripts above). |

## Limitations

Being explicit about these matters as much as the working parts do:

- **Simulated data.** AI4I 2020 is a synthetic dataset; it's a good vehicle for learning the architecture, but a real deployment needs validation against real sensor data, which behaves messier (drift, missing readings, sensor faults) than this clean CSV.
- **Risk-band thresholds are arbitrary, not calibrated.** 0.4/0.7 were chosen as reasonable round numbers, not derived from this model's calibration or the real cost of a missed failure vs. a false alarm. `RandomForestClassifier.predict_proba` output isn't guaranteed to be a calibrated probability in the strict sense (see `CalibratedClassifierCV` for a fix) - it's a threshold-independent risk *ranking* (ROC AUC 0.979) more than a literal probability today.
- **Asset context is hand-authored example data,** not sourced from a real CMMS/EAM system. The five example assets in `data/assets/asset_context.json` are illustrative, not a real asset register.
- **Decision rules cover the cases specified so far,** not an exhaustive reliability-engineering rule set. They're deliberately simple, readable, and fully tested - real deployment would want a reliability engineer's sign-off on the rule table itself, not just its code correctness.
- **The LLM explanation step has integration-level risk that unit tests don't cover.** Tests confirm the code builds a correct prompt and handles the response correctly (with a mocked client) - they don't verify the live model's wording stays descriptive rather than drifting into making its own recommendations. Worth periodic manual review of real explanations against this.
- **Single-snapshot predictions only.** The model sees one moment of telemetry, not a trend - it can't yet tell "temperature is rising toward danger" from "temperature is stable near danger." That's a time-series problem (see Stage 14).
- **No persistence of live telemetry or prediction history.** Every prediction is stateless; nothing is logged or stored for later audit/trend analysis yet.

## Future roadmap

- [x] Stage 1: Project scaffold
- [x] Stage 2: Data loading
- [x] Stage 3: Data exploration
- [x] Stage 4: Baseline ML model
- [x] Stage 5: Model evaluation
- [x] Stage 6: Feature engineering
- [x] Stage 7: Model persistence
- [x] Stage 8: Prediction API
- [x] Stage 9: Maintenance decision rules
- [x] Stage 10: LLM explanation agent
- [x] Stage 11: Asset context
- [x] Stage 12: Simple UI
- [x] Stage 13: Tests and README
- [x] Stage 14: Upgrade to time-series remaining-useful-life (RUL) model — see [V2](#v2-remaining-useful-life-nasa-c-mapss) above:
  - [x] Step 1: Load and structure FD001
  - [x] Step 2: Calculate raw training RUL
  - [x] Step 3: Create capped RUL
  - [x] Step 4: Screen low-information sensors
  - [x] Step 5: Train baseline Random Forest (current-cycle features)
  - [x] Step 6: Add causal temporal (rolling/delta/trend) features
  - [x] Step 7: Operationally meaningful error analysis (RUL bands, over/under-prediction)
  - [x] Step 8: Convert predicted RUL into deterministic health states
  - [x] Step 9: Deterministic maintenance decision engine + synthetic asset context
  - [x] Step 10: Derive model confidence from Random Forest tree disagreement
  - [x] Step 11: Evaluate the full pipeline on the official NASA FD001 test set

Beyond V2's Step 11, worth considering next: an LLM explanation layer for V2 (Step 10's docstring already flags this — explain the decision, never override it); probability calibration (`CalibratedClassifierCV`) for V1 so risk-band thresholds mean what they claim; a real live-call integration test for the explainer, opt-in and skipped without an API key; persisting prediction history for trend analysis; and cost-based threshold tuning for both V1's risk bands and V2's decision-engine thresholds once real failure/false-alarm costs are known.
