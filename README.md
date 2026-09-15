# Predictive Maintenance Agent

A predictive maintenance decision-support system: it predicts equipment failure risk from telemetry, applies deterministic maintenance rules informed by asset context, and uses an LLM strictly to *explain* the resulting decision in maintenance language — with a human reviewer as the final step for anything high-risk.

This is a learning project built to practice real AI engineering architecture, not a notebook-only ML demo. The emphasis throughout is on keeping the machine-learned, the rule-based, and the generative parts of the system cleanly separated, testable independently, and safe to compose.

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

**To download it:**
1. https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
2. Download and unzip; place `ai4i2020.csv` at `data/raw/ai4i2020.csv`

`data/raw/` is git-ignored, so this step is required on every fresh clone.

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
53 tests, covering every deterministic stage of the pipeline: data loading validation, feature engineering, evaluation metrics, risk-band thresholds, every branch of the decision rules (plus invalid-input errors), asset context lookup, and the API (including `/health`, `/predict`, and its validation/404 error paths). The LLM explainer is tested with a mocked Anthropic client — real requests/response wiring is verified without a live API call (see Limitations).

## Project structure

| Path | Purpose |
|---|---|
| `data/raw/` | Original telemetry (git-ignored - download per the Dataset section). |
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
| `tests/` | 53 automated tests. |

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
- [ ] Stage 14: Upgrade to time-series remaining-useful-life (RUL) model (e.g. NASA C-MAPSS), predicting "how long until failure" instead of "will it fail"

Beyond Stage 14, worth considering next: probability calibration (`CalibratedClassifierCV`) so risk-band thresholds mean what they claim; a real live-call integration test for the explainer, opt-in and skipped without an API key; persisting prediction history for trend analysis; and cost-based threshold tuning once real failure/false-alarm costs are known.
