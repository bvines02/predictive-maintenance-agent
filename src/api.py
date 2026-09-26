"""Stage 8 - FastAPI prediction API.

Responsible for:
- Exposing HTTP endpoints (e.g. /health, /predict)
- Validating requests and responses with Pydantic models
- Calling predict.py / agent.py - no business logic lives here
- POST /explain for the V2 Pipeline Explorer (app/): re-derives the RUL
  decision in Python, then asks the LLM to explain it

An API lets any other system (a UI, a CMMS, a script) use the model,
which a notebook cannot do.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.asset_context import has_redundancy, load_asset_context
from src.decision_rules import decide_maintenance_action
from src.explainer import build_rul_explanation_prompt, explanation_available, generate_rul_explanation
from src.pipeline_explorer import LEAD_TIME_BUFFER_CYCLES, decision_summary, explorer_decision
from src.predict import load_model, predict_failure_risk

# Loaded once at startup, not per-request - see predict.py's docstring on
# why the model parameter exists.
_model = None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _model
    # A missing V1 model disables /predict (503) rather than the whole API:
    # /explain (V2) does not use it.
    try:
        _model = load_model()
    except FileNotFoundError:
        _model = None
    yield


app = FastAPI(
    title="Predictive Maintenance Agent API",
    description="Predicts equipment failure risk from a telemetry snapshot.",
    lifespan=_lifespan,
)


class PredictRequest(BaseModel):
    """A single asset's telemetry snapshot."""

    asset_id: str
    air_temperature: float = Field(..., description="Air temperature, Kelvin")
    process_temperature: float = Field(..., description="Process temperature, Kelvin")
    rotational_speed: float = Field(..., description="Rotational speed, rpm")
    torque: float = Field(..., description="Torque, Nm")
    tool_wear: float = Field(..., description="Tool wear, minutes")


class PredictResponse(BaseModel):
    """The full risk assessment and maintenance decision for one asset.

    Combines three things that each come from a different, separately
    testable module - the ML prediction (predict.py), the asset's fixed
    context (asset_context.py), and the deterministic decision (decision_rules.py).
    """

    asset_id: str
    failure_probability: float
    risk_band: str

    # From asset context - not derived from telemetry, looked up per asset_id.
    asset_criticality: str
    single_point_of_failure: bool
    redundancy: str

    # From decision_rules.py - the actual recommendation.
    recommended_action: str
    urgency: str
    human_review_required: bool
    rationale_code: str


@app.get("/health")
def health() -> dict:
    """Liveness check - confirms the API is up and the model is loaded."""
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Predict failure risk for one asset and turn it into a maintenance decision.

    Pipeline: telemetry -> predict.py (risk score) -> asset_context.py
    (who/what is this asset) -> decision_rules.py (what to do about it).
    No step is skipped or reordered - see CLAUDE.md's "Engineering principles".
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="V1 model not trained - run `python -m src.train_model`.")
    try:
        asset = load_asset_context(request.asset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # Translate the API's field names to the raw column names predict.py
    # (and the model) expects - keeping that mapping here, not in predict.py,
    # keeps predict.py agnostic to how callers name their fields.
    telemetry = {
        "Air temperature [K]": request.air_temperature,
        "Process temperature [K]": request.process_temperature,
        "Rotational speed [rpm]": request.rotational_speed,
        "Torque [Nm]": request.torque,
        "Tool wear [min]": request.tool_wear,
    }

    prediction = predict_failure_risk(telemetry, model=_model)

    decision = decide_maintenance_action(
        failure_probability=prediction["failure_probability"],
        risk_band=prediction["risk_band"],
        asset_criticality=asset["criticality"],
        is_single_point_of_failure=asset["single_point_of_failure"],
        redundancy_available=has_redundancy(asset),
    )

    return PredictResponse(
        asset_id=request.asset_id,
        failure_probability=prediction["failure_probability"],
        risk_band=prediction["risk_band"],
        asset_criticality=asset["criticality"],
        single_point_of_failure=asset["single_point_of_failure"],
        redundancy=asset["redundancy"],
        recommended_action=decision["recommended_action"],
        urgency=decision["urgency"],
        human_review_required=decision["human_review_required"],
        rationale_code=decision["rationale_code"],
    )


class HealthThresholds(BaseModel):
    action: float
    plan: float
    watch: float


class ExplainRequest(BaseModel):
    """What the Pipeline Explorer sends: the model's output and the policy in force.

    Deliberately NOT the browser's decision. /explain re-derives the decision
    with the real Python engine, so the LLM only ever explains a decision the
    deterministic rules made - a client cannot hand it one to rationalise.
    """

    unit_number: int
    time_cycles: int
    predicted_rul: float
    prediction_p10: float
    prediction_p90: float
    model_confidence: str = Field(..., description="HIGH / MEDIUM / LOW, from tree disagreement")
    health_thresholds: HealthThresholds
    maintenance_lead_time_cycles: float = Field(..., ge=0)


class ExplainResponse(BaseModel):
    decision: dict  # pipeline_explorer.decision_summary() - the Python engine's result
    prompt: str  # exactly what the LLM received (or would receive)
    explanation: str | None  # None when no API key is configured


@app.post("/explain", response_model=ExplainResponse)
def explain(request: ExplainRequest) -> ExplainResponse:
    """V2: decide (Python rules) -> build prompt -> LLM explains. The LLM step is last and read-only."""
    thresholds = request.health_thresholds.model_dump()
    try:
        decision = decision_summary(
            explorer_decision(
                request.predicted_rul,
                request.model_confidence,
                thresholds,
                request.maintenance_lead_time_cycles,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    prompt = build_rul_explanation_prompt(
        decision,
        {
            "unit_number": request.unit_number,
            "time_cycles": request.time_cycles,
            "prediction_p10": request.prediction_p10,
            "prediction_p90": request.prediction_p90,
        },
        thresholds,
        LEAD_TIME_BUFFER_CYCLES,
    )
    # No key: still return the Python decision and the prompt, so the
    # explorer can show what the LLM would receive and check parity.
    explanation = generate_rul_explanation(prompt) if explanation_available() else None
    return ExplainResponse(decision=decision, prompt=prompt, explanation=explanation)
