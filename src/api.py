"""Stage 8 - FastAPI prediction API.

Responsible for:
- Exposing HTTP endpoints (e.g. /health, /predict)
- Validating requests and responses with Pydantic models
- Calling predict.py / agent.py - no business logic lives here

An API lets any other system (a UI, a CMMS, a script) use the model,
which a notebook cannot do.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.asset_context import has_redundancy, load_asset_context
from src.decision_rules import decide_maintenance_action
from src.predict import load_model, predict_failure_risk

# Loaded once at startup, not per-request - see predict.py's docstring on
# why the model parameter exists.
_model = None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _model
    _model = load_model()
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
