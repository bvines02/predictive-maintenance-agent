"""Tests for src/api.py.

Uses FastAPI's TestClient, which runs the app in-process (via the lifespan
context manager) without a real network server - fast, and no port conflicts.
Runs against the real saved model, so it doubles as an integration check
that api.py and predict.py are wired together correctly.
"""

import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.fixture()
def client():
    # TestClient must be used as a context manager for FastAPI's lifespan
    # (startup/shutdown) to run - otherwise _load_model_on_startup never
    # fires and _model stays None.
    with TestClient(app) as test_client:
        yield test_client


def test_health_returns_ok_and_model_loaded(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_expected_shape(client):
    payload = {
        "asset_id": "pump-01",
        "air_temperature": 300.0,
        "process_temperature": 310.5,
        "rotational_speed": 1400,
        "torque": 55.0,
        "tool_wear": 190,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["asset_id"] == "pump-01"
    assert 0.0 <= body["failure_probability"] <= 1.0
    assert body["risk_band"] in {"Low", "Medium", "High"}
    # Enriched with asset context (pump-01 is High criticality, SPOF, no redundancy).
    assert body["asset_criticality"] == "High"
    assert body["single_point_of_failure"] is True
    assert body["redundancy"] == "None"
    # And run through decision_rules.py.
    assert body["urgency"] in {"Immediate", "High", "Medium", "Routine"}
    assert isinstance(body["human_review_required"], bool)
    assert body["rationale_code"]


def test_predict_unknown_asset_returns_404(client):
    payload = {
        "asset_id": "does-not-exist",
        "air_temperature": 300.0,
        "process_temperature": 310.5,
        "rotational_speed": 1400,
        "torque": 55.0,
        "tool_wear": 190,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 404
    assert "does-not-exist" in response.json()["detail"]


def test_predict_rejects_missing_field(client):
    payload = {
        "asset_id": "pump-01",
        "air_temperature": 300.0,
        # process_temperature missing
        "rotational_speed": 1400,
        "torque": 55.0,
        "tool_wear": 190,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422  # Pydantic validation error


def test_predict_rejects_wrong_type(client):
    payload = {
        "asset_id": "pump-01",
        "air_temperature": "not-a-number",
        "process_temperature": 310.5,
        "rotational_speed": 1400,
        "torque": 55.0,
        "tool_wear": 190,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


EXPLAIN_PAYLOAD = {
    "unit_number": 7,
    "time_cycles": 180,
    "predicted_rul": 12.0,
    "prediction_p10": 8.0,
    "prediction_p90": 19.5,
    "model_confidence": "HIGH",
    "health_thresholds": {"action": 15, "plan": 30, "watch": 60},
    "maintenance_lead_time_cycles": 15,
}


def test_explain_rederives_decision_in_python(client, monkeypatch):
    monkeypatch.setattr("src.api.explanation_available", lambda: True)
    monkeypatch.setattr("src.api.generate_rul_explanation", lambda prompt: "fake explanation")

    response = client.post("/explain", json=EXPLAIN_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["decision"]["recommended_action"] == "INTERVENE_NOW"
    assert body["decision"]["rule_ids"] == ["BASE_01", "LEAD_01"]
    assert "INTERVENE_NOW" in body["prompt"]
    assert body["explanation"] == "fake explanation"


def test_explain_without_key_still_returns_decision_and_prompt(client, monkeypatch):
    monkeypatch.setattr("src.api.explanation_available", lambda: False)

    body = client.post("/explain", json=EXPLAIN_PAYLOAD).json()

    assert body["explanation"] is None
    assert body["decision"]["health_state"] == "ACTION"
    assert body["prompt"]


def test_explain_rejects_unordered_thresholds(client):
    payload = {**EXPLAIN_PAYLOAD, "health_thresholds": {"action": 40, "plan": 30, "watch": 60}}
    assert client.post("/explain", json=payload).status_code == 422


def test_explain_rejects_unknown_confidence(client):
    assert client.post("/explain", json={**EXPLAIN_PAYLOAD, "model_confidence": "SURE"}).status_code == 422
