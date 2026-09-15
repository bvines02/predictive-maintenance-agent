"""Tests for src/predict.py.

Uses a tiny fake model instead of the real trained one, so tests are fast,
deterministic, and don't depend on models/failure_risk_model.pkl existing.
"""

import numpy as np
import pytest

from src.predict import predict_failure_risk, risk_band

SAMPLE_TELEMETRY = {
    "Air temperature [K]": 300.0,
    "Process temperature [K]": 310.0,
    "Rotational speed [rpm]": 1500,
    "Torque [Nm]": 40.0,
    "Tool wear [min]": 10,
}


class FixedProbabilityModel:
    """A fake model whose predict_proba always returns a fixed value."""

    def __init__(self, failure_probability: float):
        self._p = failure_probability

    def predict_proba(self, X):
        return np.array([[1 - self._p, self._p]])


@pytest.mark.parametrize(
    "probability, expected_band",
    [
        (0.0, "Low"),
        (0.39, "Low"),
        (0.4, "Medium"),
        (0.55, "Medium"),
        (0.7, "Medium"),
        (0.71, "High"),
        (1.0, "High"),
    ],
)
def test_risk_band_thresholds(probability, expected_band):
    assert risk_band(probability) == expected_band


def test_predict_failure_risk_returns_expected_shape():
    model = FixedProbabilityModel(failure_probability=0.62)

    result = predict_failure_risk(SAMPLE_TELEMETRY, model=model)

    assert result == {"failure_probability": 0.62, "risk_band": "Medium"}


def test_predict_failure_risk_applies_feature_engineering():
    """Confirms predict.py feeds the model engineered features, not just raw
    columns - if this broke, the model would silently see wrong inputs.
    """
    captured_columns = {}

    class RecordingModel:
        def predict_proba(self, X):
            captured_columns["columns"] = list(X.columns)
            return np.array([[0.9, 0.1]])

    predict_failure_risk(SAMPLE_TELEMETRY, model=RecordingModel())

    assert "power_proxy" in captured_columns["columns"]
    assert "temperature_difference" in captured_columns["columns"]
    assert "wear_torque_interaction" in captured_columns["columns"]
