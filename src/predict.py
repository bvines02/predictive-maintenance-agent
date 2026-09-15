"""Stage 7 - Prediction using a saved model.

Responsible for:
- Loading a trained model from models/
- Applying features.py to new telemetry
- Returning a risk score between 0 and 1

This is the only module the API and agent use to talk to the ML model.
"""

from pathlib import Path

import joblib
import pandas as pd

from src.features import ENGINEERED_FEATURE_COLUMNS, add_engineered_features
from src.train_model import MODEL_PATH

# Risk band thresholds on failure probability. Arbitrary starting points,
# not calibrated - see predict.py's module docstring / README for caveats.
LOW_RISK_MAX = 0.4
MEDIUM_RISK_MAX = 0.7


def load_model(path: Path = MODEL_PATH):
    """Load a trained model from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            "Train a model first - run: python -m src.train_model"
        )
    return joblib.load(path)


def risk_band(probability: float) -> str:
    """Convert a failure probability into a Low / Medium / High risk band.

    Thresholds (see module docstring): below 0.4 = Low, 0.4-0.7 = Medium,
    above 0.7 = High.
    """
    if probability < LOW_RISK_MAX:
        return "Low"
    if probability <= MEDIUM_RISK_MAX:
        return "Medium"
    return "High"


def predict_failure_risk(telemetry: dict, model=None) -> dict:
    """Predict failure risk for a single asset telemetry snapshot.

    Args:
        telemetry: A dict with the raw telemetry fields, e.g.:
            {
                "Air temperature [K]": 300.0,
                "Process temperature [K]": 310.0,
                "Rotational speed [rpm]": 1500,
                "Torque [Nm]": 40.0,
                "Tool wear [min]": 10,
            }
        model: A pre-loaded model. Loads MODEL_PATH if not given - pass one
            in when scoring many snapshots, to avoid reloading it each time.

    Returns:
        {"failure_probability": float, "risk_band": "Low" | "Medium" | "High"}
    """
    if model is None:
        model = load_model()

    # A one-row DataFrame so we can reuse the exact same feature pipeline
    # (add_engineered_features) that training used - see module docstring.
    df = pd.DataFrame([telemetry])
    df = add_engineered_features(df)

    X = df[ENGINEERED_FEATURE_COLUMNS]
    probability = float(model.predict_proba(X)[0, 1])

    return {
        "failure_probability": round(probability, 4),
        "risk_band": risk_band(probability),
    }


if __name__ == "__main__":
    # A representative snapshot for a quick manual check.
    example_telemetry = {
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.5,
        "Rotational speed [rpm]": 1400,
        "Torque [Nm]": 55.0,
        "Tool wear [min]": 190,
    }

    result = predict_failure_risk(example_telemetry)
    print(f"Input: {example_telemetry}")
    print(f"Failure probability: {result['failure_probability']}")
    print(f"Risk band: {result['risk_band']}")
