"""Stage 4 - Baseline ML model training.

Responsible for:
- Splitting data into training and test sets
- Training a scikit-learn classifier (starting with Random Forest)
- Saving the trained model to models/ (Stage 7)

Output is a failure-risk probability, not a maintenance decision.
"""

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.config import MODELS_DIR
from src.data_loader import load_telemetry
from src.evaluate import evaluate_model
from src.features import ENGINEERED_FEATURE_COLUMNS, RAW_FEATURE_COLUMNS, add_engineered_features

TARGET_COLUMN = "Machine failure"

MODEL_PATH = MODELS_DIR / "failure_risk_model.pkl"
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_model.pkl"


def train_model(
    df,
    feature_columns: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Train a Random Forest and evaluate it on a held-out test set.

    Args:
        df: Telemetry DataFrame containing feature_columns and TARGET_COLUMN.
        feature_columns: Which columns to use as model inputs.
        test_size: Fraction of rows held out for testing.
        random_state: Seed for the train/test split and the model, so runs
            are reproducible and comparable across feature sets.

    Returns:
        (model, metrics) - the trained classifier and its evaluation metrics.
    """
    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    # stratify=y keeps the ~3.4% failure rate consistent between the train
    # and test splits - without it, a random split could leave the test set
    # with too few failure examples to evaluate meaningfully.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(
        class_weight="balanced",
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)

    return model, metrics


def save_model(model, path: Path = MODEL_PATH) -> None:
    """Save a trained model to disk with joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def _print_metrics(label: str, metrics: dict) -> None:
    print(f"\n{label}:")
    for name, value in metrics.items():
        if name == "confusion_matrix":
            print(f"  confusion_matrix:\n{value}")
        else:
            print(f"  {name}: {value:.3f}")


if __name__ == "__main__":
    raw_df = load_telemetry()
    engineered_df = add_engineered_features(raw_df)

    # Same random_state and test_size for both, so the comparison isolates
    # the effect of the features rather than differences in the data split.
    baseline_model, baseline_metrics = train_model(raw_df, RAW_FEATURE_COLUMNS)
    engineered_model, engineered_metrics = train_model(engineered_df, ENGINEERED_FEATURE_COLUMNS)

    _print_metrics("Baseline (raw features)", baseline_metrics)
    _print_metrics("Engineered (raw + engineered features)", engineered_metrics)

    # The engineered model is the one the rest of the pipeline (predict.py,
    # the API) will load, since predict.py applies add_engineered_features
    # before scoring. Baseline is also saved so the comparison is reproducible
    # without retraining.
    save_model(engineered_model, MODEL_PATH)
    save_model(baseline_model, BASELINE_MODEL_PATH)
    print(f"\nEngineered model saved to {MODEL_PATH}")
    print(f"Baseline model saved to {BASELINE_MODEL_PATH}")
