"""Stage 14e - Baseline RUL regression model.

Responsible for:
- Building the modelling dataset (current-cycle features -> rul_capped)
- Splitting train_FD001 into training/validation engines (no leakage)
- Training a baseline RandomForestRegressor
- Evaluating it with MAE/RMSE on train and validation
- Producing a per-row prediction table and a per-engine prediction plot
- Reporting feature importance as an exploratory diagnostic
- Saving the trained model and the exact feature list it was trained on

Not responsible for (see V2 principles in CLAUDE.md - these come later):
- Rolling / lagged time-series features
- Hyperparameter tuning
- Evaluating against the official C-MAPSS test set (test_FD001 + RUL_FD001)
- Decision rules, LLM explanation, API, agent

This step uses ONLY the current cycle's own telemetry to predict
rul_capped - no history, no rolling windows. That is a deliberately weak
baseline: it cannot see a trend, only a single instant. The point of a
baseline is to have an honest floor to compare later, more sophisticated
feature sets against - if a rolling-feature model claims a big improvement,
we want a real "before" number to check that against, not an assumption.

Leakage risk in this step specifically: beyond the engine-level split
handled by src/engine_split.py, note that sensor screening (src/
sensor_screening.py) is re-run here on this same train_FD001 DataFrame,
BEFORE the engine split. That is intentional, not a leak: the C-MAPSS
"train" file is still only training data relative to the untouched
official test set (test_FD001 / RUL_FD001), which this step does not
touch. It would become a leak only if we screened sensors using
test_FD001 or using the validation engines carved out below.

How this differs from a real industrial deployment: here, engines are
independent turbofan units simulated once and never revisited. In a real
fleet, the same physical asset generates new data every day, so a
production system retrains/re-validates on a rolling basis and must guard
against a subtler version of this same leakage - e.g. accidentally using
an asset's own future sensor readings (from after the prediction date) when
building historical features for it.
"""

import json

import joblib
import matplotlib

matplotlib.use("Agg")  # headless: save figures to disk, never open a window
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.cmapss_loader import (
    OPERATIONAL_SETTING_COLUMNS,
    SENSOR_COLUMNS,
    load_train_fd001,
    validate_cmapss_dataframe,
)
from src.config import (
    RUL_BASELINE_ARTIFACTS_DIR,
    RUL_BASELINE_FEATURES_PATH,
    RUL_BASELINE_MODEL_PATH,
)
from src.engine_split import split_dataframe_by_unit, split_units, validate_engine_split
from src.evaluate_rul import evaluate_rul_predictions
from src.rul import add_capped_rul, add_rul_target
from src.sensor_screening import STATUS_KEEP, screen_sensors, sensors_with_status

TARGET_COLUMN = "rul_capped"
NON_FEATURE_COLUMNS = {"unit_number", "rul", "rul_capped"}

RANDOM_STATE = 42
VALIDATION_FRACTION = 0.2

# Whether to include time_cycles as a baseline feature - see the module
# docstring / project explanation for the reasoning. We include it for
# this first baseline, then revisit once rolling features exist.
INCLUDE_TIME_CYCLES = True


def get_kept_sensor_columns(train_df: pd.DataFrame) -> list[str]:
    """The Step 4 sensor screening result: sensors worth keeping, from training data only.

    Re-run here (rather than hard-coded) so the result always reflects
    actual data - if the training data changes, this list updates with it.
    Shared by both the baseline (this module) and the temporal feature
    step (src/train_rul_temporal.py), so both models start from the exact
    same sensor set - see the module docstring's leakage note on why this
    must only ever run on training data.
    """
    sensor_columns = [col for col in SENSOR_COLUMNS if col in train_df.columns]
    screening_table = screen_sensors(train_df, sensor_columns=sensor_columns)
    return sensors_with_status(screening_table, STATUS_KEEP)


def build_feature_columns(train_df: pd.DataFrame) -> list[str]:
    """Decide the baseline model's feature list: current-cycle features only."""
    kept_sensors = get_kept_sensor_columns(train_df)

    feature_columns = list(OPERATIONAL_SETTING_COLUMNS) + kept_sensors
    if INCLUDE_TIME_CYCLES:
        feature_columns = ["time_cycles"] + feature_columns

    for excluded in NON_FEATURE_COLUMNS:
        if excluded in feature_columns:
            raise AssertionError(f"{excluded} must never be a model feature.")

    return feature_columns


def build_modeling_dataset(
    df: pd.DataFrame, feature_columns: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into X (features) and y (rul_capped)."""
    X = df[feature_columns]
    y = df[TARGET_COLUMN]
    return X, y


def train_baseline_rf(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = RANDOM_STATE
) -> RandomForestRegressor:
    """Train a baseline RandomForestRegressor with a straightforward config.

    n_estimators=200: enough trees for stable averaging without tuning.
    random_state=42: reproducible bootstrap sampling and feature subsets.
    n_jobs=-1: use all available CPU cores, since trees train independently.
    No max_depth/min_samples_leaf tuning yet - that is a later step.
    """
    model = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def build_predictions_df(val_df: pd.DataFrame, y_pred) -> pd.DataFrame:
    """Build a tidy per-row prediction table for the validation set.

    error = predicted_rul - actual_rul: positive means the model predicted
    MORE remaining life than the engine actually had left (an
    over-optimistic, potentially dangerous mistake); negative means it
    predicted less (over-cautious, safer but less economical).
    """
    predictions = pd.DataFrame(
        {
            "unit_number": val_df["unit_number"].to_numpy(),
            "time_cycles": val_df["time_cycles"].to_numpy(),
            "actual_rul": val_df[TARGET_COLUMN].to_numpy(),
            "predicted_rul": y_pred,
        }
    )
    predictions["error"] = predictions["predicted_rul"] - predictions["actual_rul"]
    return predictions


def get_feature_importance(
    model: RandomForestRegressor, feature_columns: list[str], top_n: int = 10
) -> pd.DataFrame:
    """Top-`top_n` features by Random Forest importance, most important first.

    This is an exploratory diagnostic only. It measures how much each
    feature reduced prediction error INSIDE this particular model, on this
    particular data - it does not measure real-world engineering
    importance or causality. A sensor could rank high because it is
    genuinely diagnostic, or simply because it is correlated with a truly
    causal sensor, or an artifact of how these particular trees happened to
    split. Treat it as a hint for where to look next, not a conclusion.
    """
    importance_df = pd.DataFrame(
        {"feature": feature_columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    return importance_df.head(top_n).reset_index(drop=True)


def plot_engine_predictions(predictions_df: pd.DataFrame, unit_number: int, save_path) -> None:
    """Plot actual vs. predicted RUL over time_cycles for one engine."""
    engine_predictions = predictions_df[predictions_df["unit_number"] == unit_number].sort_values(
        "time_cycles"
    )

    plt.figure(figsize=(8, 5))
    plt.plot(
        engine_predictions["time_cycles"],
        engine_predictions["actual_rul"],
        label="Actual RUL (capped)",
        color="black",
    )
    plt.plot(
        engine_predictions["time_cycles"],
        engine_predictions["predicted_rul"],
        label="Predicted RUL",
        color="tab:orange",
        linestyle="--",
    )
    plt.xlabel("time_cycles")
    plt.ylabel("RUL")
    plt.title(f"Engine {unit_number}: actual vs. predicted RUL")
    plt.legend()
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def save_model_and_features(
    model: RandomForestRegressor,
    feature_columns: list[str],
    model_path=RUL_BASELINE_MODEL_PATH,
    features_path=RUL_BASELINE_FEATURES_PATH,
) -> None:
    """Save the trained model and the exact feature list/order it expects.

    The feature list is saved alongside the model - not just documented in
    code - because future inference must build a DataFrame with these exact
    columns in this exact order. If the feature list ever changes, saving
    it prevents a silent mismatch between "what the model was trained on"
    and "what we hand it at prediction time".
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    features_path.parent.mkdir(parents=True, exist_ok=True)
    features_path.write_text(json.dumps(feature_columns, indent=2))


if __name__ == "__main__":
    train_df = load_train_fd001()
    validate_cmapss_dataframe(train_df, "train_FD001")

    train_df = add_rul_target(train_df)
    train_df = add_capped_rul(train_df)

    feature_columns = build_feature_columns(train_df)
    print(f"{'=' * 70}\nFinal feature list ({len(feature_columns)} features)\n{'=' * 70}")
    for column in feature_columns:
        print(f"  - {column}")

    # --- Engine-level split ---------------------------------------------
    train_units, val_units = split_units(
        train_df["unit_number"], test_size=VALIDATION_FRACTION, random_state=RANDOM_STATE
    )
    train_split_df, val_split_df = split_dataframe_by_unit(train_df, train_units, val_units)
    validate_engine_split(train_df, train_split_df, val_split_df, train_units, val_units)

    print(f"\n{'=' * 70}\nEngine-level train/validation split\n{'=' * 70}")
    print(f"Total engines: {train_df['unit_number'].nunique()}")
    print(f"Training engines: {len(train_units)}")
    print(f"Validation engines: {len(val_units)}")
    print(f"Training rows: {len(train_split_df)}")
    print(f"Validation rows: {len(val_split_df)}")

    X_train, y_train = build_modeling_dataset(train_split_df, feature_columns)
    X_val, y_val = build_modeling_dataset(val_split_df, feature_columns)
    assert TARGET_COLUMN not in X_train.columns and TARGET_COLUMN not in X_val.columns

    # --- Train -------------------------------------------------------------
    model = train_baseline_rf(X_train, y_train)

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_metrics = evaluate_rul_predictions(y_train, train_pred)
    val_metrics = evaluate_rul_predictions(y_val, val_pred)

    print(f"\n{'=' * 70}\nBaseline Random Forest\n{'=' * 70}")
    print(f"Train MAE:  {train_metrics['mae']:.2f}")
    print(f"Train RMSE: {train_metrics['rmse']:.2f}")
    print(f"\nValidation MAE:  {val_metrics['mae']:.2f}")
    print(f"Validation RMSE: {val_metrics['rmse']:.2f}")

    # --- Predictions table ---------------------------------------------
    predictions_df = build_predictions_df(val_split_df, val_pred)

    print(f"\n{'=' * 70}\nSample validation predictions\n{'=' * 70}")
    sample_unit = predictions_df["unit_number"].iloc[0]
    print(predictions_df[predictions_df["unit_number"] == sample_unit].head(10).to_string(index=False))

    # --- Plot one validation engine -------------------------------------
    plot_unit = int(sorted(val_units)[0])
    plot_path = RUL_BASELINE_ARTIFACTS_DIR / f"engine_{plot_unit}_predictions.png"
    plot_engine_predictions(predictions_df, plot_unit, plot_path)
    print(f"\nSaved prediction plot for engine {plot_unit} to {plot_path}")

    # --- Feature importance ---------------------------------------------
    importance_df = get_feature_importance(model, feature_columns)
    print(f"\n{'=' * 70}\nTop 10 feature importances (exploratory only)\n{'=' * 70}")
    print(importance_df.to_string(index=False))

    # --- Save -------------------------------------------------------------
    save_model_and_features(model, feature_columns)
    print(f"\nModel saved to {RUL_BASELINE_MODEL_PATH}")
    print(f"Feature list saved to {RUL_BASELINE_FEATURES_PATH}")
