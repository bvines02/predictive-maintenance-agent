"""Stage 14g (model) - temporal-feature RUL model vs. the Step 5 baseline.

Responsible for:
- Adding rolling/trend features (src/cmapss_features.py) on top of the
  retained sensors from Step 4
- Training a second RandomForestRegressor ("Model B") on current-cycle +
  temporal features
- Re-training the unchanged Step 5 baseline ("Model A") on the SAME
  engine-level split, so the two models are compared fairly
- Reporting a side-by-side comparison, an example-engine plot, and
  temporal-model feature importances
- Saving the temporal model and its exact feature list separately from
  the baseline

Not responsible for (see V2 principles in CLAUDE.md - these come later):
- Hyperparameter tuning
- XGBoost / neural networks
- Evaluating against the official C-MAPSS test set
- Decision rules, LLM explanation, API, agent

Model A is NOT loaded from the Step 5 joblib file - it is re-trained here,
using src.train_rul_baseline's own unmodified functions, on the identical
engine split used for Model B. This is deliberate: split_units() is a pure
function of (unit_number values, test_size, random_state), all three of
which are unchanged from Step 5, so it produces the exact same train/val
engine partition every time - re-deriving it is equivalent to reusing a
saved split, without needing to persist split indices to disk. Re-training
Model A from the same unmodified code also means this comparison can never
silently drift from what Step 5 actually did.

Leakage risk specific to this step: the temporal features
(src/cmapss_features.py) are computed on the FULL train_FD001 DataFrame,
BEFORE the engine split below. This is safe, not a leak, because every
rolling/delta/trend value for engine E's cycle t is a function only of
engine E's own cycles <= t (enforced inside add_rolling_features) - it
never uses another engine's data or a later cycle, regardless of which
side of the train/validation split engine E ends up on. Computing the
split first and features second would give an identical result, just
slower (features would be computed twice, once per split); computing
features first is safe specifically because they are per-engine and
causal by construction.

How this differs from real industrial condition monitoring: FD001's
"rolling window" is a fixed number of simulation cycles. In a real plant,
the equivalent windows are wall-clock time (e.g. "last 24 hours", "last 7
days") over irregularly-sampled sensor tags, often with missing readings,
sensor replacements mid-life, and multiple simultaneous operating modes -
all of which complicate what "the last 5 cycles" even means in a way this
clean simulated dataset does not have to face.
"""

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.cmapss_features import add_rolling_features, rolling_feature_names
from src.cmapss_loader import load_train_fd001, validate_cmapss_dataframe
from src.config import ARTIFACTS_DIR, MODELS_DIR
from src.engine_split import split_dataframe_by_unit, split_units, validate_engine_split
from src.evaluate_rul import evaluate_rul_predictions
from src.rul import add_capped_rul, add_rul_target
from src.train_rul_baseline import (
    RANDOM_STATE,
    TARGET_COLUMN,
    VALIDATION_FRACTION,
    build_feature_columns,
    build_modeling_dataset,
    build_predictions_df,
    get_kept_sensor_columns,
    train_baseline_rf,
)

ROLLING_WINDOWS = (5, 10)
TREND_WINDOW = 5

TEMPORAL_MODEL_PATH = MODELS_DIR / "random_forest_temporal.joblib"
TEMPORAL_FEATURES_PATH = MODELS_DIR / "random_forest_temporal_features.json"
TEMPORAL_ARTIFACTS_DIR = ARTIFACTS_DIR / "rul_temporal"


def build_temporal_feature_columns(train_df: pd.DataFrame, kept_sensors: list[str]) -> list[str]:
    """Model B's feature list: every baseline feature + every rolling/trend feature.

    Built explicitly (rather than inferring it from the featured DataFrame's
    columns) so the order is guaranteed to match `rolling_feature_names`,
    which is what actually gets computed - see that function's docstring.
    """
    baseline_columns = build_feature_columns(train_df)
    temporal_columns = rolling_feature_names(
        kept_sensors, windows=ROLLING_WINDOWS, trend_window=TREND_WINDOW
    )
    return baseline_columns + temporal_columns


def classify_feature(feature_name: str) -> str:
    """Label a feature column as raw / rolling mean / rolling std / delta / trend.

    Exploratory labelling only, used to make the feature-importance table
    easier to read - it does not affect training.
    """
    if "_roll_mean_" in feature_name:
        return "rolling_mean"
    if "_roll_std_" in feature_name:
        return "rolling_std"
    if "_delta_" in feature_name:
        return "delta"
    if "_trend_" in feature_name:
        return "trend"
    return "raw"


def get_feature_importance_by_category(
    model: RandomForestRegressor, feature_columns: list[str], top_n: int = 15
) -> pd.DataFrame:
    """Top-`top_n` features by importance, each labelled with its category.

    Same exploratory-only caveat as src/train_rul_baseline.py's version:
    this measures how much each feature reduced error inside THIS model on
    THIS data, not real-world engineering importance or causality.
    """
    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "category": [classify_feature(f) for f in feature_columns],
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return importance_df.head(top_n).reset_index(drop=True)


def compare_models(
    model_a_name: str,
    model_a_metrics: dict,
    model_b_name: str,
    model_b_metrics: dict,
) -> pd.DataFrame:
    """Build the Model A vs. Model B comparison table, plus % change (B vs. A)."""
    rows = [
        {
            "model": model_a_name,
            "train_mae": model_a_metrics["train"]["mae"],
            "train_rmse": model_a_metrics["train"]["rmse"],
            "val_mae": model_a_metrics["val"]["mae"],
            "val_rmse": model_a_metrics["val"]["rmse"],
        },
        {
            "model": model_b_name,
            "train_mae": model_b_metrics["train"]["mae"],
            "train_rmse": model_b_metrics["train"]["rmse"],
            "val_mae": model_b_metrics["val"]["mae"],
            "val_rmse": model_b_metrics["val"]["rmse"],
        },
    ]
    return pd.DataFrame(rows)


def improvement(baseline_value: float, new_value: float) -> tuple[float, float]:
    """Absolute and percentage change of `new_value` vs. `baseline_value`.

    Negative = improvement (lower error). Positive = the new model did
    worse. Reported as a signed number, deliberately not "improvement" in
    the name - see requirement 19, "do not assume the temporal model is
    better".
    """
    absolute_change = new_value - baseline_value
    percent_change = 100 * absolute_change / baseline_value
    return absolute_change, percent_change


def plot_model_comparison(
    predictions_a: pd.DataFrame, predictions_b: pd.DataFrame, unit_number: int, save_path
) -> None:
    """Plot actual vs. baseline-predicted vs. temporal-predicted RUL for one engine."""
    engine_a = predictions_a[predictions_a["unit_number"] == unit_number].sort_values(
        "time_cycles"
    )
    engine_b = predictions_b[predictions_b["unit_number"] == unit_number].sort_values(
        "time_cycles"
    )

    plt.figure(figsize=(8, 5))
    plt.plot(engine_a["time_cycles"], engine_a["actual_rul"], label="Actual RUL (capped)", color="black")
    plt.plot(
        engine_a["time_cycles"],
        engine_a["predicted_rul"],
        label="Baseline prediction (current-cycle)",
        color="tab:orange",
        linestyle="--",
    )
    plt.plot(
        engine_b["time_cycles"],
        engine_b["predicted_rul"],
        label="Temporal-feature prediction",
        color="tab:blue",
        linestyle=":",
    )
    plt.xlabel("time_cycles")
    plt.ylabel("RUL")
    plt.title(f"Engine {unit_number}: baseline vs. temporal-feature prediction")
    plt.legend()
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def save_temporal_model(
    model: RandomForestRegressor,
    feature_columns: list[str],
    model_path=TEMPORAL_MODEL_PATH,
    features_path=TEMPORAL_FEATURES_PATH,
) -> None:
    """Save the temporal model and its feature list, separately from the baseline."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    features_path.parent.mkdir(parents=True, exist_ok=True)
    features_path.write_text(json.dumps(feature_columns, indent=2))


if __name__ == "__main__":
    train_df = load_train_fd001()
    validate_cmapss_dataframe(train_df, "train_FD001")
    train_df = add_rul_target(train_df)
    train_df = add_capped_rul(train_df)

    kept_sensors = get_kept_sensor_columns(train_df)

    # --- Worked example: one sensor, one engine, before any split -------
    example_df = add_rolling_features(train_df, kept_sensors, windows=ROLLING_WINDOWS, trend_window=TREND_WINDOW)
    print(f"{'=' * 70}\nWorked example - engine 1, sensor_11, cycles 1-10\n{'=' * 70}")
    example_columns = [
        "unit_number",
        "time_cycles",
        "sensor_11",
        "sensor_11_roll_mean_5",
        "sensor_11_roll_std_5",
        "sensor_11_delta_1",
        "sensor_11_trend_5",
    ]
    engine_1 = example_df[example_df["unit_number"] == 1]
    print(engine_1[example_columns].head(10).to_string(index=False))

    # --- Feature lists ----------------------------------------------------
    baseline_feature_columns = build_feature_columns(train_df)
    temporal_feature_columns = build_temporal_feature_columns(train_df, kept_sensors)

    print(f"\n{'=' * 70}\nFeature counts\n{'=' * 70}")
    print(f"Model A (baseline, current-cycle only): {len(baseline_feature_columns)} features")
    print(f"Model B (baseline + temporal):          {len(temporal_feature_columns)} features")

    # --- SAME engine-level split as Step 5 -------------------------------
    featured_df = add_rolling_features(
        train_df, kept_sensors, windows=ROLLING_WINDOWS, trend_window=TREND_WINDOW
    )
    train_units, val_units = split_units(
        featured_df["unit_number"], test_size=VALIDATION_FRACTION, random_state=RANDOM_STATE
    )
    train_split_df, val_split_df = split_dataframe_by_unit(featured_df, train_units, val_units)
    validate_engine_split(featured_df, train_split_df, val_split_df, train_units, val_units)

    print(f"\n{'=' * 70}\nEngine-level split (same as Step 5)\n{'=' * 70}")
    print(f"Training engines: {len(train_units)}, validation engines: {len(val_units)}")
    print(f"Training rows: {len(train_split_df)}, validation rows: {len(val_split_df)}")

    # --- Model A: re-trained baseline, on the identical split ------------
    X_train_a, y_train = build_modeling_dataset(train_split_df, baseline_feature_columns)
    X_val_a, y_val = build_modeling_dataset(val_split_df, baseline_feature_columns)
    model_a = train_baseline_rf(X_train_a, y_train)

    model_a_metrics = {
        "train": evaluate_rul_predictions(y_train, model_a.predict(X_train_a)),
        "val": evaluate_rul_predictions(y_val, model_a.predict(X_val_a)),
    }

    # --- Model B: baseline + temporal features ---------------------------
    X_train_b, _ = build_modeling_dataset(train_split_df, temporal_feature_columns)
    X_val_b, _ = build_modeling_dataset(val_split_df, temporal_feature_columns)
    model_b = train_baseline_rf(X_train_b, y_train)

    model_b_metrics = {
        "train": evaluate_rul_predictions(y_train, model_b.predict(X_train_b)),
        "val": evaluate_rul_predictions(y_val, model_b.predict(X_val_b)),
    }

    # --- Comparison table ---------------------------------------------
    comparison = compare_models("Baseline RF", model_a_metrics, "Temporal-feature RF", model_b_metrics)
    print(f"\n{'=' * 70}\nModel comparison\n{'=' * 70}")
    print(comparison.to_string(index=False))

    mae_abs, mae_pct = improvement(model_a_metrics["val"]["mae"], model_b_metrics["val"]["mae"])
    rmse_abs, rmse_pct = improvement(model_a_metrics["val"]["rmse"], model_b_metrics["val"]["rmse"])
    print(f"\nValidation MAE change:  {mae_abs:+.2f} ({mae_pct:+.1f}%)")
    print(f"Validation RMSE change: {rmse_abs:+.2f} ({rmse_pct:+.1f}%)")
    print("(negative = temporal model did better; positive = it did worse)")

    # --- Overfitting check ------------------------------------------------
    gap_a = model_a_metrics["val"]["mae"] - model_a_metrics["train"]["mae"]
    gap_b = model_b_metrics["val"]["mae"] - model_b_metrics["train"]["mae"]
    print(f"\n{'=' * 70}\nTrain/validation MAE gap (overfitting check)\n{'=' * 70}")
    print(f"Model A gap: {gap_a:.2f}")
    print(f"Model B gap: {gap_b:.2f}")

    # --- Predictions + example engine plot -------------------------------
    predictions_a = build_predictions_df(val_split_df, model_a.predict(X_val_a))
    predictions_b = build_predictions_df(val_split_df, model_b.predict(X_val_b))

    plot_unit = int(sorted(val_units)[0])
    plot_path = TEMPORAL_ARTIFACTS_DIR / f"engine_{plot_unit}_comparison.png"
    plot_model_comparison(predictions_a, predictions_b, plot_unit, plot_path)
    print(f"\nSaved comparison plot for engine {plot_unit} to {plot_path}")

    # --- Feature importance ---------------------------------------------
    importance_df = get_feature_importance_by_category(model_b, temporal_feature_columns, top_n=15)
    print(f"\n{'=' * 70}\nTop 15 feature importances - Model B (exploratory only)\n{'=' * 70}")
    print(importance_df.to_string(index=False))

    # --- Save ---------------------------------------------------------
    save_temporal_model(model_b, temporal_feature_columns)
    print(f"\nTemporal model saved to {TEMPORAL_MODEL_PATH}")
    print(f"Temporal feature list saved to {TEMPORAL_FEATURES_PATH}")
