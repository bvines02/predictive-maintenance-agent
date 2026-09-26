"""Stage 14h - Step 11: evaluate the final model on the official FD001 test set.

Responsible for:
- Loading test_FD001 + RUL_FD001 and checking they actually line up
- Recreating the exact training feature pipeline on test data (temporal
  features computed causally, per engine, from the saved feature list -
  nothing re-fit or re-selected on test data)
- Selecting each test engine's FINAL observed cycle - the one point where
  RUL_FD001's label applies
- Predicting RUL for those rows with the saved model

Not responsible for:
- Any form of tuning, re-selection, or threshold-fitting using test results
  (see the "test-set discipline" explanation in src/run_test_evaluation.py)

WHY THE TEST SET IS STRUCTURED DIFFERENTLY FROM TRAINING
train_FD001 engines run all the way to simulated failure, so training RUL
is calculable directly: `max_cycle_for_engine - current_cycle` (src/rul.py).
test_FD001 engines are deliberately cut off BEFORE failure - if engine 1's
test data ends at cycle 31, that does NOT mean it fails at cycle 31. The
whole point of the test set is to ask "given only what we observed up to
cycle 31, how much life is left?" - and RUL_FD001.txt holds NASA's answer:
one number per engine, the true remaining life measured from that engine's
LAST OBSERVED test cycle. So for engine 1:
    true failure cycle = 31 (last observed) + RUL_FD001 value (e.g. 112) = 143
Calculating `max(test cycle) - current cycle` on test data, the way we do
for training, would silently compute "cycles until the recording stops",
which is not the engine's remaining life at all - it would be a bug, not a
different modelling choice. This module never does that.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.cmapss_features import add_rolling_features
from src.cmapss_loader import (
    SENSOR_COLUMNS,
    load_rul_fd001,
    load_test_fd001,
    validate_cmapss_dataframe,
)
from src.rul import DEFAULT_RUL_CAP
from src.train_rul_temporal import ROLLING_WINDOWS, TREND_WINDOW


def load_and_validate_test_data(
    test_path=None, rul_path=None
) -> tuple[pd.DataFrame, pd.Series]:
    """Load test_FD001 and RUL_FD001, and PROVE (not assume) they are aligned.

    Checks:
    - test_FD001 parses to the expected structure (reuses validate_cmapss_dataframe)
    - one RUL value exists per test engine (counts match)
    - unit numbers are the dense sequence 1..N (RUL_FD001 has no unit_number
      column of its own - the only thing tying a value to an engine is that
      its row order matches the order engines first appear in test_FD001,
      which is only safe to rely on if that order is exactly 1, 2, 3, ...)
    """
    kwargs_test = {"file_path": test_path} if test_path else {}
    kwargs_rul = {"file_path": rul_path} if rul_path else {}

    test_df = load_test_fd001(**kwargs_test)
    validate_cmapss_dataframe(test_df, "test_FD001")
    rul = load_rul_fd001(**kwargs_rul)

    n_engines = test_df["unit_number"].nunique()
    if len(rul) != n_engines:
        raise ValueError(
            f"RUL_FD001 has {len(rul)} values but test_FD001 has {n_engines} engines - "
            "cannot align them one-to-one."
        )

    unit_numbers = sorted(test_df["unit_number"].unique())
    expected = list(range(1, n_engines + 1))
    if unit_numbers != expected:
        raise ValueError(
            "test_FD001 unit_number values are not the dense sequence 1..N "
            f"(got {unit_numbers[:3]}...{unit_numbers[-3:]}) - RUL_FD001 row order "
            "cannot be safely assumed to align with engine order."
        )

    return test_df, rul


def infer_kept_sensors_from_feature_list(
    feature_columns: list[str], sensor_columns: list[str] = SENSOR_COLUMNS
) -> list[str]:
    """Recover which sensors survived Step 4's screening, from the SAVED feature list.

    Deliberately reads this off the model's own saved feature list rather
    than re-running sensor screening here - screening is a training-data-only
    decision (Step 4), and this function's whole job is to never re-derive
    anything from data at test time. A kept sensor's raw name is always one
    of the baseline feature columns (see build_feature_columns), so
    intersecting with SENSOR_COLUMNS recovers the exact set.
    """
    return [sensor for sensor in sensor_columns if sensor in feature_columns]


def build_test_features(test_df: pd.DataFrame, kept_sensors: list[str]) -> pd.DataFrame:
    """Apply the SAME causal rolling/trend feature pipeline used in training (Step 6).

    Leakage note: add_rolling_features groups by unit_number and only ever
    looks backward within a group (see its own docstring/tests) - the same
    property that made it safe for the train/validation split applies
    identically here. Nothing about running it on test engines instead of
    validation engines changes that guarantee.
    """
    return add_rolling_features(test_df, kept_sensors, windows=ROLLING_WINDOWS, trend_window=TREND_WINDOW)


def select_final_cycle_rows(featured_test_df: pd.DataFrame) -> pd.DataFrame:
    """One row per engine: its LAST observed cycle - the point RUL_FD001 labels.

    This is the standard FD001 test protocol: NASA scores exactly one
    prediction per engine, made at the final available observation, because
    that is the realistic deployment moment - "given everything we've seen
    for this engine so far, how much life is left right now?"
    """
    idx = featured_test_df.groupby("unit_number")["time_cycles"].idxmax()
    final_rows = featured_test_df.loc[idx].sort_values("unit_number").reset_index(drop=True)
    # Keep `time_cycles` itself (it's one of the model's trained-on features -
    # see build_feature_columns) and ADD a clearly-named alias for display/output.
    final_rows["last_observed_cycle"] = final_rows["time_cycles"]
    return final_rows


def load_saved_temporal_model(model_path: Path, features_path: Path):
    """Load the saved model and its exact feature list/order; fail loudly if incompatible."""
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found: {model_path}. Run: python -m src.train_rul_temporal")
    if not features_path.exists():
        raise FileNotFoundError(f"Saved feature list not found: {features_path}.")

    model = joblib.load(model_path)
    import json

    feature_columns = json.loads(features_path.read_text())
    return model, feature_columns


def verify_features_present(df: pd.DataFrame, feature_columns: list[str]) -> None:
    """Fail loudly, rather than silently continuing, if the test pipeline produced
    a different feature set than what the model was trained on."""
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{len(missing)} expected feature(s) missing from the test feature set: {missing[:10]}"
        )


def build_final_predictions_df(
    final_rows_df: pd.DataFrame,
    feature_columns: list[str],
    model,
    actual_rul: pd.Series,
    rul_cap: int = DEFAULT_RUL_CAP,
) -> pd.DataFrame:
    """One row per test engine: prediction + both capped and uncapped ground truth.

    error = predicted_rul - actual_rul_capped (the SAME sign convention used
    everywhere else in this project: positive = model predicts more life
    than reality). Compared against the CAPPED target, not the raw NASA
    value - see the module-level note on why (src/run_test_evaluation.py
    explains this at print time too, for the reader).
    """
    verify_features_present(final_rows_df, feature_columns)

    X = final_rows_df[feature_columns]
    predicted_rul = model.predict(X)

    actual_rul = actual_rul.to_numpy()
    actual_rul_capped = pd.Series(actual_rul).clip(upper=rul_cap).to_numpy()

    result = pd.DataFrame(
        {
            "unit_number": final_rows_df["unit_number"].to_numpy(),
            "last_observed_cycle": final_rows_df["last_observed_cycle"].to_numpy(),
            "time_cycles": final_rows_df["last_observed_cycle"].to_numpy(),  # alias: reuse cycle-indexed helpers
            "actual_rul_uncapped": actual_rul,
            "actual_rul": actual_rul_capped,  # capped - the primary evaluation target
            "predicted_rul": predicted_rul,
        }
    )
    result["error"] = result["predicted_rul"] - result["actual_rul"]
    result["absolute_error"] = result["error"].abs()
    return result


def compute_summary_metrics(df: pd.DataFrame, label: str = "") -> dict:
    """MAE, RMSE and the extra distribution stats Step 11 asks for, from any
    DataFrame with `error` and `absolute_error` columns.

    Generic on purpose - used for both the official test set (100 rows, one
    per engine) and the validation set (thousands of rows, one per cycle),
    so the comparison table in src/run_test_evaluation.py is computed the
    same way on both sides.
    """
    errors = df["error"]
    return {
        "label": label,
        "n": len(df),
        "mae": float(df["absolute_error"].mean()),
        "rmse": float(np.sqrt((errors**2).mean())),
        "mean_signed_error": float(errors.mean()),
        "median_absolute_error": float(df["absolute_error"].median()),
        "max_over_prediction": float(errors.max()),
        "max_under_prediction": float(errors.min()),
    }


def compare_validation_vs_test(validation_metrics: dict, test_metrics: dict) -> pd.DataFrame:
    """A 2-row MAE/RMSE comparison table - see run_test_evaluation.py for how to read it."""
    return pd.DataFrame(
        [
            {"dataset": validation_metrics["label"], "n": validation_metrics["n"],
             "mae": validation_metrics["mae"], "rmse": validation_metrics["rmse"]},
            {"dataset": test_metrics["label"], "n": test_metrics["n"],
             "mae": test_metrics["mae"], "rmse": test_metrics["rmse"]},
        ]
    )
