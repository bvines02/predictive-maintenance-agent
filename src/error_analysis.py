"""Stage 14h (diagnostic) - operationally meaningful RUL error analysis.

Responsible for:
- Labelling validation predictions with an RUL band
- Summarising error by band (MAE, RMSE, signed error, over/under-prediction rate)
- A focused look at the critical (near-failure) region
- Finding the worst dangerous over-predictions and worst under-predictions
- A simple, DIAGNOSTIC-ONLY threshold check: how well does "predicted RUL is
  low" line up with "actual RUL is low"?

Not responsible for:
- Changing the model into a classifier (see threshold_diagnostics docstring)
- Deciding maintenance actions (that is decision_rules.py, later)
- Model training or tuning

Why band/direction analysis matters, beyond overall MAE/RMSE: two models
can have identical overall MAE while behaving completely differently where
it counts. A model that is slightly wrong everywhere is a very different
(and safer) thing than a model that is very accurate on healthy engines but
badly over-predicts remaining life on the handful of engines about to fail.
Overall metrics average these together and hide the difference - this
module exists to stop hiding it.

Sign convention (shared with src/train_rul_baseline.py's build_predictions_df):
    error = predicted_rul - actual_rul
    positive error -> the model thinks MORE life remains than actually does
                      (optimistic / "the asset looks healthier than it is")
    negative error -> the model thinks LESS life remains than actually does
                      (conservative / "the asset looks worse than it is")
This sign choice is what lets "% over" mean "% optimistic" consistently
throughout this module - flipping the sign would flip that meaning, so it
must stay consistent with build_predictions_df.
"""

import numpy as np
import pandas as pd

# Configurable RUL bands - deliberately plain data (a list of dicts), not
# hard-coded into the functions below, so the boundaries can be revisited
# (e.g. once we have real failure-cost information) without touching logic.
DEFAULT_RUL_BANDS = [
    {"label": "Critical", "min": 0, "max": 15},
    {"label": "High concern", "min": 16, "max": 30},
    {"label": "Medium", "min": 31, "max": 60},
    {"label": "Healthy / long horizon", "min": 61, "max": 125},
]

DEFAULT_CRITICAL_MAX = 15


def add_error_analysis_columns(
    predictions_df: pd.DataFrame, bands: list[dict] = DEFAULT_RUL_BANDS
) -> pd.DataFrame:
    """Add `absolute_error` and `rul_band` to a validation predictions DataFrame.

    Expects `predictions_df` to already have `actual_rul` and `error`
    (as produced by src.train_rul_baseline.build_predictions_df). Returns a
    new DataFrame - does not modify the input.
    """
    result = predictions_df.copy()
    result["absolute_error"] = result["error"].abs()
    result["rul_band"] = assign_rul_band(result["actual_rul"], bands)
    return result


def assign_rul_band(actual_rul: pd.Series, bands: list[dict] = DEFAULT_RUL_BANDS) -> pd.Series:
    """Map each actual_rul value to a band label, using inclusive [min, max] ranges.

    A value that falls outside every configured band is labelled "Unbanded"
    rather than silently dropped or mis-assigned - with rul_capped's fixed
    0-125 range and the default bands covering 0-125 exactly, this should
    never trigger, but a config change that leaves a gap will show up
    loudly in the resulting table instead of vanishing quietly.
    """
    conditions = [(actual_rul >= band["min"]) & (actual_rul <= band["max"]) for band in bands]
    choices = [band["label"] for band in bands]
    labels = np.select(conditions, choices, default="Unbanded")
    return pd.Series(labels, index=actual_rul.index, name="rul_band")


def compute_band_error_table(
    predictions_df: pd.DataFrame, bands: list[dict] = DEFAULT_RUL_BANDS
) -> pd.DataFrame:
    """Per-RUL-band error summary: observations, MAE, RMSE, signed error, over/under rate.

    Requires `predictions_df` to already carry `rul_band` and
    `absolute_error` (see add_error_analysis_columns). Rows are ordered by
    the band definitions' own order (Critical first), not alphabetically.
    """
    band_order = [band["label"] for band in bands]

    rows = []
    for label in band_order:
        band_rows = predictions_df[predictions_df["rul_band"] == label]
        if len(band_rows) == 0:
            continue
        errors = band_rows["error"]
        rows.append(
            {
                "rul_band": label,
                "observations": len(band_rows),
                "mae": errors.abs().mean(),
                "rmse": float(np.sqrt((errors**2).mean())),
                "mean_error": errors.mean(),
                "median_abs_error": errors.abs().median(),
                "pct_over": 100 * (errors > 0).mean(),
                "pct_under": 100 * (errors < 0).mean(),
            }
        )

    return pd.DataFrame(rows)


def critical_region_analysis(
    predictions_df: pd.DataFrame, critical_max: int = DEFAULT_CRITICAL_MAX
) -> dict:
    """Focused error summary for rows where actual_rul <= critical_max.

    This is the region where a wrong prediction is most consequential - see
    the module docstring and the "maintenance asymmetry" explanation in the
    project write-up.
    """
    critical_rows = predictions_df[predictions_df["actual_rul"] <= critical_max]
    errors = critical_rows["error"]

    return {
        "n_observations": len(critical_rows),
        "mae": errors.abs().mean(),
        "rmse": float(np.sqrt((errors**2).mean())),
        "mean_error": errors.mean(),
        "max_over_prediction": errors.max(),
        "max_under_prediction": errors.min(),
        "pct_abs_error_gt_5": 100 * (errors.abs() > 5).mean(),
        "pct_abs_error_gt_10": 100 * (errors.abs() > 10).mean(),
    }


def top_over_predictions(predictions_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """The `n` most dangerously optimistic predictions (largest positive error)."""
    columns = ["unit_number", "time_cycles", "actual_rul", "predicted_rul", "error"]
    return predictions_df.sort_values("error", ascending=False)[columns].head(n).reset_index(drop=True)


def top_under_predictions(predictions_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """The `n` most conservative predictions (most negative error)."""
    columns = ["unit_number", "time_cycles", "actual_rul", "predicted_rul", "error"]
    return predictions_df.sort_values("error", ascending=True)[columns].head(n).reset_index(drop=True)


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def threshold_diagnostics(
    predictions_df: pd.DataFrame,
    actual_threshold: int = DEFAULT_CRITICAL_MAX,
    predicted_threshold: int = DEFAULT_CRITICAL_MAX,
) -> dict:
    """Diagnostic-only confusion matrix: does a low PREDICTED RUL line up with a low ACTUAL RUL?

    This does NOT turn the RUL model into a classifier - the model still
    outputs a cycle count, exactly as before. This function simply asks a
    yes/no question of those existing regression outputs, purely to
    understand how well "the model's number is low" tracks "the true
    number is low" - a natural early-warning use case, not a redefinition
    of the modelling problem.

    "failure-near" (actual) = actual_rul <= actual_threshold
    "warning" (predicted)   = predicted_rul <= predicted_threshold

    TP: failure-near AND warned - correctly flagged as close to failure.
    FP: NOT failure-near BUT warned - a false alarm.
    FN: failure-near BUT NOT warned - a missed near-failure engine.
    TN: NOT failure-near AND NOT warned - correctly left alone.
    """
    actual_near = predictions_df["actual_rul"] <= actual_threshold
    predicted_near = predictions_df["predicted_rul"] <= predicted_threshold

    tp = int((actual_near & predicted_near).sum())
    fp = int((~actual_near & predicted_near).sum())
    fn = int((actual_near & ~predicted_near).sum())
    tn = int((~actual_near & ~predicted_near).sum())

    return {
        "actual_threshold": actual_threshold,
        "predicted_threshold": predicted_threshold,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": _safe_divide(tp, tp + fp),
        "recall": _safe_divide(tp, tp + fn),
    }


def build_near_failure_comparison(
    predictions_by_model: dict[str, pd.DataFrame], rul_le_30: int = 30, rul_le_15: int = 15
) -> pd.DataFrame:
    """Compare models on overall MAE vs. MAE restricted to near-failure rows.

    Answers requirement 15 directly: is the model with the best OVERALL MAE
    also the best where it matters most (near failure)? Those can differ -
    this table is what lets you check rather than assume.
    """
    rows = []
    for model_name, predictions_df in predictions_by_model.items():
        near_30 = predictions_df[predictions_df["actual_rul"] <= rul_le_30]
        near_15 = predictions_df[predictions_df["actual_rul"] <= rul_le_15]
        rows.append(
            {
                "model": model_name,
                "overall_mae": predictions_df["error"].abs().mean(),
                f"rul_le_{rul_le_30}_mae": near_30["error"].abs().mean(),
                f"rul_le_{rul_le_15}_mae": near_15["error"].abs().mean(),
                f"mean_error_le_{rul_le_15}": near_15["error"].mean(),
            }
        )
    return pd.DataFrame(rows)
