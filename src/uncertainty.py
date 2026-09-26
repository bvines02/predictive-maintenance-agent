"""Stage 14h (diagnostic) - Random Forest ensemble-disagreement uncertainty.

Responsible for:
- Extracting each individual tree's prediction from a fitted RandomForestRegressor
- Turning that spread of tree predictions into per-row uncertainty metrics
- Deriving a 3-level `model_confidence` label from those metrics, using
  thresholds fixed from TRAINING data only (see compute_uncertainty_thresholds)
- Diagnostics: does higher disagreement actually predict larger error? how
  often does the true value fall inside the ensemble's own p10-p90 range?

Not responsible for:
- Model training or feature engineering (reuses the exact fitted model,
  split, and feature set from Steps 5-7 - see src/run_uncertainty_analysis.py)
- Deciding maintenance actions (Step 9's src/decision_engine.py only CONSUMES
  the `model_confidence` this module produces)

WHAT THIS IS: a RandomForestRegressor's prediction is the mean of every
tree's own prediction. Trees are grown on different bootstrap samples and
different random feature subsets, so on an "easy" row (clear signal) they
tend to agree; on an ambiguous row they scatter. Spread across the trees is
therefore a signal of ensemble DISAGREEMENT.

WHAT THIS IS NOT: a calibrated predictive interval. There is no theoretical
guarantee that a [p10, p90] band across trees covers the true value 80% of
the time - trees are correlated (same data, similar splits), so the spread
can understate real uncertainty. Step 20's docstring below and the
empirical_coverage() function exist specifically to check this rather than
assume it. Throughout this module and its outputs, this is called an
"ensemble prediction range", never a "confidence interval".

See the module docstring in src/run_uncertainty_analysis.py for the fuller
discussion of aleatoric/epistemic/out-of-distribution/sensor-quality
uncertainty this method does NOT capture, and what production alternatives
(conformal prediction, quantile regression forests, OOD detection) would add.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

UNCERTAINTY_BANDS = ["LOW", "MEDIUM", "HIGH"]
CONFIDENCE_LABELS = ["HIGH", "MEDIUM", "LOW"]

# Ensemble disagreement (LOW/MEDIUM/HIGH) maps INVERSELY to trust in the
# prediction (HIGH/MEDIUM/LOW confidence): trees agreeing a lot -> we trust
# the number more.
UNCERTAINTY_TO_CONFIDENCE = {"LOW": "HIGH", "MEDIUM": "MEDIUM", "HIGH": "LOW"}

DEFAULT_LOW_PERCENTILE = 33
DEFAULT_HIGH_PERCENTILE = 67


def extract_tree_predictions(model: RandomForestRegressor, X) -> np.ndarray:
    """Every tree's prediction for every row: shape (n_rows, n_trees).

    `model.predict(X)` already IS the row-wise mean of this matrix (that's
    the definition of a Random Forest regressor's output) - this function
    just exposes the individual trees behind that average instead of only
    the average itself.
    """
    # A plain list comprehension over model.estimators_ (one .predict() call
    # per tree) is simple and, at ~200 trees x a few thousand rows, fast
    # enough here; np.column_stack avoids a manual pre-allocated loop.
    # X.to_numpy(): each tree was fit on a plain array internally (that's
    # how RandomForestRegressor builds its estimators_), so it carries no
    # feature names - predicting with a DataFrame triggers a spurious
    # "fitted without feature names" warning per tree for no functional
    # difference in the result.
    values = X.to_numpy() if hasattr(X, "to_numpy") else X
    return np.column_stack([tree.predict(values) for tree in model.estimators_])


def compute_prediction_distribution(tree_predictions: np.ndarray) -> pd.DataFrame:
    """Per-row summary of the tree-prediction spread.

    prediction_std: how much the trees disagree, in cycles. Low = trees
        broadly agree on this row's RUL. High = trees give materially
        different estimates - a signal the row is harder for this forest.
    prediction_p10 / prediction_p90: the 10th/90th percentile VALUES across
        this row's trees - an EMPIRICAL range covering the middle 80% of
        tree opinions. Not a calibrated 80% confidence interval - see the
        module docstring.
    prediction_range: p90 - p10, a second, more robust (less outlier-
        sensitive) view of spread alongside prediction_std.
    """
    return pd.DataFrame(
        {
            "mean_prediction": tree_predictions.mean(axis=1),
            "median_prediction": np.median(tree_predictions, axis=1),
            "prediction_std": tree_predictions.std(axis=1, ddof=0),
            "prediction_min": tree_predictions.min(axis=1),
            "prediction_max": tree_predictions.max(axis=1),
            "prediction_p10": np.percentile(tree_predictions, 10, axis=1),
            "prediction_p90": np.percentile(tree_predictions, 90, axis=1),
        }
    ).assign(prediction_range=lambda d: d["prediction_p90"] - d["prediction_p10"])


def add_uncertainty_columns(
    predictions_df: pd.DataFrame, tree_predictions: np.ndarray, atol: float = 1e-6
) -> pd.DataFrame:
    """Attach the tree-prediction distribution to an existing predictions frame.

    Asserts the ensemble mean matches `predicted_rul` (the RF's own
    .predict() output already in `predictions_df`) - if these ever diverge,
    the tree predictions were extracted from a different model, a different
    row order, or a different X than what actually produced `predicted_rul`,
    which would silently invalidate every metric downstream.
    """
    if len(predictions_df) != len(tree_predictions):
        raise ValueError(
            f"Row count mismatch: predictions_df has {len(predictions_df)} rows, "
            f"tree_predictions has {len(tree_predictions)}."
        )

    distribution = compute_prediction_distribution(tree_predictions).reset_index(drop=True)
    result = pd.concat([predictions_df.reset_index(drop=True), distribution], axis=1)

    max_diff = (result["mean_prediction"] - result["predicted_rul"]).abs().max()
    if max_diff > atol:
        raise ValueError(
            f"Mean tree prediction does not match predicted_rul (max abs diff {max_diff:.6g}). "
            "tree_predictions must come from the same fitted model and the same X as predicted_rul."
        )

    result["absolute_error"] = result["error"].abs()
    return result


def compute_uncertainty_thresholds(
    prediction_std: pd.Series,
    low_percentile: float = DEFAULT_LOW_PERCENTILE,
    high_percentile: float = DEFAULT_HIGH_PERCENTILE,
) -> dict:
    """Fixed prediction_std cut points, from a distribution of `prediction_std` values.

    Percentile-based cut points are RELATIVE, not physical: "LOW uncertainty"
    here means "one of the third of rows where these trees agreed most",
    not "trustworthy in any absolute sense". If every row happened to have
    high disagreement (e.g. a much harder dataset), a third of them would
    still be labelled LOW. Compute this on TRAINING rows only (see
    src/run_uncertainty_analysis.py) so the cut points are fixed before
    looking at validation performance - the same reasoning as Step 4's
    sensor screening being training-only.
    """
    if not 0 <= low_percentile < high_percentile <= 100:
        raise ValueError("Require 0 <= low_percentile < high_percentile <= 100.")

    return {
        "low_max": float(np.percentile(prediction_std, low_percentile)),
        "medium_max": float(np.percentile(prediction_std, high_percentile)),
    }


def assign_uncertainty_band(prediction_std: pd.Series, thresholds: dict) -> pd.Series:
    """LOW/MEDIUM/HIGH disagreement band from prediction_std and fixed thresholds."""
    conditions = [prediction_std <= thresholds["low_max"], prediction_std <= thresholds["medium_max"]]
    labels = np.select(conditions, ["LOW", "MEDIUM"], default="HIGH")
    return pd.Series(labels, index=prediction_std.index, name="uncertainty_band")


def assign_model_confidence(uncertainty_band: pd.Series) -> pd.Series:
    """LOW/MEDIUM/HIGH disagreement -> HIGH/MEDIUM/LOW model_confidence (inverted).

    Named `model_confidence` throughout to keep it visibly distinct from
    "ground truth": this is ENSEMBLE-AGREEMENT confidence - how much this
    particular forest's trees agree with each other - not a statement that
    the prediction is actually correct.
    """
    mapped = uncertainty_band.map(UNCERTAINTY_TO_CONFIDENCE)
    if mapped.isna().any():
        bad = sorted(set(uncertainty_band) - set(UNCERTAINTY_TO_CONFIDENCE))
        raise ValueError(f"Unknown uncertainty_band value(s): {bad}")
    return mapped.rename("model_confidence")


def correlation_uncertainty_vs_error(df: pd.DataFrame) -> float:
    """Pearson correlation between prediction_std and absolute_error.

    A positive correlation supports "more disagreement -> bigger mistakes".
    It is not assumed to be strong - report whatever value comes out.
    """
    return float(df["prediction_std"].corr(df["absolute_error"]))


def error_by_uncertainty_band(df: pd.DataFrame) -> pd.DataFrame:
    """Row count and mean absolute error per uncertainty band, in LOW/MEDIUM/HIGH order."""
    grouped = df.groupby("uncertainty_band")["absolute_error"].agg(["count", "mean"])
    grouped = grouped.reindex(UNCERTAINTY_BANDS).rename(columns={"mean": "mean_absolute_error"})
    grouped.index.name = "uncertainty_band"
    return grouped.reset_index()


def near_failure_uncertainty(df: pd.DataFrame, thresholds: tuple = (30, 15)) -> pd.DataFrame:
    """Compare avg prediction_std/range and MAE/RMSE for the full set vs. near-failure subsets."""
    rows = []
    for label, subset in [("All validation rows", df)] + [
        (f"actual RUL <= {t}", df[df["actual_rul"] <= t]) for t in thresholds
    ]:
        errors = subset["error"]
        rows.append(
            {
                "subset": label,
                "n": len(subset),
                "avg_prediction_std": subset["prediction_std"].mean(),
                "avg_prediction_range": subset["prediction_range"].mean(),
                "mae": errors.abs().mean(),
                "rmse": float(np.sqrt((errors**2).mean())),
            }
        )
    return pd.DataFrame(rows)


_DISPLAY_COLUMNS = [
    "unit_number", "time_cycles", "actual_rul", "predicted_rul", "error",
    "prediction_std", "model_confidence",
]


def dangerous_high_confidence_errors(
    df: pd.DataFrame, actual_max: int = 15, min_over_prediction: float = 10.0, n: int = 10
) -> pd.DataFrame:
    """Wrong AND apparently sure of itself: near failure, badly over-predicting, HIGH confidence.

    This is the worst combination operationally - nothing about the model's
    own output (its ensemble agreement) would have warned a planner to
    doubt this particular prediction.
    """
    rows = df[
        (df["actual_rul"] <= actual_max)
        & (df["error"] >= min_over_prediction)
        & (df["model_confidence"] == "HIGH")
    ]
    return rows.sort_values("error", ascending=False)[_DISPLAY_COLUMNS].head(n).reset_index(drop=True)


def low_confidence_large_errors(
    df: pd.DataFrame, min_absolute_error: float = 10.0, n: int = 10
) -> pd.DataFrame:
    """Large error, but the model's own trees disagreed (LOW confidence) - less surprising."""
    rows = df[(df["absolute_error"] >= min_absolute_error) & (df["model_confidence"] == "LOW")]
    return rows.sort_values("absolute_error", ascending=False)[_DISPLAY_COLUMNS].head(n).reset_index(drop=True)


def empirical_coverage(df: pd.DataFrame) -> float:
    """Fraction of rows where actual_rul falls within [prediction_p10, prediction_p90].

    Purely diagnostic - see the module docstring for why this is not
    expected to equal 80% (tree percentiles are not a calibrated interval).
    """
    within = (df["actual_rul"] >= df["prediction_p10"]) & (df["actual_rul"] <= df["prediction_p90"])
    return float(within.mean())
