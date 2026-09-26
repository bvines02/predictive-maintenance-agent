"""Stage 14h - Step 10: Random Forest ensemble-disagreement uncertainty report.

Reuses the exact fitted temporal model, engine split, and feature set from
Steps 5-7 (via src.run_error_analysis's _prepare_split/_train_and_predict) -
no retraining beyond what those steps already do, no new algorithm.

Pipeline:
1. Extract every tree's prediction for train and validation rows.
2. Fix LOW/MEDIUM/HIGH uncertainty-band thresholds from the TRAINING rows'
   prediction_std distribution only (never validation - see Step 4's
   sensor-screening precedent for the same reasoning).
3. Derive model_confidence for validation rows and check whether it is
   correlated with actual error, whether it holds near failure, and how
   often the true value falls inside the ensemble's own p10-p90 range.
4. Feed model_confidence into Step 8's health-state table and Step 9's
   decision engine, replacing the manually supplied confidence value.
"""

import pandas as pd

from src.config import (
    UNCERTAINTY_ARTIFACTS_DIR,
    UNCERTAINTY_SUMMARY_PATH,
    VALIDATION_PREDICTIONS_WITH_UNCERTAINTY_PATH,
)
from src.decision_engine import AssetContext, decide_for_dataframe_with_confidence
from src.health_state import add_health_states
from src.run_error_analysis import MODEL_B_NAME, _prepare_split, _train_and_predict
from src.train_rul_baseline import build_modeling_dataset
from src.uncertainty import (
    add_uncertainty_columns,
    assign_model_confidence,
    assign_uncertainty_band,
    compute_uncertainty_thresholds,
    correlation_uncertainty_vs_error,
    dangerous_high_confidence_errors,
    empirical_coverage,
    error_by_uncertainty_band,
    extract_tree_predictions,
    low_confidence_large_errors,
    near_failure_uncertainty,
)
from src.uncertainty_plots import plot_engine_uncertainty, plot_error_vs_uncertainty


def _section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


if __name__ == "__main__":
    train_split_df, val_split_df, _, temporal_columns = _prepare_split()
    model, val_predictions = _train_and_predict(train_split_df, val_split_df, temporal_columns)

    print(f"Using {MODEL_B_NAME} (Step 6/7's best model) - {len(model.estimators_)} trees, "
          f"{len(temporal_columns)} features, same engine split as Steps 5-9.")

    # --- Extract tree-level predictions ------------------------------------
    X_train, _ = build_modeling_dataset(train_split_df, temporal_columns)
    X_val, _ = build_modeling_dataset(val_split_df, temporal_columns)
    train_tree_predictions = extract_tree_predictions(model, X_train)
    val_tree_predictions = extract_tree_predictions(model, X_val)

    _section("Tree prediction matrix")
    print(f"Validation matrix shape: {val_tree_predictions.shape} (rows x trees)")

    # --- Uncertainty metrics + verification --------------------------------
    df = add_uncertainty_columns(val_predictions, val_tree_predictions)
    max_diff = (df["mean_prediction"] - df["predicted_rul"]).abs().max()
    print(f"Verified: mean tree prediction matches RF predicted_rul (max abs diff = {max_diff:.2e}).")

    # --- Thresholds from TRAINING rows only --------------------------------
    train_std = pd.Series(train_tree_predictions.std(axis=1, ddof=0))
    thresholds = compute_uncertainty_thresholds(train_std)
    _section("Uncertainty-band thresholds (from TRAINING prediction_std distribution)")
    print(f"  LOW:    prediction_std <= {thresholds['low_max']:.3f}")
    print(f"  MEDIUM: {thresholds['low_max']:.3f} < prediction_std <= {thresholds['medium_max']:.3f}")
    print(f"  HIGH:   prediction_std > {thresholds['medium_max']:.3f}")
    print("  (relative cut points - see src/uncertainty.py docstring on why these are not physical thresholds)")

    df["uncertainty_band"] = assign_uncertainty_band(df["prediction_std"], thresholds)
    df["model_confidence"] = assign_model_confidence(df["uncertainty_band"])

    # --- Does disagreement predict error? -----------------------------------
    corr = correlation_uncertainty_vs_error(df)
    _section("Does ensemble disagreement predict error?")
    print(f"Correlation(prediction_std, absolute_error) = {corr:.3f}")
    print(error_by_uncertainty_band(df).to_string(index=False))

    # --- Coverage -----------------------------------------------------------
    coverage = empirical_coverage(df)
    _section("Empirical coverage of the ensemble [p10, p90] range")
    print(f"actual_rul within [p10, p90]: {100 * coverage:.1f}% of validation rows")
    print("This is a diagnostic, not a target - tree percentiles are not a calibrated interval (see module docstring).")

    # --- Near failure --------------------------------------------------------
    _section("Uncertainty near failure")
    print(near_failure_uncertainty(df).to_string(index=False))

    # --- Dangerous cases -------------------------------------------------
    _section("Dangerous: near failure, badly over-predicting, HIGH confidence")
    dangerous = dangerous_high_confidence_errors(df)
    print(dangerous.to_string(index=False) if len(dangerous) else "  none found")

    _section("Large error, LOW confidence (model itself signalled disagreement)")
    print(low_confidence_large_errors(df).to_string(index=False))

    # --- Plots ----------------------------------------------------------
    example_unit = int(df["unit_number"].min())
    widest_unit = int(df.groupby("unit_number")["prediction_std"].mean().idxmax())
    for unit in {example_unit, widest_unit}:
        path = UNCERTAINTY_ARTIFACTS_DIR / f"engine_{unit}_uncertainty.png"
        plot_engine_uncertainty(df, unit, path)
        print(f"\nSaved uncertainty plot for engine {unit} to {path}")

    plot_error_vs_uncertainty(df, UNCERTAINTY_ARTIFACTS_DIR / "error_vs_uncertainty.png")
    print(f"Saved error-vs-uncertainty plot to {UNCERTAINTY_ARTIFACTS_DIR / 'error_vs_uncertainty.png'}")

    # --- Connect to Step 8 health states -------------------------------
    df_with_health = add_health_states(df)
    _section("Same PLAN state, different model_confidence")
    plan_rows = df_with_health[df_with_health["health_state"] == "PLAN"]
    sample = (
        plan_rows.sort_values("predicted_rul")
        .groupby("model_confidence")
        .head(1)[["unit_number", "time_cycles", "predicted_rul", "health_state", "model_confidence"]]
    )
    print(sample.to_string(index=False))
    print(
        "These should not necessarily lead to the same maintenance decision: a PLAN state the "
        "model is confident about is a more solid basis for committing resources than one where "
        "the trees themselves disagree - the health-state THRESHOLDS are unchanged, only how much "
        "we lean on the number changes downstream, in the decision engine."
    )

    # --- Connect to Step 9 decision engine -------------------------------
    _section("Step 9 decision engine, now driven by derived model_confidence")
    context = AssetContext(criticality="HIGH", redundancy="PARTIAL", maintenance_lead_time_cycles=15, confidence="HIGH")
    decided = decide_for_dataframe_with_confidence(df_with_health, context)
    print("Context: HIGH criticality / PARTIAL redundancy / lead time 15 (confidence column overrides the base HIGH placeholder per row)")
    print(
        decided.groupby(["health_state", "model_confidence"])["recommended_action"]
        .agg(lambda s: s.value_counts().idxmax())
        .rename("most_common_action")
        .reset_index()
        .to_string(index=False)
    )

    # --- Save -------------------------------------------------------------
    keep_columns = [
        "unit_number", "time_cycles", "actual_rul", "predicted_rul", "error", "absolute_error",
        "prediction_std", "prediction_p10", "prediction_p90", "prediction_range",
        "uncertainty_band", "model_confidence",
    ]
    df[keep_columns].to_csv(VALIDATION_PREDICTIONS_WITH_UNCERTAINTY_PATH, index=False)
    error_by_uncertainty_band(df).to_csv(UNCERTAINTY_SUMMARY_PATH, index=False)
    print(f"\nSaved {VALIDATION_PREDICTIONS_WITH_UNCERTAINTY_PATH}")
    print(f"Saved {UNCERTAINTY_SUMMARY_PATH}")
