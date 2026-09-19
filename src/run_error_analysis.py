"""Stage 14h - operationally meaningful RUL error-analysis report.

Responsible for:
- Re-deriving the exact Step 5/6 engine-level validation split and models
- Choosing the better-performing model (by overall validation MAE) as the
  primary subject of this analysis, while still comparing both models
  specifically near failure
- Running every diagnostic in src/error_analysis.py and src/error_analysis_plots.py
- Saving validation predictions and the per-band error table to results/

Not responsible for:
- Model training/tuning (re-trains Model A and Model B exactly as Steps 5/6
  did, using their unmodified functions - see src/train_rul_temporal.py's
  module docstring for why re-deriving the split this way is equivalent to
  reusing a saved one)
- Turning the RUL model into a classifier, deciding maintenance actions,
  or anything else beyond diagnostics (see error_analysis.py)
"""

from src.cmapss_features import add_rolling_features
from src.cmapss_loader import load_train_fd001, validate_cmapss_dataframe
from src.config import (
    ERROR_ANALYSIS_ARTIFACTS_DIR,
    ERROR_BY_RUL_BAND_PATH,
    RESULTS_DIR,
    VALIDATION_PREDICTIONS_PATH,
)
from src.engine_split import split_dataframe_by_unit, split_units, validate_engine_split
from src.error_analysis import (
    add_error_analysis_columns,
    build_near_failure_comparison,
    compute_band_error_table,
    critical_region_analysis,
    threshold_diagnostics,
    top_over_predictions,
    top_under_predictions,
)
from src.error_analysis_plots import (
    plot_actual_vs_predicted,
    plot_error_histogram,
    plot_error_vs_actual_rul,
)
from src.rul import add_capped_rul, add_rul_target
from src.train_rul_baseline import (
    RANDOM_STATE,
    VALIDATION_FRACTION,
    build_feature_columns,
    build_modeling_dataset,
    build_predictions_df,
    get_kept_sensor_columns,
    train_baseline_rf,
)
from src.train_rul_temporal import ROLLING_WINDOWS, TREND_WINDOW, build_temporal_feature_columns

MODEL_A_NAME = "Baseline RF"
MODEL_B_NAME = "Temporal-feature RF"


def _prepare_split():
    """Re-derive train_df, the temporal feature set, and Step 5/6's engine split."""
    train_df = load_train_fd001()
    validate_cmapss_dataframe(train_df, "train_FD001")
    train_df = add_rul_target(train_df)
    train_df = add_capped_rul(train_df)

    kept_sensors = get_kept_sensor_columns(train_df)
    baseline_feature_columns = build_feature_columns(train_df)
    temporal_feature_columns = build_temporal_feature_columns(train_df, kept_sensors)

    featured_df = add_rolling_features(
        train_df, kept_sensors, windows=ROLLING_WINDOWS, trend_window=TREND_WINDOW
    )
    train_units, val_units = split_units(
        featured_df["unit_number"], test_size=VALIDATION_FRACTION, random_state=RANDOM_STATE
    )
    train_split_df, val_split_df = split_dataframe_by_unit(featured_df, train_units, val_units)
    validate_engine_split(featured_df, train_split_df, val_split_df, train_units, val_units)

    return train_split_df, val_split_df, baseline_feature_columns, temporal_feature_columns


def _train_and_predict(train_split_df, val_split_df, feature_columns):
    X_train, y_train = build_modeling_dataset(train_split_df, feature_columns)
    X_val, _ = build_modeling_dataset(val_split_df, feature_columns)

    model = train_baseline_rf(X_train, y_train)
    val_predictions = build_predictions_df(val_split_df, model.predict(X_val))
    return val_predictions


if __name__ == "__main__":
    train_split_df, val_split_df, baseline_columns, temporal_columns = _prepare_split()

    predictions_a = _train_and_predict(train_split_df, val_split_df, baseline_columns)
    predictions_b = _train_and_predict(train_split_df, val_split_df, temporal_columns)

    overall_mae_a = predictions_a["error"].abs().mean()
    overall_mae_b = predictions_b["error"].abs().mean()

    if overall_mae_b <= overall_mae_a:
        primary_name, primary_predictions = MODEL_B_NAME, predictions_b
    else:
        primary_name, primary_predictions = MODEL_A_NAME, predictions_a

    print(f"{'=' * 70}\nModel selection\n{'=' * 70}")
    print(f"{MODEL_A_NAME} overall val MAE: {overall_mae_a:.2f}")
    print(f"{MODEL_B_NAME} overall val MAE: {overall_mae_b:.2f}")
    print(f"Using '{primary_name}' as the primary model for this error analysis.")

    primary_predictions = add_error_analysis_columns(primary_predictions)

    # --- Per-band error table ---------------------------------------------
    band_table = compute_band_error_table(primary_predictions)
    print(f"\n{'=' * 70}\nError by RUL band ({primary_name})\n{'=' * 70}")
    print(band_table.to_string(index=False))

    # --- Critical region analysis ------------------------------------------
    critical_stats = critical_region_analysis(primary_predictions)
    print(f"\n{'=' * 70}\nCritical region (actual RUL <= 15) - {primary_name}\n{'=' * 70}")
    for key, value in critical_stats.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")

    # --- Worst dangerous over-predictions and under-predictions ----------
    print(f"\n{'=' * 70}\nTop 10 over-predictions (most dangerous) - {primary_name}\n{'=' * 70}")
    print(top_over_predictions(primary_predictions).to_string(index=False))

    print(f"\n{'=' * 70}\nTop 10 under-predictions - {primary_name}\n{'=' * 70}")
    print(top_under_predictions(primary_predictions).to_string(index=False))

    # --- Plots --------------------------------------------------------
    plot_error_histogram(primary_predictions, ERROR_ANALYSIS_ARTIFACTS_DIR / "error_histogram.png")
    plot_actual_vs_predicted(
        primary_predictions, ERROR_ANALYSIS_ARTIFACTS_DIR / "actual_vs_predicted.png"
    )
    plot_error_vs_actual_rul(
        primary_predictions, ERROR_ANALYSIS_ARTIFACTS_DIR / "error_vs_actual_rul.png"
    )
    print(f"\nSaved 3 diagnostic plots to {ERROR_ANALYSIS_ARTIFACTS_DIR}")

    # --- Baseline vs. temporal, specifically near failure -----------------
    near_failure_comparison = build_near_failure_comparison(
        {MODEL_A_NAME: predictions_a, MODEL_B_NAME: predictions_b}
    )
    print(f"\n{'=' * 70}\nBaseline vs. temporal model - overall vs. near failure\n{'=' * 70}")
    print(near_failure_comparison.to_string(index=False))

    # --- Threshold diagnostics (exploratory only, NOT a classifier) -------
    diagnostics_15 = threshold_diagnostics(primary_predictions, actual_threshold=15, predicted_threshold=15)
    print(f"\n{'=' * 70}\nThreshold diagnostic: predicted RUL <= 15 vs. actual RUL <= 15\n{'=' * 70}")
    for key, value in diagnostics_15.items():
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")

    diagnostics_25 = threshold_diagnostics(primary_predictions, actual_threshold=15, predicted_threshold=25)
    print(f"\n{'=' * 70}\nThreshold diagnostic: predicted RUL <= 25 vs. actual RUL <= 15\n{'=' * 70}")
    for key, value in diagnostics_25.items():
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")

    # --- Save results -------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    primary_predictions.to_csv(VALIDATION_PREDICTIONS_PATH, index=False)
    band_table.to_csv(ERROR_BY_RUL_BAND_PATH, index=False)
    print(f"\nSaved validation predictions to {VALIDATION_PREDICTIONS_PATH}")
    print(f"Saved error-by-band table to {ERROR_BY_RUL_BAND_PATH}")
