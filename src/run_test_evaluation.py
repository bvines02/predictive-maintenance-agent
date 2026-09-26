"""Stage 14h - Step 11: final evaluation on the official NASA FD001 test set.

This is the one script in the project that touches test_FD001/RUL_FD001.
Everything it needs - the model, the feature list, the sensor screening
result, the health-state thresholds, the uncertainty-band cut points - is
either LOADED from a saved artifact or RE-DERIVED FROM TRAINING DATA ONLY.
Nothing here is fit, tuned, or selected using the test results themselves.

TEST-SET DISCIPLINE (read this before looking at the numbers below)
This run happens exactly once, as a final exam - not a scratchpad. If the
numbers below are disappointing, the correct response is to say so plainly,
not to quietly go back and retune sensor screening, features, thresholds or
hyperparameters until this script reports a better score. Doing that turns
the test set into just another validation set, and the resulting "test"
score would no longer mean what a held-out score is supposed to mean. If a
materially different model is ever built after seeing these results, that
should be acknowledged explicitly, and ideally checked against yet another
held-out set the new decisions haven't seen either.

TRAINING vs VALIDATION vs TEST, plainly:
    training   = revision:   how well the model fits engines it studied
    validation = mock exam:  used, during development, to compare choices
                              (sensor screening, features, model) on engines
                              never trained on - but SEEN many times while
                              deciding what to build
    test       = final exam: seen ONCE, after every modelling decision was
                              already locked in - the only honest estimate
                              of how this pipeline performs on engines nobody
                              involved in building it has ever looked at
"""

import json

import pandas as pd

from src.config import (
    FD001_TEST_HEALTH_CONFUSION_PATH,
    FD001_TEST_METRICS_PATH,
    FD001_TEST_PLOTS_DIR,
    FD001_TEST_PREDICTIONS_PATH,
)
from src.decision_engine import AssetContext, decide_for_dataframe_with_confidence
from src.error_analysis import add_error_analysis_columns, compute_band_error_table, critical_region_analysis
from src.error_analysis_plots import plot_actual_vs_predicted, plot_error_vs_actual_rul
from src.evaluate_test_set import (
    build_final_predictions_df,
    build_test_features,
    compare_validation_vs_test,
    compute_summary_metrics,
    infer_kept_sensors_from_feature_list,
    load_and_validate_test_data,
    load_saved_temporal_model,
    select_final_cycle_rows,
)
from src.health_state import add_health_states, state_confusion_matrix, state_metrics
from src.nasa_score import nasa_cmapss_score
from src.run_error_analysis import _prepare_split
from src.synthetic_asset_profiles import SYNTHETIC_PROFILES
from src.test_evaluation_plots import plot_health_state_confusion_matrix
from src.train_rul_baseline import build_modeling_dataset
from src.train_rul_temporal import TEMPORAL_FEATURES_PATH, TEMPORAL_MODEL_PATH
from src.uncertainty import (
    add_uncertainty_columns,
    assign_model_confidence,
    assign_uncertainty_band,
    compute_uncertainty_thresholds,
    dangerous_high_confidence_errors,
    empirical_coverage,
    extract_tree_predictions,
)
from src.uncertainty_plots import plot_error_vs_uncertainty


def _section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


if __name__ == "__main__":
    _section("How the FD001 test set works (read before trusting any number below)")
    print(
        "train_FD001 engines run to simulated failure, so training RUL = max_cycle - current_cycle.\n"
        "test_FD001 engines are cut off BEFORE failure - a test trajectory ending at cycle 31 does NOT\n"
        "mean the engine fails at cycle 31. RUL_FD001.txt gives the true remaining life measured from\n"
        "each engine's LAST observed test cycle, e.g. engine 1: last cycle 31 + RUL_FD001 value 112\n"
        "= true failure at cycle 143. We evaluate ONE prediction per engine, made at that last cycle -\n"
        "the realistic 'how much life is left, right now?' moment."
    )

    # --- Load and validate --------------------------------------------------
    test_df, rul = load_and_validate_test_data()
    _section("Test/RUL alignment check")
    print(f"test_FD001 engines: {test_df['unit_number'].nunique()}   RUL_FD001 values: {len(rul)}   (must match, and did)")

    model, feature_columns = load_saved_temporal_model(TEMPORAL_MODEL_PATH, TEMPORAL_FEATURES_PATH)
    kept_sensors = infer_kept_sensors_from_feature_list(feature_columns)
    print(f"Loaded saved model ({len(model.estimators_)} trees) and its {len(feature_columns)}-feature list.")
    print(f"Sensors used (from Step 4's training-only screening, recovered from the saved feature list): {kept_sensors}")

    # --- Recreate the exact feature pipeline, causally, per engine ---------
    featured_test_df = build_test_features(test_df, kept_sensors)
    final_rows_df = select_final_cycle_rows(featured_test_df)
    predictions_df = build_final_predictions_df(final_rows_df, feature_columns, model, rul)

    _section("Final-cycle test predictions (sample)")
    print(predictions_df.head(10).to_string(index=False))
    print(
        f"\n{(predictions_df['actual_rul_uncapped'] > predictions_df['actual_rul']).sum()} of "
        f"{len(predictions_df)} engines had an official RUL above the {125} cycle training cap - "
        "for those, predicted_rul is compared against the CAPPED value (actual_rul), since the model "
        "was trained on rul_capped and cannot be meaningfully interpreted above that cap."
    )

    # --- Primary metrics ------------------------------------------------
    test_metrics = compute_summary_metrics(predictions_df, label="Official FD001 test")
    nasa_score = nasa_cmapss_score(predictions_df["error"])
    _section("OFFICIAL FD001 TEST RESULTS")
    print(f"Number of engines: {test_metrics['n']}")
    print(f"MAE:                 {test_metrics['mae']:.2f}")
    print(f"RMSE:                {test_metrics['rmse']:.2f}")
    print(f"Mean signed error:   {test_metrics['mean_signed_error']:+.2f}")
    print(f"Median abs. error:   {test_metrics['median_absolute_error']:.2f}")
    print(f"Max over-prediction: {test_metrics['max_over_prediction']:+.2f}")
    print(f"Max under-prediction:{test_metrics['max_under_prediction']:+.2f}")
    print(f"\nSecondary metric - PHM08/C-MAPSS asymmetric score: {nasa_score:.1f} "
          "(lower is better; penalises late/optimistic errors more than early ones - see src/nasa_score.py)")

    # --- Validation vs. test -------------------------------------------
    train_split_df, val_split_df, _, temporal_columns = _prepare_split()
    assert temporal_columns == feature_columns, "Re-derived feature list no longer matches the saved model's."
    X_val, _ = build_modeling_dataset(val_split_df, feature_columns)
    val_predictions = val_split_df[["unit_number", "time_cycles", "rul_capped"]].rename(
        columns={"rul_capped": "actual_rul"}
    ).reset_index(drop=True)
    val_predictions["predicted_rul"] = model.predict(X_val)
    val_predictions["error"] = val_predictions["predicted_rul"] - val_predictions["actual_rul"]
    val_predictions["absolute_error"] = val_predictions["error"].abs()
    val_metrics = compute_summary_metrics(val_predictions, label="Validation (cycle-level)")

    _section("Validation vs. official test")
    print(compare_validation_vs_test(val_metrics, test_metrics).to_string(index=False))
    print(
        "\nA. Similar MAE/RMSE -> the validation split was a fair proxy for genuinely unseen engines.\n"
        "B. Materially WORSE test performance -> validation was optimistic, e.g. because validation\n"
        "   engines still shared enough with training engines (same simulator, same fault mode) that\n"
        "   held-out performance overstated how the model handles fresh data - a sign to be more\n"
        "   cautious about deployment claims, not to retune against this test result.\n"
        "C. Unexpectedly BETTER test performance -> plausible with only 100 test engines (one point\n"
        "   each) vs thousands of validation cycles - a small evaluation set has more sampling noise,\n"
        "   so a single lucky/unlucky draw of engines can shift the average either way. Investigate\n"
        "   before celebrating; don't generalise from one 100-engine sample."
    )

    # --- RUL-band analysis (Step 7's bands, ONE row per engine here) -----
    predictions_df = add_error_analysis_columns(predictions_df)
    _section("Error by RUL band - test set (100 engines total, ONE observation each)")
    print("Unlike Step 7's validation analysis (thousands of per-cycle rows), each row below is a whole engine.")
    print(compute_band_error_table(predictions_df).to_string(index=False))

    # --- Near-failure ------------------------------------------------
    _section("Near-failure performance")
    for cutoff in (30, 15):
        stats = critical_region_analysis(predictions_df, critical_max=cutoff)
        print(f"\nactual_rul <= {cutoff}: n={stats['n_observations']}  MAE={stats['mae']:.2f}  "
              f"RMSE={stats['rmse']:.2f}  mean_error={stats['mean_error']:+.2f}  "
              f"max_over={stats['max_over_prediction']:+.2f}  max_under={stats['max_under_prediction']:+.2f}")

    dangerous_test = predictions_df[
        (predictions_df["actual_rul"] <= 15) & (predictions_df["error"] >= 10)
    ].sort_values("error", ascending=False)
    _section("Serious optimistic errors near failure (actual RUL low, predicted materially higher)")
    print(dangerous_test[["unit_number", "last_observed_cycle", "actual_rul", "predicted_rul", "error"]].to_string(index=False)
          if len(dangerous_test) else "  none found")

    # --- Health states -----------------------------------------------
    predictions_df = add_health_states(predictions_df)
    matrix = state_confusion_matrix(predictions_df)
    metrics = state_metrics(predictions_df)
    _section("Health-state performance (Step 8's thresholds, unchanged)")
    print(matrix.to_string())
    print(f"\nState accuracy: {metrics['state_accuracy']:.3f}")
    print(f"ACTION precision: {metrics['action_precision']:.3f}   ACTION recall: {metrics['action_recall']:.3f}")
    print(f"PLAN+ACTION recall: {metrics['plan_or_action_recall']:.3f}")

    # --- Uncertainty (thresholds from TRAINING rows only) ------------------
    X_train, _ = build_modeling_dataset(train_split_df, feature_columns)
    train_tree_predictions = extract_tree_predictions(model, X_train)
    train_prediction_std = pd.Series(train_tree_predictions.std(axis=1, ddof=0))
    thresholds = compute_uncertainty_thresholds(train_prediction_std)

    X_final = final_rows_df[feature_columns]
    test_tree_predictions = extract_tree_predictions(model, X_final)
    predictions_df = add_uncertainty_columns(predictions_df, test_tree_predictions)
    predictions_df["uncertainty_band"] = assign_uncertainty_band(predictions_df["prediction_std"], thresholds)
    predictions_df["model_confidence"] = assign_model_confidence(predictions_df["uncertainty_band"])

    _section("Model uncertainty on the test set (thresholds fixed from TRAINING data, not re-derived here)")
    print(f"Thresholds reused: LOW <= {thresholds['low_max']:.3f}  MEDIUM <= {thresholds['medium_max']:.3f}  else HIGH")
    print(predictions_df["model_confidence"].value_counts().reindex(["HIGH", "MEDIUM", "LOW"], fill_value=0).to_string())

    _section("Dangerous: near failure, badly over-predicting, and HIGH confidence")
    dangerous_confident = dangerous_high_confidence_errors(predictions_df, actual_max=15, min_over_prediction=10)
    print(dangerous_confident.to_string(index=False) if len(dangerous_confident) else "  none found")
    print(
        "\nWhy this combination matters for deployment governance: nothing about the model's OWN output\n"
        "(its tree agreement) would have flagged these predictions for a second look. A model that is\n"
        "wrong but visibly unsure (LOW confidence) can be caught by a human review policy; wrong AND\n"
        "confident cannot be caught by the model's own signals at all - only by an independent check."
    )

    coverage = empirical_coverage(predictions_df)
    _section("Empirical ensemble-range coverage")
    print(f"actual_rul within [p10, p90]: {100 * coverage:.1f}% of test engines")
    print("Diagnostic only - tree percentiles are not a calibrated statistical prediction interval (Step 10).")

    # --- Plots -----------------------------------------------------------
    plot_actual_vs_predicted(predictions_df, FD001_TEST_PLOTS_DIR / "actual_vs_predicted.png")
    plot_error_vs_actual_rul(predictions_df, FD001_TEST_PLOTS_DIR / "error_vs_actual_rul.png")
    plot_error_vs_uncertainty(predictions_df, FD001_TEST_PLOTS_DIR / "error_vs_uncertainty.png")
    plot_health_state_confusion_matrix(matrix, FD001_TEST_PLOTS_DIR / "health_state_confusion_matrix.png")
    print(f"\nSaved 4 plots to {FD001_TEST_PLOTS_DIR}")

    # --- Decision engine (SYNTHETIC context - NASA provides none of this) --
    _section("Decision engine on test-set predictions (SYNTHETIC asset context, unchanged from Step 9)")
    print("NASA C-MAPSS has no criticality, redundancy or maintenance-lead-time data for these engines.")
    print("The profiles below are the SAME invented demonstrations used in Step 9 - not derived from this test set.")
    for name, base_context in SYNTHETIC_PROFILES.items():
        decided = decide_for_dataframe_with_confidence(predictions_df, base_context)
        counts = decided["recommended_action"].value_counts()
        print(f"\n{name}: {dict(counts)}")

    # --- Save --------------------------------------------------------
    save_columns = [
        "unit_number", "last_observed_cycle", "actual_rul_uncapped", "actual_rul", "predicted_rul",
        "error", "absolute_error", "actual_health_state", "health_state",
        "prediction_std", "prediction_p10", "prediction_p90", "prediction_range", "model_confidence",
    ]
    predictions_df[save_columns].to_csv(FD001_TEST_PREDICTIONS_PATH, index=False)
    matrix.to_csv(FD001_TEST_HEALTH_CONFUSION_PATH)
    FD001_TEST_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FD001_TEST_METRICS_PATH.write_text(
        json.dumps({**test_metrics, "nasa_cmapss_score": nasa_score, "empirical_coverage": coverage}, indent=2)
    )
    print(f"\nSaved {FD001_TEST_PREDICTIONS_PATH}")
    print(f"Saved {FD001_TEST_METRICS_PATH}")
    print(f"Saved {FD001_TEST_HEALTH_CONFUSION_PATH}")

    # --- Full pipeline summary ------------------------------------------
    _section("Full pipeline")
    print(
        "NASA training telemetry\n"
        "        |\n"
        "feature engineering (Steps 4, 6)\n"
        "        |\n"
        "Random Forest (Steps 5-6)\n"
        "        |\n"
        "predicted RUL\n"
        "        |\n"
        "ensemble uncertainty (Step 10)\n"
        "        |\n"
        "health state (Step 8)\n"
        "        |\n"
        "deterministic decision logic (Step 9)\n"
        "        |\n"
        "maintenance recommendation"
    )
    print(
        "\nNASA-supported by this test run: telemetry, RUL modelling, prediction error, ensemble\n"
        "uncertainty, and health-state comparison - all evaluated against real NASA ground truth above.\n"
        "Still SYNTHETIC: criticality, redundancy, maintenance lead time, and therefore the final\n"
        "maintenance recommendation itself - those numbers are demonstrations of the architecture,\n"
        "not NASA-validated facts about these engines."
    )
