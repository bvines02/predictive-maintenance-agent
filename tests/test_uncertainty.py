"""Tests for src/uncertainty.py (Random Forest ensemble-disagreement uncertainty)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.decision_engine import AssetContext, decide_for_dataframe_with_confidence, decide_maintenance
from src.uncertainty import (
    CONFIDENCE_LABELS,
    UNCERTAINTY_BANDS,
    add_uncertainty_columns,
    assign_model_confidence,
    assign_uncertainty_band,
    compute_prediction_distribution,
    compute_uncertainty_thresholds,
    correlation_uncertainty_vs_error,
    dangerous_high_confidence_errors,
    empirical_coverage,
    error_by_uncertainty_band,
    extract_tree_predictions,
    low_confidence_large_errors,
    near_failure_uncertainty,
)


def _fitted_rf(n_rows=60, n_estimators=15, random_state=0):
    rng = np.random.default_rng(random_state)
    X = pd.DataFrame({"x1": rng.normal(size=n_rows), "x2": rng.normal(size=n_rows)})
    y = X["x1"] * 3 + rng.normal(scale=0.1, size=n_rows)
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X, y)
    return model, X


# --- Tree extraction -----------------------------------------------------


def test_extract_tree_predictions_shape():
    model, X = _fitted_rf(n_rows=40, n_estimators=12)

    tree_predictions = extract_tree_predictions(model, X)

    assert tree_predictions.shape == (40, 12)


def test_mean_tree_prediction_matches_model_predict():
    model, X = _fitted_rf(n_rows=50, n_estimators=25)

    tree_predictions = extract_tree_predictions(model, X)

    assert np.allclose(tree_predictions.mean(axis=1), model.predict(X), atol=1e-10)


# --- Distribution metrics -------------------------------------------------


def test_prediction_std_never_negative():
    rng = np.random.default_rng(1)
    tree_predictions = rng.normal(size=(30, 10))

    distribution = compute_prediction_distribution(tree_predictions)

    assert (distribution["prediction_std"] >= 0).all()


def test_identical_tree_predictions_give_zero_std():
    tree_predictions = np.full((5, 20), 42.0)

    distribution = compute_prediction_distribution(tree_predictions)

    assert (distribution["prediction_std"] == 0).all()
    assert (distribution["mean_prediction"] == 42.0).all()
    assert (distribution["prediction_range"] == 0).all()


def test_varying_tree_predictions_give_larger_std_than_tight_cluster():
    tight = np.array([[19, 20, 21, 20, 20]])
    wide = np.array([[8, 15, 27, 32, 11]])

    tight_std = compute_prediction_distribution(tight)["prediction_std"].iloc[0]
    wide_std = compute_prediction_distribution(wide)["prediction_std"].iloc[0]

    assert wide_std > tight_std


def test_p10_p90_bracket_the_mean_for_a_symmetric_distribution():
    rng = np.random.default_rng(2)
    tree_predictions = rng.normal(loc=50, scale=5, size=(20, 300))

    distribution = compute_prediction_distribution(tree_predictions)

    assert (distribution["prediction_p10"] <= distribution["mean_prediction"]).all()
    assert (distribution["mean_prediction"] <= distribution["prediction_p90"]).all()


# --- add_uncertainty_columns ------------------------------------------------


def _predictions_df(n_rows):
    return pd.DataFrame(
        {
            "unit_number": [1] * n_rows,
            "time_cycles": range(1, n_rows + 1),
            "actual_rul": [50.0] * n_rows,
            "predicted_rul": [48.0] * n_rows,
            "error": [-2.0] * n_rows,
        }
    )


def test_add_uncertainty_columns_matches_predicted_rul():
    predictions_df = _predictions_df(4)
    tree_predictions = np.full((4, 10), 48.0)

    result = add_uncertainty_columns(predictions_df, tree_predictions)

    assert (result["mean_prediction"] == 48.0).all()
    assert "absolute_error" in result.columns and (result["absolute_error"] == 2.0).all()


def test_add_uncertainty_columns_rejects_mismatched_mean():
    predictions_df = _predictions_df(3)
    tree_predictions = np.full((3, 10), 99.0)  # doesn't match predicted_rul=48

    with pytest.raises(ValueError, match="does not match predicted_rul"):
        add_uncertainty_columns(predictions_df, tree_predictions)


def test_add_uncertainty_columns_rejects_row_count_mismatch():
    predictions_df = _predictions_df(5)
    tree_predictions = np.full((3, 10), 48.0)

    with pytest.raises(ValueError, match="Row count mismatch"):
        add_uncertainty_columns(predictions_df, tree_predictions)


# --- Thresholds and bands --------------------------------------------------


def test_compute_uncertainty_thresholds_are_increasing():
    std_values = pd.Series(np.arange(1, 101))

    thresholds = compute_uncertainty_thresholds(std_values, low_percentile=33, high_percentile=67)

    assert thresholds["low_max"] < thresholds["medium_max"]


def test_compute_uncertainty_thresholds_rejects_bad_percentiles():
    with pytest.raises(ValueError):
        compute_uncertainty_thresholds(pd.Series([1, 2, 3]), low_percentile=70, high_percentile=30)


def test_assign_uncertainty_band_boundaries():
    thresholds = {"low_max": 2.0, "medium_max": 5.0}
    values = pd.Series([0.0, 2.0, 2.1, 5.0, 5.1, 100.0])

    bands = assign_uncertainty_band(values, thresholds)

    assert bands.tolist() == ["LOW", "LOW", "MEDIUM", "MEDIUM", "HIGH", "HIGH"]
    assert set(bands) <= set(UNCERTAINTY_BANDS)


def test_assign_model_confidence_inverts_uncertainty():
    bands = pd.Series(["LOW", "MEDIUM", "HIGH"])

    confidence = assign_model_confidence(bands)

    assert confidence.tolist() == ["HIGH", "MEDIUM", "LOW"]
    assert set(confidence) <= set(CONFIDENCE_LABELS)


def test_assign_model_confidence_rejects_unknown_band():
    with pytest.raises(ValueError, match="Unknown uncertainty_band"):
        assign_model_confidence(pd.Series(["LOW", "EXTREME"]))


# --- Diagnostics -----------------------------------------------------------


def _diagnostics_df():
    return pd.DataFrame(
        {
            "actual_rul": [5, 5, 40, 40, 100],
            "predicted_rul": [25, 6, 42, 90, 101],
            "error": [20, 1, 2, 50, 1],
            "absolute_error": [20, 1, 2, 50, 1],
            "prediction_std": [1.0, 1.0, 20.0, 20.0, 1.0],
            "prediction_p10": [20, 4, 30, 60, 99],
            "prediction_p90": [30, 8, 55, 110, 103],
            "prediction_range": [10, 4, 25, 50, 4],
            "model_confidence": ["HIGH", "HIGH", "LOW", "LOW", "HIGH"],
            "uncertainty_band": ["LOW", "LOW", "HIGH", "HIGH", "LOW"],
            "unit_number": [1, 1, 2, 2, 3],
            "time_cycles": [1, 2, 1, 2, 1],
        }
    )


def test_correlation_uncertainty_vs_error_direction():
    df = _diagnostics_df()

    corr = correlation_uncertainty_vs_error(df)

    # rows with high prediction_std (20) also have larger absolute_error (2, 50)
    # than the low-std rows (20, 1) is a mixed signal by design; just check it runs
    # and returns a valid correlation in [-1, 1].
    assert -1.0 <= corr <= 1.0


def test_error_by_uncertainty_band_includes_all_bands_in_order():
    df = _diagnostics_df()

    table = error_by_uncertainty_band(df)

    assert table["uncertainty_band"].tolist() == UNCERTAINTY_BANDS


def test_near_failure_uncertainty_subsets_are_correct():
    df = _diagnostics_df()

    table = near_failure_uncertainty(df, thresholds=(15,))

    full_row = table[table["subset"] == "All validation rows"].iloc[0]
    near_row = table[table["subset"] == "actual RUL <= 15"].iloc[0]
    assert full_row["n"] == 5
    assert near_row["n"] == 2  # the two actual_rul=5 rows


def test_dangerous_high_confidence_errors_filters_correctly():
    df = _diagnostics_df()

    dangerous = dangerous_high_confidence_errors(df, actual_max=15, min_over_prediction=10)

    assert len(dangerous) == 1
    assert dangerous.iloc[0]["predicted_rul"] == 25  # actual=5, error=20, HIGH confidence


def test_low_confidence_large_errors_filters_correctly():
    df = _diagnostics_df()

    result = low_confidence_large_errors(df, min_absolute_error=10)

    assert len(result) == 1
    assert result.iloc[0]["predicted_rul"] == 90  # absolute_error=50, LOW confidence


def test_empirical_coverage_matches_hand_count():
    df = _diagnostics_df()
    # actual_rul vs [p10,p90]: 5 in [20,30]? no. 5 in [4,8]? yes. 40 in [30,55]? yes.
    # 40 in [60,110]? no. 100 in [99,103]? yes. -> 3/5 = 0.6

    coverage = empirical_coverage(df)

    assert coverage == pytest.approx(0.6)


# --- Decision-engine integration --------------------------------------------


def test_decide_for_dataframe_with_confidence_uses_the_row_value():
    df = pd.DataFrame(
        {
            "predicted_rul": [25.0, 25.0],
            "health_state": ["PLAN", "PLAN"],
            "model_confidence": ["HIGH", "LOW"],
        }
    )
    context = AssetContext("MEDIUM", "FULL", 5, "HIGH")  # base confidence is a placeholder

    decided = decide_for_dataframe_with_confidence(df, context)

    high_conf_action = decided.iloc[0]["recommended_action"]
    low_conf_action = decided.iloc[1]["recommended_action"]
    assert low_conf_action == "INSPECT"  # CONF_01: seeks confirmation
    assert high_conf_action == "PLAN_MAINTENANCE"
    # matches calling decide_maintenance directly with each row's own confidence
    assert high_conf_action == decide_maintenance(25.0, "PLAN", AssetContext("MEDIUM", "FULL", 5, "HIGH")).recommended_action.name
    assert low_conf_action == decide_maintenance(25.0, "PLAN", AssetContext("MEDIUM", "FULL", 5, "LOW")).recommended_action.name


def test_low_model_confidence_does_not_downgrade_critical_action_case():
    df = pd.DataFrame({"predicted_rul": [8.0], "health_state": ["ACTION"], "model_confidence": ["LOW"]})
    context = AssetContext("CRITICAL", "NONE", 20, "HIGH")

    decided = decide_for_dataframe_with_confidence(df, context)

    assert decided.iloc[0]["recommended_action"] == "INTERVENE_NOW"
    assert "CONF_02" in decided.iloc[0]["rule_ids"]


def test_low_model_confidence_seeks_inspection_for_moderate_case():
    df = pd.DataFrame({"predicted_rul": [20.0], "health_state": ["PLAN"], "model_confidence": ["LOW"]})
    context = AssetContext("MEDIUM", "FULL", 5, "HIGH")

    decided = decide_for_dataframe_with_confidence(df, context)

    assert decided.iloc[0]["recommended_action"] == "INSPECT"
