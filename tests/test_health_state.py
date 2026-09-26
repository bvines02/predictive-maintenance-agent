"""Tests for src/health_state.py."""

import math

import pandas as pd
import pytest

from src.config import HEALTH_THRESHOLDS
from src.health_state import (
    HEALTH_STATES,
    STATE_SEVERITY,
    add_health_states,
    assign_health_state,
    assign_health_states,
    conservative_over_classifications,
    serious_under_classifications,
    state_confusion_matrix,
    state_distribution,
    state_metrics,
)


@pytest.mark.parametrize(
    "rul, expected",
    [
        (61, "HEALTHY"),
        (60.5, "HEALTHY"),
        (60, "WATCH"),
        (31, "WATCH"),
        (30.5, "WATCH"),
        (30, "PLAN"),
        (16, "PLAN"),
        (15.01, "PLAN"),
        (15, "ACTION"),
        (0, "ACTION"),
    ],
)
def test_boundary_conditions(rul, expected):
    assert assign_health_state(rul) == expected


def test_negative_prediction_is_clamped_to_action():
    assert assign_health_state(-7.3) == "ACTION"


def test_nan_prediction_raises_instead_of_guessing():
    with pytest.raises(ValueError, match="NaN"):
        assign_health_state(float("nan"))
    with pytest.raises(ValueError, match="NaN"):
        assign_health_states(pd.Series([50.0, float("nan")]))


def test_exactly_one_valid_state_per_value():
    states = assign_health_states(pd.Series(range(-5, 130)))

    assert set(states) <= set(HEALTH_STATES)
    assert len(states) == 135


def test_thresholds_are_configurable():
    custom = {"watch": 100, "plan": 50, "action": 20}

    assert assign_health_state(90, custom) == "WATCH"
    assert assign_health_state(90) == "HEALTHY"  # default 60 threshold unchanged


def test_misordered_thresholds_are_rejected():
    with pytest.raises(ValueError, match="action < plan < watch"):
        assign_health_state(10, {"watch": 30, "plan": 60, "action": 15})


def test_missing_threshold_key_is_rejected():
    with pytest.raises(ValueError, match="missing"):
        assign_health_state(10, {"watch": 60, "plan": 30})


def test_default_thresholds_come_from_config():
    assert HEALTH_THRESHOLDS == {"watch": 60, "plan": 30, "action": 15}


def test_severity_mapping_is_ordinal():
    assert [STATE_SEVERITY[s] for s in HEALTH_STATES] == [0, 1, 2, 3]


def _predictions(rows):
    df = pd.DataFrame(rows, columns=["unit_number", "time_cycles", "actual_rul", "predicted_rul"])
    df["error"] = df["predicted_rul"] - df["actual_rul"]
    return add_health_states(df)


def test_state_error_sign_negative_means_less_concerned():
    df = _predictions([(1, 1, 10, 70)])  # truly ACTION, predicted HEALTHY

    assert df.loc[0, "actual_health_state"] == "ACTION"
    assert df.loc[0, "health_state"] == "HEALTHY"
    assert df.loc[0, "state_error"] == -3


def test_state_error_sign_positive_means_more_conservative():
    df = _predictions([(1, 1, 70, 10)])  # truly HEALTHY, predicted ACTION

    assert df.loc[0, "state_error"] == 3


def test_add_health_states_does_not_mutate_input():
    raw = pd.DataFrame({"unit_number": [1], "time_cycles": [1], "actual_rul": [5], "predicted_rul": [5]})

    add_health_states(raw)

    assert "health_state" not in raw.columns


def test_confusion_matrix_has_all_states_and_counts_every_row():
    df = _predictions([(1, 1, 5, 5), (1, 2, 10, 40), (1, 3, 70, 70)])

    matrix = state_confusion_matrix(df)

    assert list(matrix.index) == HEALTH_STATES
    assert list(matrix.columns) == HEALTH_STATES
    assert matrix.values.sum() == 3
    assert matrix.loc["ACTION", "ACTION"] == 1
    assert matrix.loc["ACTION", "WATCH"] == 1
    assert matrix.loc["HEALTHY", "HEALTHY"] == 1


def test_state_distribution_includes_empty_states():
    dist = state_distribution(pd.Series(["ACTION", "ACTION"]))

    assert dist["ACTION"] == 2
    assert dist["HEALTHY"] == 0


def test_state_metrics_match_hand_calculation():
    # (actual, predicted): ACTION/ACTION, ACTION/PLAN, ACTION/HEALTHY,
    # PLAN/ACTION, HEALTHY/ACTION, HEALTHY/HEALTHY
    df = _predictions(
        [(1, 1, 5, 5), (1, 2, 5, 20), (1, 3, 5, 70), (1, 4, 20, 5), (1, 5, 70, 5), (1, 6, 70, 70)]
    )

    m = state_metrics(df)

    assert m["action_recall"] == pytest.approx(1 / 3)  # 1 of 3 actual ACTION
    assert m["action_precision"] == pytest.approx(1 / 3)  # 1 of 3 predicted ACTION
    assert m["plan_or_action_recall"] == pytest.approx(3 / 4)  # rows 1-4 truly PLAN/ACTION; 1,2,4 predicted PLAN/ACTION
    assert m["pct_actual_action_missed_as_healthy_or_watch"] == pytest.approx(100 / 3)
    assert m["pct_actual_healthy_flagged_action"] == pytest.approx(50.0)
    assert m["state_accuracy"] == pytest.approx(2 / 6)


def test_state_metrics_undefined_ratio_is_nan_not_zero():
    df = _predictions([(1, 1, 70, 70)])  # no actual ACTION rows at all

    assert math.isnan(state_metrics(df)["action_recall"])


def test_serious_under_classifications_sorted_by_severity_gap():
    df = _predictions([(1, 1, 5, 70), (1, 2, 20, 70), (1, 3, 40, 70), (1, 4, 70, 5)])

    under = serious_under_classifications(df)

    assert under["state_error"].tolist() == [-3, -2, -1]
    assert (under["state_error"] < 0).all()


def test_conservative_over_classifications_sorted_and_positive_only():
    df = _predictions([(1, 1, 70, 5), (1, 2, 40, 5), (1, 3, 5, 70)])

    over = conservative_over_classifications(df)

    assert over["state_error"].tolist() == [3, 2]
