"""Tests for src/decision_engine.py (deterministic maintenance decision engine)."""

import itertools
import math

import pandas as pd
import pytest

from src.decision_engine import (
    ACTION_MEANINGS,
    AssetContext,
    Confidence,
    Criticality,
    DecisionConfig,
    MaintenanceAction,
    Redundancy,
    decide_for_dataframe,
    decide_maintenance,
    get_base_action,
)
from src.health_state import HEALTH_STATES, assign_health_state
from src.synthetic_asset_profiles import SYNTHETIC_PROFILES

A = MaintenanceAction


def ctx(criticality, redundancy, lead, confidence="HIGH"):
    return AssetContext(criticality, redundancy, lead, confidence)


# --- Vocabulary and validation ----------------------------------------------


def test_action_severity_hierarchy_is_ordered():
    assert [int(a) for a in A] == [0, 1, 2, 3, 4, 5]
    assert A.NO_ACTION < A.MONITOR < A.INSPECT < A.PLAN_MAINTENANCE < A.SCHEDULE_MAINTENANCE < A.INTERVENE_NOW


def test_every_action_has_a_meaning():
    assert set(ACTION_MEANINGS) == set(A)


def test_context_accepts_case_insensitive_strings():
    c = AssetContext("high", "none", 10, "low")
    assert (c.criticality, c.redundancy, c.confidence) == (Criticality.HIGH, Redundancy.NONE, Confidence.LOW)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"criticality": "SEVERE"},
        {"redundancy": "SOME"},
        {"confidence": "SURE"},
        {"maintenance_lead_time_cycles": -1},
        {"maintenance_lead_time_cycles": float("nan")},
        {"maintenance_lead_time_cycles": "20"},
        {"criticality": 2},
    ],
)
def test_invalid_context_is_rejected(kwargs):
    base = {"criticality": "LOW", "redundancy": "FULL", "maintenance_lead_time_cycles": 5, "confidence": "HIGH"}
    with pytest.raises(ValueError):
        AssetContext(**{**base, **kwargs})


def test_context_is_marked_synthetic_by_default():
    assert "SYNTHETIC" in ctx("LOW", "FULL", 5).source


def test_base_action_mapping():
    assert [get_base_action(s) for s in HEALTH_STATES] == [A.NO_ACTION, A.MONITOR, A.PLAN_MAINTENANCE, A.SCHEDULE_MAINTENANCE]


def test_unknown_health_state_rejected():
    with pytest.raises(ValueError):
        get_base_action("CALM")


# --- Input validation --------------------------------------------------------


def test_nan_prediction_is_rejected():
    with pytest.raises(ValueError):
        decide_maintenance(float("nan"), None, ctx("LOW", "FULL", 5))


def test_inconsistent_health_state_is_rejected():
    with pytest.raises(ValueError, match="inconsistent"):
        decide_maintenance(5.0, "HEALTHY", ctx("CRITICAL", "NONE", 20))


def test_health_state_is_derived_when_omitted():
    assert decide_maintenance(25.0, None, ctx("MEDIUM", "PARTIAL", 10)).health_state == "PLAN"


def test_negative_prediction_treated_as_zero_remaining_life():
    d = decide_maintenance(-3.0, None, ctx("LOW", "NONE", 5))
    assert d.health_state == "ACTION" and d.recommended_action == A.INTERVENE_NOW


# --- Scenarios A-F -----------------------------------------------------------


def test_scenario_a_healthy_low_risk():
    d = decide_maintenance(100, "HEALTHY", ctx("LOW", "FULL", 5, "HIGH"))
    assert d.recommended_action == A.NO_ACTION


def test_scenario_b_entering_planning_window():
    d = decide_maintenance(25, "PLAN", ctx("MEDIUM", "PARTIAL", 10, "HIGH"))
    assert d.recommended_action >= A.PLAN_MAINTENANCE


def test_scenario_c_high_consequence_no_redundancy_is_stronger_than_b():
    b = decide_maintenance(25, "PLAN", ctx("MEDIUM", "PARTIAL", 10, "HIGH"))
    c = decide_maintenance(25, "PLAN", ctx("HIGH", "NONE", 20, "HIGH"))
    assert c.recommended_action == A.SCHEDULE_MAINTENANCE
    assert c.recommended_action > b.recommended_action
    assert "LEAD_02" in c.rule_ids


def test_scenario_d_inside_lead_time():
    d = decide_maintenance(12, "ACTION", ctx("HIGH", "NONE", 20, "HIGH"))
    assert d.recommended_action == A.INTERVENE_NOW
    assert "LEAD_01" in d.rule_ids


def test_scenario_e_low_confidence_seeks_confirmation():
    d = decide_maintenance(25, "PLAN", ctx("MEDIUM", "FULL", 5, "LOW"))
    assert d.recommended_action == A.INSPECT
    assert "CONF_01" in d.rule_ids and d.de_escalation_reasons


def test_scenario_f_low_confidence_but_severe_consequence_stays_urgent():
    d = decide_maintenance(10, "ACTION", ctx("CRITICAL", "NONE", 20, "LOW"))
    assert d.recommended_action == A.INTERVENE_NOW
    assert "CONF_02" in d.rule_ids and "CONF_01" not in d.rule_ids


def test_scenario_g_redundancy_changes_consequence_not_condition():
    full = decide_maintenance(22, "PLAN", ctx("MEDIUM", "FULL", 20))
    none = decide_maintenance(22, "PLAN", ctx("MEDIUM", "NONE", 20))
    assert none.recommended_action == A.SCHEDULE_MAINTENANCE
    assert full.recommended_action == A.PLAN_MAINTENANCE
    assert "RED_01" in full.rule_ids
    assert full.health_state == none.health_state  # equipment condition is identical


# --- Lead-time boundaries ------------------------------------------------------


@pytest.mark.parametrize(
    "rul, expected_rule",
    [(20, "LEAD_01"), (25, "LEAD_02"), (25.5, None)],
)
def test_lead_time_boundaries(rul, expected_rule):
    d = decide_maintenance(rul, None, ctx("MEDIUM", "NONE", 20))
    fired = [r for r in d.rule_ids if r.startswith("LEAD")]
    assert fired == ([expected_rule] if expected_rule else [])


def test_lead_time_can_escalate_a_healthy_state():
    d = decide_maintenance(70, "HEALTHY", ctx("MEDIUM", "NONE", 90))
    assert d.recommended_action == A.INTERVENE_NOW  # work takes 90 cycles, failure predicted in 70


def test_lead_time_buffer_is_configurable():
    c = ctx("MEDIUM", "NONE", 10)
    assert "LEAD_02" not in decide_maintenance(25, "PLAN", c).rule_ids
    wide = DecisionConfig(lead_time_buffer_cycles=20)
    assert "LEAD_02" in decide_maintenance(25, "PLAN", c, wide).rule_ids


# --- Criticality rules ---------------------------------------------------------


def test_high_criticality_watch_becomes_inspect():
    assert decide_maintenance(45, "WATCH", ctx("HIGH", "NONE", 5)).recommended_action == A.INSPECT
    assert decide_maintenance(45, "WATCH", ctx("MEDIUM", "NONE", 5)).recommended_action == A.MONITOR


def test_critical_plan_becomes_at_least_schedule():
    d = decide_maintenance(28, "PLAN", ctx("CRITICAL", "FULL", 5))
    assert d.recommended_action >= A.SCHEDULE_MAINTENANCE
    assert "CRIT_02" in d.rule_ids


def test_critical_action_becomes_intervene_now():
    assert decide_maintenance(14, "ACTION", ctx("CRITICAL", "FULL", 1)).recommended_action == A.INTERVENE_NOW


# --- Redundancy guardrails -----------------------------------------------------


def test_full_redundancy_never_reduces_action_state():
    d = decide_maintenance(3, "ACTION", ctx("LOW", "FULL", 5))
    assert d.recommended_action == A.INTERVENE_NOW
    assert "RED_02" in d.rule_ids and "RED_01" not in d.rule_ids


def test_full_redundancy_never_reduces_serious_criticality():
    d = decide_maintenance(22, "PLAN", ctx("HIGH", "FULL", 20))
    assert d.recommended_action == A.SCHEDULE_MAINTENANCE
    assert "RED_02" in d.rule_ids


def test_partial_and_no_redundancy_give_no_reduction():
    for redundancy in ("PARTIAL", "NONE"):
        d = decide_maintenance(22, "PLAN", ctx("MEDIUM", redundancy, 20))
        assert d.recommended_action == A.SCHEDULE_MAINTENANCE
        assert not any(r.startswith("RED") for r in d.rule_ids)


def test_full_redundancy_reduces_by_exactly_one_level():
    # WATCH state (base MONITOR) driven up by a long lead time.
    within_buffer = decide_maintenance(50, "WATCH", ctx("LOW", "FULL", 45))  # LEAD_02 -> SCHEDULE
    inside_lead = decide_maintenance(44, "WATCH", ctx("LOW", "FULL", 45))  # LEAD_01 -> INTERVENE_NOW
    assert within_buffer.recommended_action == A.PLAN_MAINTENANCE
    assert inside_lead.recommended_action == A.SCHEDULE_MAINTENANCE
    assert within_buffer.recommended_action > within_buffer.base_action  # never cut below base action


# --- Confidence rules ----------------------------------------------------------


def test_medium_and_high_confidence_do_not_change_the_action():
    base = decide_maintenance(25, "PLAN", ctx("MEDIUM", "PARTIAL", 5, "HIGH")).recommended_action
    assert decide_maintenance(25, "PLAN", ctx("MEDIUM", "PARTIAL", 5, "MEDIUM")).recommended_action == base


def test_low_confidence_does_not_affect_watch_or_healthy():
    assert decide_maintenance(45, "WATCH", ctx("LOW", "NONE", 5, "LOW")).recommended_action == A.MONITOR
    assert decide_maintenance(100, "HEALTHY", ctx("LOW", "NONE", 5, "LOW")).recommended_action == A.NO_ACTION


def test_low_confidence_outside_lead_time_is_downgraded_even_when_critical():
    d = decide_maintenance(28, "PLAN", ctx("CRITICAL", "NONE", 20, "LOW"))
    assert d.recommended_action == A.INSPECT and "CONF_01" in d.rule_ids


# --- Trace and structure ---------------------------------------------------------


def test_result_is_traceable():
    d = decide_maintenance(22, "PLAN", ctx("HIGH", "NONE", 18))
    assert d.rule_ids[0] == "BASE_01"
    assert d.base_action == A.PLAN_MAINTENANCE and d.recommended_action == A.SCHEDULE_MAINTENANCE
    assert d.escalation_reasons
    as_dict = d.to_dict()
    assert as_dict["recommended_action"] == "SCHEDULE_MAINTENANCE"
    assert {"rule_ids", "escalation_reasons", "criticality", "confidence", "context_source"} <= set(as_dict)


def test_decisions_are_deterministic():
    context = ctx("HIGH", "PARTIAL", 15)
    assert decide_maintenance(19, "PLAN", context) == decide_maintenance(19, "PLAN", context)


def test_human_review_required_for_intrusive_actions():
    assert decide_maintenance(10, "ACTION", ctx("LOW", "NONE", 1)).requires_human_review
    assert not decide_maintenance(100, "HEALTHY", ctx("LOW", "NONE", 1)).requires_human_review


# --- Exhaustive guardrail property test ------------------------------------------


def test_invariants_hold_across_the_whole_input_grid():
    checked = 0
    for rul, crit, red, conf, lead in itertools.product(
        [0, 5, 14.9, 15, 22, 30, 31, 45, 60, 61, 100],
        Criticality,
        Redundancy,
        Confidence,
        [0, 5, 15, 25, 90],
    ):
        d = decide_maintenance(rul, None, AssetContext(crit, red, lead, conf))
        state = assign_health_state(rul)

        if state in ("PLAN", "ACTION"):
            assert d.recommended_action >= A.INSPECT  # warning is never ignored
        if "CONF_01" not in d.rule_ids:
            assert d.recommended_action >= d.base_action  # lowering only via explicit rule
        if state == "ACTION" and conf != Confidence.LOW:
            assert d.recommended_action >= A.SCHEDULE_MAINTENANCE  # redundancy can't cut ACTION
        if crit == Criticality.CRITICAL and state == "ACTION" and "CONF_01" not in d.rule_ids:
            assert d.recommended_action == A.INTERVENE_NOW
        if rul <= lead and crit >= Criticality.HIGH:
            assert d.recommended_action == A.INTERVENE_NOW  # inside lead time, serious consequence
        checked += 1
    assert checked == 11 * 4 * 3 * 3 * 5


# --- DataFrame wrapper and profiles -----------------------------------------------


def _predictions():
    df = pd.DataFrame({"predicted_rul": [100.0, 45.0, 25.0, 12.0], "actual_rul": [90, 50, 20, 15]})
    df["health_state"] = [assign_health_state(x) for x in df["predicted_rul"]]
    return df


def test_decide_for_dataframe_preserves_rows_and_predictions():
    df = _predictions()
    out = decide_for_dataframe(df, ctx("HIGH", "NONE", 15))
    assert len(out) == len(df)
    assert out["predicted_rul"].tolist() == df["predicted_rul"].tolist()
    assert not any(c in df.columns for c in ("recommended_action", "rule_ids"))  # input untouched


def test_decide_for_dataframe_matches_scalar_decisions():
    df = _predictions()
    context = ctx("CRITICAL", "NONE", 25)
    out = decide_for_dataframe(df, context)
    for rul, state, action in zip(df["predicted_rul"], df["health_state"], out["recommended_action"]):
        assert decide_maintenance(rul, state, context).recommended_action.name == action


def test_same_predictions_different_context_gives_different_actions():
    df = _predictions()
    low = decide_for_dataframe(df, SYNTHETIC_PROFILES["P1 non-critical redundant"])
    crit = decide_for_dataframe(df, SYNTHETIC_PROFILES["P3 critical single point"])
    assert low["predicted_rul"].tolist() == crit["predicted_rul"].tolist()
    assert low["recommended_action"].tolist() != crit["recommended_action"].tolist()


def test_synthetic_profiles_are_marked_synthetic():
    assert len(SYNTHETIC_PROFILES) == 3
    assert all("SYNTHETIC" in c.source for c in SYNTHETIC_PROFILES.values())
