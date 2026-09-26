"""Tests for src/pipeline_explorer.py - the Pipeline Explorer's reduced decision context."""

import json

import pytest

from src.pipeline_explorer import (
    EXPLORER_RULE_IDS,
    PARITY_CASES_PATH,
    build_parity_cases,
    decision_summary,
    explorer_decision,
    parity_document,
)

DEFAULT_THRESHOLDS = {"action": 15, "plan": 30, "watch": 60}


def test_healthy_high_confidence_is_no_action():
    decision = explorer_decision(90, "HIGH", DEFAULT_THRESHOLDS, 15)
    assert decision.health_state == "HEALTHY"
    assert decision.recommended_action.name == "NO_ACTION"
    assert decision.rule_ids == ("BASE_01",)


def test_inside_lead_time_intervenes_now():
    decision = explorer_decision(12, "HIGH", DEFAULT_THRESHOLDS, 15)
    assert decision.recommended_action.name == "INTERVENE_NOW"
    assert "LEAD_01" in decision.rule_ids
    assert decision.requires_human_review is True


def test_within_buffer_schedules():
    decision = explorer_decision(18, "HIGH", DEFAULT_THRESHOLDS, 15)
    assert decision.health_state == "PLAN"
    assert decision.recommended_action.name == "SCHEDULE_MAINTENANCE"
    assert decision.rule_ids == ("BASE_01", "LEAD_02")


def test_low_confidence_drops_to_inspect_even_inside_lead_time():
    # With criticality excluded there is no CONF_02 override: low confidence
    # always means "seek confirmation first" for PLAN/ACTION.
    decision = explorer_decision(12, "LOW", DEFAULT_THRESHOLDS, 15)
    assert decision.recommended_action.name == "INSPECT"
    assert decision.rule_ids == ("BASE_01", "LEAD_01", "CONF_01")


def test_edited_thresholds_change_the_state():
    assert explorer_decision(40, "HIGH", DEFAULT_THRESHOLDS, 0).health_state == "WATCH"
    assert explorer_decision(40, "HIGH", {"action": 20, "plan": 45, "watch": 90}, 0).health_state == "PLAN"


def test_unordered_thresholds_rejected():
    with pytest.raises(ValueError):
        explorer_decision(40, "HIGH", {"action": 30, "plan": 20, "watch": 60}, 15)


def test_excluded_rules_never_fire_across_parity_grid():
    for case in build_parity_cases():
        assert set(case["expected"]["rule_ids"]) <= set(EXPLORER_RULE_IDS)


def test_parity_grid_exercises_every_explorer_rule():
    fired = {rule for case in build_parity_cases() for rule in case["expected"]["rule_ids"]}
    assert fired == set(EXPLORER_RULE_IDS)


def test_summary_has_no_criticality_or_redundancy():
    summary = decision_summary(explorer_decision(20, "MEDIUM", DEFAULT_THRESHOLDS, 15))
    assert "criticality" not in summary and "redundancy" not in summary
    assert summary["trace"][0]["rule_id"] == "BASE_01"


def test_committed_parity_file_is_current():
    """If this fails, the Python engine changed: run `python -m src.pipeline_explorer`
    and make the TypeScript port in app/src/pipeline/rules.ts pass again."""
    committed = json.loads(PARITY_CASES_PATH.read_text())
    assert committed == json.loads(json.dumps(parity_document()))
