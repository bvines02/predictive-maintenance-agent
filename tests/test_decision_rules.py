"""Tests for src/decision_rules.py.

One test per branch, so every possible decision path is exercised directly -
this is exactly what "deterministic rules" buys you: each outcome is
provable by a single, simple test rather than inferred from examples.
"""

import pytest

from src.decision_rules import decide_maintenance_action


def test_high_risk_critical_spof_no_redundancy_is_immediate():
    result = decide_maintenance_action(
        failure_probability=0.85,
        risk_band="High",
        asset_criticality="High",
        is_single_point_of_failure=True,
        redundancy_available=False,
    )

    assert result["recommended_action"] == "Immediate engineering review"
    assert result["urgency"] == "Immediate"
    assert result["human_review_required"] is True
    assert result["rationale_code"] == "HIGH_RISK_CRITICAL_SPOF_NO_REDUNDANCY"


def test_high_risk_critical_with_redundancy_is_24_48_hours():
    result = decide_maintenance_action(
        failure_probability=0.85,
        risk_band="High",
        asset_criticality="High",
        is_single_point_of_failure=True,
        redundancy_available=True,
    )

    assert result["recommended_action"] == "Inspect within 24-48 hours"
    assert result["urgency"] == "High"
    assert result["human_review_required"] is True
    assert result["rationale_code"] == "HIGH_RISK_CRITICAL_REDUNDANCY_AVAILABLE"


def test_high_risk_low_criticality_is_standard_high_risk():
    result = decide_maintenance_action(
        failure_probability=0.85,
        risk_band="High",
        asset_criticality="Low",
        is_single_point_of_failure=False,
        redundancy_available=False,
    )

    assert result["recommended_action"] == "Inspect within 24-48 hours"
    assert result["urgency"] == "High"
    assert result["human_review_required"] is True
    assert result["rationale_code"] == "HIGH_RISK_STANDARD"


def test_medium_risk_high_criticality_requires_human_review():
    result = decide_maintenance_action(
        failure_probability=0.55,
        risk_band="Medium",
        asset_criticality="High",
        is_single_point_of_failure=False,
        redundancy_available=True,
    )

    assert result["recommended_action"] == "Plan inspection in next maintenance window"
    assert result["urgency"] == "Medium"
    assert result["human_review_required"] is True
    assert result["rationale_code"] == "MEDIUM_RISK_CRITICAL_PLANNED_INSPECTION"


def test_medium_risk_low_criticality_does_not_require_human_review():
    result = decide_maintenance_action(
        failure_probability=0.55,
        risk_band="Medium",
        asset_criticality="Low",
        is_single_point_of_failure=False,
        redundancy_available=True,
    )

    assert result["recommended_action"] == "Plan inspection in next maintenance window"
    assert result["urgency"] == "Medium"
    assert result["human_review_required"] is False
    assert result["rationale_code"] == "MEDIUM_RISK_PLANNED_INSPECTION"


def test_low_risk_continues_monitoring_regardless_of_criticality():
    result = decide_maintenance_action(
        failure_probability=0.1,
        risk_band="Low",
        asset_criticality="High",
        is_single_point_of_failure=True,
        redundancy_available=False,
    )

    assert result["recommended_action"] == "Continue monitoring"
    assert result["urgency"] == "Routine"
    assert result["human_review_required"] is False
    assert result["rationale_code"] == "LOW_RISK_CONTINUE_MONITORING"


def test_unknown_risk_band_raises_value_error():
    with pytest.raises(ValueError, match="Unknown risk_band"):
        decide_maintenance_action(
            failure_probability=0.5,
            risk_band="Extreme",
            asset_criticality="Low",
            is_single_point_of_failure=False,
            redundancy_available=False,
        )


def test_unknown_criticality_raises_value_error():
    with pytest.raises(ValueError, match="Unknown asset_criticality"):
        decide_maintenance_action(
            failure_probability=0.5,
            risk_band="Low",
            asset_criticality="Extreme",
            is_single_point_of_failure=False,
            redundancy_available=False,
        )
