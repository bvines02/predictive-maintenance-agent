"""Stage 9 - Deterministic maintenance decision rules.

Responsible for:
- Converting a risk score + asset context into a maintenance action
- Deciding whether escalation and human review are required

Rules are plain, testable Python. The LLM never makes or overrides these
decisions - it only explains them (see explainer.py). See CLAUDE.md's
"Engineering principles" section for why this ordering matters.
"""

RISK_BANDS = ("Low", "Medium", "High")
CRITICALITY_LEVELS = ("Low", "Medium", "High")


def decide_maintenance_action(
    failure_probability: float,
    risk_band: str,
    asset_criticality: str,
    is_single_point_of_failure: bool,
    redundancy_available: bool,
) -> dict:
    """Turn a risk assessment + asset context into a maintenance decision.

    This function is the only place recommended actions are decided. It is
    deliberately simple, explicit branches rather than a scoring formula, so
    every possible outcome can be read directly from the code and tested.

    Args:
        failure_probability: The model's predicted failure probability, 0-1.
            Carried through into the rationale for traceability; the branch
            taken is driven by risk_band, not by re-thresholding this here.
        risk_band: "Low", "Medium", or "High" (see predict.py's risk_band()).
        asset_criticality: "Low", "Medium", or "High" - how much this asset
            matters to operations if it fails.
        is_single_point_of_failure: True if there is no redundant path
            around this asset - its failure stops the process it's part of.
        redundancy_available: True if a standby unit or workaround exists.

    Returns:
        {
            "recommended_action": str,
            "urgency": "Immediate" | "High" | "Medium" | "Routine",
            "human_review_required": bool,
            "rationale_code": str,
        }

    Raises:
        ValueError: If risk_band or asset_criticality isn't a recognized value.
            Fails loudly rather than silently falling through to a default -
            an unrecognized value here is a bug worth surfacing, not guessing past.
    """
    if risk_band not in RISK_BANDS:
        raise ValueError(f"Unknown risk_band: {risk_band!r}. Expected one of {RISK_BANDS}.")
    if asset_criticality not in CRITICALITY_LEVELS:
        raise ValueError(
            f"Unknown asset_criticality: {asset_criticality!r}. "
            f"Expected one of {CRITICALITY_LEVELS}."
        )

    is_high_criticality = asset_criticality == "High"

    # --- High risk ---------------------------------------------------------
    if risk_band == "High":
        if is_high_criticality and is_single_point_of_failure and not redundancy_available:
            # Worst case: likely to fail soon, matters a lot, and nothing
            # else can take over if it does.
            return {
                "recommended_action": "Immediate engineering review",
                "urgency": "Immediate",
                "human_review_required": True,
                "rationale_code": "HIGH_RISK_CRITICAL_SPOF_NO_REDUNDANCY",
            }

        if is_high_criticality and redundancy_available:
            # Still high risk and high criticality, but a standby unit
            # buys time to inspect rather than act immediately.
            return {
                "recommended_action": "Inspect within 24-48 hours",
                "urgency": "High",
                "human_review_required": True,
                "rationale_code": "HIGH_RISK_CRITICAL_REDUNDANCY_AVAILABLE",
            }

        # High risk, but not (high criticality + single point of failure)
        # and not (high criticality + redundancy) - e.g. lower criticality,
        # or high criticality with an unclear redundancy picture. Still
        # warrants prompt attention and a human in the loop.
        return {
            "recommended_action": "Inspect within 24-48 hours",
            "urgency": "High",
            "human_review_required": True,
            "rationale_code": "HIGH_RISK_STANDARD",
        }

    # --- Medium risk ---------------------------------------------------------
    if risk_band == "Medium":
        return {
            "recommended_action": "Plan inspection in next maintenance window",
            "urgency": "Medium",
            # A medium-risk reading on a high-criticality asset still gets a
            # human look before it's just scheduled and forgotten about.
            "human_review_required": is_high_criticality,
            "rationale_code": (
                "MEDIUM_RISK_CRITICAL_PLANNED_INSPECTION"
                if is_high_criticality
                else "MEDIUM_RISK_PLANNED_INSPECTION"
            ),
        }

    # --- Low risk ---------------------------------------------------------
    return {
        "recommended_action": "Continue monitoring",
        "urgency": "Routine",
        "human_review_required": False,
        "rationale_code": "LOW_RISK_CONTINUE_MONITORING",
    }


if __name__ == "__main__":
    example = decide_maintenance_action(
        failure_probability=0.82,
        risk_band="High",
        asset_criticality="High",
        is_single_point_of_failure=True,
        redundancy_available=False,
    )
    print(example)
