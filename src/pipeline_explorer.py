"""Pipeline Explorer - the decision policy behind the interactive React app (app/).

Responsible for:
- The ONE place that says what the explorer's reduced decision context is:
  criticality and redundancy are EXCLUDED, lead time and health thresholds
  are editable, confidence comes from the model's own tree disagreement
- Running the real decision engine (src/decision_engine.py) under that
  context, so the app, the /explain endpoint and the fixture export all
  agree on what "excluded" means
- Generating the Python<->TypeScript parity cases the app's tests must
  reproduce exactly (app/src/pipeline/parity_cases.json)

Not responsible for:
- Any new decision logic. The TypeScript rules in app/src/pipeline/rules.ts
  are a port of decide_maintenance() restricted to this context; this
  module is what that port is checked against.

How "excluded" is implemented: the engine requires a criticality and a
redundancy, so the explorer passes the values under which those rules can
never fire - criticality LOW (CRIT_01..03 need HIGH or CRITICAL; CONF_02's
override needs HIGH) and redundancy NONE (RED_01/RED_02 need FULL). The
remaining rules are BASE_01, LEAD_01, LEAD_02 and CONF_01. These are not
claims about any real asset - the explorer never shows or sends them.
"""

import itertools
import json

from src.config import HEALTH_THRESHOLDS, PROJECT_ROOT
from src.decision_engine import (
    ACTION_MEANINGS,
    DEFAULT_DECISION_CONFIG,
    AssetContext,
    Confidence,
    Criticality,
    DecisionConfig,
    MaintenanceDecision,
    Redundancy,
    decide_maintenance,
)

EXPLORER_CRITICALITY = Criticality.LOW
EXPLORER_REDUNDANCY = Redundancy.NONE
EXPLORER_RULE_IDS = ("BASE_01", "LEAD_01", "LEAD_02", "CONF_01")

DEFAULT_LEAD_TIME_CYCLES = 15
LEAD_TIME_BUFFER_CYCLES = DEFAULT_DECISION_CONFIG.lead_time_buffer_cycles

PARITY_CASES_PATH = PROJECT_ROOT / "app" / "src" / "pipeline" / "parity_cases.json"
PROMPT_CASES_PATH = PROJECT_ROOT / "app" / "src" / "pipeline" / "prompt_cases.json"


def explorer_decision(
    predicted_rul: float,
    model_confidence: str,
    health_thresholds: dict = HEALTH_THRESHOLDS,
    maintenance_lead_time_cycles: float = DEFAULT_LEAD_TIME_CYCLES,
) -> MaintenanceDecision:
    """decide_maintenance() with criticality and redundancy excluded (see module docstring)."""
    context = AssetContext(
        criticality=EXPLORER_CRITICALITY,
        redundancy=EXPLORER_REDUNDANCY,
        maintenance_lead_time_cycles=maintenance_lead_time_cycles,
        confidence=model_confidence,
    )
    config = DecisionConfig(health_thresholds=dict(health_thresholds))
    decision = decide_maintenance(predicted_rul, None, context, config)

    unexpected = set(decision.rule_ids) - set(EXPLORER_RULE_IDS)
    if unexpected:
        raise RuntimeError(f"Excluded rules fired in the explorer context: {sorted(unexpected)}")
    return decision


def decision_summary(decision: MaintenanceDecision) -> dict:
    """The explorer-facing view of a decision: no criticality or redundancy fields."""
    return {
        "predicted_rul": decision.predicted_rul,
        "health_state": decision.health_state,
        "base_action": decision.base_action.name,
        "recommended_action": decision.recommended_action.name,
        "recommended_action_meaning": ACTION_MEANINGS[decision.recommended_action],
        "model_confidence": decision.confidence.name,
        "maintenance_lead_time_cycles": decision.maintenance_lead_time_cycles,
        "requires_human_review": decision.requires_human_review,
        "rule_ids": list(decision.rule_ids),
        "trace": [
            {
                "rule_id": step.rule_id,
                "kind": step.kind,
                "action_before": step.action_before.name if step.action_before is not None else None,
                "action_after": step.action_after.name,
                "reason": step.reason,
            }
            for step in decision.trace
        ],
    }


# Chosen to hit every boundary: exactly on each threshold, just either side
# of it, negative (clamped) RUL, and lead time at 0 / inside / above each band.
PARITY_RULS = (-3, 0, 4.5, 10, 14.99, 15, 15.01, 20, 22.5, 25, 29.99, 30, 30.01, 45, 60, 60.01, 90, 125)
PARITY_THRESHOLDS = (
    {"action": 15, "plan": 30, "watch": 60},
    {"action": 5, "plan": 20, "watch": 40},
    {"action": 25, "plan": 50, "watch": 100},
)
PARITY_LEAD_TIMES = (0, 5, 15, 25.5, 40)


def build_parity_cases() -> list[dict]:
    """Every combination of the grids above, decided by the real Python engine."""
    cases = []
    for rul, thresholds, lead, confidence in itertools.product(
        PARITY_RULS, PARITY_THRESHOLDS, PARITY_LEAD_TIMES, [c.name for c in Confidence]
    ):
        decision = explorer_decision(rul, confidence, thresholds, lead)
        cases.append(
            {
                "input": {
                    "predicted_rul": rul,
                    "model_confidence": confidence,
                    "health_thresholds": thresholds,
                    "maintenance_lead_time_cycles": lead,
                },
                "expected": {
                    "health_state": decision.health_state,
                    "base_action": decision.base_action.name,
                    "recommended_action": decision.recommended_action.name,
                    "rule_ids": list(decision.rule_ids),
                    "requires_human_review": decision.requires_human_review,
                },
            }
        )
    return cases


def parity_document() -> dict:
    return {
        "_comment": (
            "Generated by `python -m src.pipeline_explorer` from the real Python decision engine. "
            "Do not edit by hand: app/src/pipeline/rules.test.ts must reproduce every case, and "
            "tests/test_pipeline_explorer.py fails if this file is stale."
        ),
        "lead_time_buffer_cycles": LEAD_TIME_BUFFER_CYCLES,
        "cases": build_parity_cases(),
    }


# Prompt parity: the Vercel deployment builds the LLM prompt in TypeScript
# (app/src/pipeline/prompt.ts). These cases pin it to the Python prompt,
# byte for byte. RULs include exact .x5 ties (35.25, 12.75) because Python
# rounds those half-to-even and JavaScript's toFixed does not.
PROMPT_CASE_INPUTS = (
    (7, 180, 12.0, 8.0, 19.5, "HIGH", {"action": 15, "plan": 30, "watch": 60}, 15),
    (1, 163, 35.55, 19.4, 53.2, "LOW", {"action": 15, "plan": 30, "watch": 60}, 40),
    (31, 120, 35.25, 30.05, 44.95, "MEDIUM", {"action": 15, "plan": 40, "watch": 100}, 15),
    (54, 200, 12.75, 3.25, 22.75, "LOW", {"action": 20, "plan": 45, "watch": 90}, 5),
    (91, 40, 118.4, 101.0, 124.9, "HIGH", {"action": 15, "plan": 30, "watch": 60}, 0),
    (13, 150, -2.5, 0.0, 6.35, "MEDIUM", {"action": 15, "plan": 30, "watch": 60}, 25.5),
)


def build_prompt_cases() -> list[dict]:
    from src.explainer import build_rul_explanation_prompt

    cases = []
    for unit, cycle, rul, p10, p90, confidence, thresholds, lead in PROMPT_CASE_INPUTS:
        summary = decision_summary(explorer_decision(rul, confidence, thresholds, lead))
        prediction = {"unit_number": unit, "time_cycles": cycle, "prediction_p10": p10, "prediction_p90": p90}
        cases.append(
            {
                "input": {**prediction, "predicted_rul": rul, "model_confidence": confidence,
                          "health_thresholds": thresholds, "maintenance_lead_time_cycles": lead},
                "prompt": build_rul_explanation_prompt(summary, prediction, thresholds, LEAD_TIME_BUFFER_CYCLES),
            }
        )
    return cases


def prompt_document() -> dict:
    return {
        "_comment": (
            "Generated by `python -m src.pipeline_explorer`. app/src/pipeline/prompt.test.ts must rebuild "
            "every prompt exactly; tests/test_pipeline_explorer.py fails if this file is stale."
        ),
        "lead_time_buffer_cycles": LEAD_TIME_BUFFER_CYCLES,
        "action_meanings": {action.name: meaning for action, meaning in ACTION_MEANINGS.items()},
        "cases": build_prompt_cases(),
    }


def write_prompt_cases(path=PROMPT_CASES_PATH) -> None:
    path.write_text(json.dumps(prompt_document(), indent=1) + "\n")


def write_parity_cases(path=PARITY_CASES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = parity_document()
    # One case per line: compact, but a regenerated file still diffs readably.
    cases = ",\n".join("  " + json.dumps(case) for case in document.pop("cases"))
    header = json.dumps(document)[:-1]
    path.write_text(f'{header}, "cases": [\n{cases}\n]}}\n')


if __name__ == "__main__":
    write_parity_cases()
    write_prompt_cases()
    print(f"Wrote {len(PROMPT_CASE_INPUTS)} prompt cases to {PROMPT_CASES_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {len(build_parity_cases())} parity cases to {PARITY_CASES_PATH.relative_to(PROJECT_ROOT)}")
