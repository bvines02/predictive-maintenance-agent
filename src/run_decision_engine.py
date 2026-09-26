"""Stage 14h - run the decision engine over the Step 8 validation predictions.

Reads results/health_state_predictions.csv (does NOT retrain or re-predict),
applies three SYNTHETIC asset profiles to the SAME predictions, and compares
the recommended-action distributions.
"""

import pandas as pd

from src.config import (
    DECISION_ACTION_DISTRIBUTION_PATH,
    DECISION_ENGINE_RESULTS_PATH,
    HEALTH_STATE_PREDICTIONS_PATH,
    RESULTS_DIR,
)
from src.decision_engine import (
    ACTION_MEANINGS,
    BASE_ACTIONS,
    AssetContext,
    MaintenanceAction,
    MaintenanceDecision,
    decide_for_dataframe,
    decide_maintenance,
)
from src.synthetic_asset_profiles import SYNTHETIC_PROFILES


def load_health_state_predictions(path=HEALTH_STATE_PREDICTIONS_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Generate it first with: python -m src.run_health_state")
    return pd.read_csv(path)


def describe_context(context: AssetContext) -> str:
    return (
        f"{context.criticality.name} criticality / {context.redundancy.name} redundancy / "
        f"lead time {context.maintenance_lead_time_cycles:g} / {context.confidence.name} confidence"
    )


def format_trace(decision: MaintenanceDecision) -> str:
    lines = []
    for step in decision.trace:
        if step.kind == "base":
            lines.append(f"    {step.rule_id:8s} -> {step.action_after.name:22s} {step.reason}")
        elif step.kind == "guardrail":
            lines.append(f"    {step.rule_id:8s} (no change: {step.action_after.name}) {step.reason}")
        else:
            arrow = "UP  " if step.kind == "escalate" else "DOWN"
            lines.append(
                f"    {step.rule_id:8s} {arrow} {step.action_before.name} -> {step.action_after.name}: {step.reason}"
            )
    return "\n".join(lines)


def _section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def print_rule_set_and_hierarchy() -> None:
    _section("Action severity hierarchy")
    for action in MaintenanceAction:
        print(f"  {int(action)}  {action.name:22s} {ACTION_MEANINGS[action]}")
    _section("Base actions (BASE_01)")
    for state, action in BASE_ACTIONS.items():
        print(f"  {state:8s} -> {action.name}")


def print_worked_example(df: pd.DataFrame, target_rul: float, label: str) -> None:
    candidates = df[(df["health_state"] == "PLAN") & (df["predicted_rul"] <= target_rul)]
    row = candidates.loc[candidates["predicted_rul"].idxmax()]

    _section(f"Worked decision trace - {label}")
    print("MODEL OUTPUT")
    print(
        f"  Engine {int(row['unit_number'])}, cycle {int(row['time_cycles'])}: "
        f"predicted RUL = {row['predicted_rul']:.1f} cycles (true RUL {row['actual_rul']:.0f}, not seen by the engine)"
    )
    print("HEALTH INTERPRETATION")
    print(f"  {row['health_state']}")
    for name, context in SYNTHETIC_PROFILES.items():
        decision = decide_maintenance(row["predicted_rul"], row["health_state"], context)
        print(f"\nPROFILE {name}: {describe_context(context)}")
        print(format_trace(decision))
        print(f"  RECOMMENDED: {decision.recommended_action.name}   (human review: {decision.requires_human_review})")


if __name__ == "__main__":
    predictions = load_health_state_predictions()

    print_rule_set_and_hierarchy()

    _section("SYNTHETIC context profiles (NASA data has none of this)")
    for name, context in SYNTHETIC_PROFILES.items():
        print(f"  {name}: {describe_context(context)}  [source: {context.source}]")

    results = []
    for name, context in SYNTHETIC_PROFILES.items():
        decided = decide_for_dataframe(predictions, context)
        decided.insert(0, "profile", name)
        results.append(decided)
    all_results = pd.concat(results, ignore_index=True)

    order = [a.name for a in MaintenanceAction]
    counts = (
        all_results.groupby(["recommended_action", "profile"]).size().unstack("profile", fill_value=0)
        .reindex(order, fill_value=0)
    )
    base_counts = predictions["health_state"].map(lambda s: BASE_ACTIONS[s].name).value_counts()

    _section(f"Recommended action distribution ({len(predictions)} validation rows per profile)")
    table = counts
    table.insert(0, "base (no context)", [int(base_counts.get(a, 0)) for a in table.index])
    print(table.to_string())

    _section("Change vs. base action, per profile")
    for name in SYNTHETIC_PROFILES:
        sub = all_results[all_results["profile"] == name]
        base_sev = sub["base_action"].map(lambda a: int(MaintenanceAction[a]))
        delta = sub["action_severity"] - base_sev
        print(f"  {name}: raised {int((delta > 0).sum())}, unchanged {int((delta == 0).sum())}, lowered {int((delta < 0).sum())}")

    print_worked_example(predictions, 20.0, "predicted RUL just under 20 (all three profiles differ here)")
    print_worked_example(predictions, 25.0, "predicted RUL just under 25")

    _section("Scenario G - same prediction, FULL vs NONE redundancy (PLAN state, RUL 22, lead time 20, HIGH confidence)")
    cases = [("MEDIUM", 22.0, "PLAN"), ("HIGH", 22.0, "PLAN"), ("CRITICAL", 22.0, "PLAN")]
    print(f"  {'criticality':10s} {'RUL':>5s} {'state':7s} {'FULL redundancy':24s} {'NONE redundancy':24s}")
    for criticality, rul, state in cases:
        row = []
        for redundancy in ("FULL", "NONE"):
            ctx = AssetContext(criticality, redundancy, 20, "HIGH")
            row.append(decide_maintenance(rul, state, ctx).recommended_action.name)
        print(f"  {criticality:10s} {rul:5.0f} {state:7s} {row[0]:24s} {row[1]:24s}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    keep = [
        "profile", "unit_number", "time_cycles", "actual_rul", "predicted_rul", "health_state",
        "base_action", "recommended_action", "rule_ids", "requires_human_review",
    ]
    all_results[keep].to_csv(DECISION_ENGINE_RESULTS_PATH, index=False)
    table.to_csv(DECISION_ACTION_DISTRIBUTION_PATH)
    print(f"\nSaved {DECISION_ENGINE_RESULTS_PATH}")
    print(f"Saved {DECISION_ACTION_DISTRIBUTION_PATH}")
