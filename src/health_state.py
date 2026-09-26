"""Stage 14g/h - convert predicted RUL into an operational health state.

Responsible for:
- Assigning exactly one health state (HEALTHY / WATCH / PLAN / ACTION) to an
  RUL value, using thresholds from src/config.py
- Applying the SAME function to actual RUL, so predicted and actual states
  are directly comparable
- State-level evaluation: confusion matrix, ACTION recall/precision, and an
  ordinal severity gap (state_error)

Not responsible for (later steps):
- Maintenance recommendations or decisions
- Asset criticality, redundancy, lead time, cost or consequence
- Any ML or LLM component

Three different things, kept separate on purpose:
    PREDICTION     "predicted RUL is 22 cycles"           (the ML model)
    INTERPRETATION "the asset is in the PLAN state"       (this module)
    DECISION       "schedule maintenance next window"     (not built yet)
This module stops at interpretation.

Why deterministic rather than ML: the mapping from a number to a state is a
policy choice about planning horizons, not something to be learned from
data. A fixed rule gives the same answer for the same input every time, can
be read and audited by a reliability engineer, and can be changed by editing
one dict when the organisation's planning horizon changes.

Sign convention for state_error matches the RUL error convention used
elsewhere (positive = more optimistic):
    state_error = predicted_severity - actual_severity
    negative -> model is LESS concerned than reality (dangerous direction)
    positive -> model is MORE conservative than reality
"""

import numpy as np
import pandas as pd

from src.config import HEALTH_THRESHOLDS

HEALTH_STATES = ["HEALTHY", "WATCH", "PLAN", "ACTION"]
STATE_SEVERITY = {state: level for level, state in enumerate(HEALTH_STATES)}


def validate_thresholds(thresholds: dict) -> None:
    """Thresholds must be present and strictly ordered: action < plan < watch."""
    missing = {"watch", "plan", "action"} - set(thresholds)
    if missing:
        raise ValueError(f"Health thresholds missing keys: {sorted(missing)}")
    if not (thresholds["action"] < thresholds["plan"] < thresholds["watch"]):
        raise ValueError(
            "Health thresholds must satisfy action < plan < watch, got "
            f"{thresholds['action']}, {thresholds['plan']}, {thresholds['watch']}."
        )


def assign_health_states(rul: pd.Series, thresholds: dict = HEALTH_THRESHOLDS) -> pd.Series:
    """Map each RUL value to exactly one health state.

    ACTION: rul <= action     PLAN: action < rul <= plan
    WATCH:  plan < rul <= watch     HEALTHY: rul > watch

    Edge cases, decided deliberately:
    - Negative RUL is clamped to 0 first. A regression model can output a
      negative number, but physical remaining life cannot be negative, and
      "already past zero" belongs in the most urgent state anyway.
    - NaN raises ValueError. A missing prediction must never silently map to
      a state (least of all HEALTHY, the calm one) - in a safety-relevant
      layer, failing loudly is safer than guessing.
    """
    validate_thresholds(thresholds)

    values = pd.Series(rul).astype(float)
    if values.isna().any():
        raise ValueError("Cannot assign a health state to a NaN RUL.")

    clamped = values.clip(lower=0)
    conditions = [
        clamped <= thresholds["action"],
        clamped <= thresholds["plan"],
        clamped <= thresholds["watch"],
    ]
    states = np.select(conditions, ["ACTION", "PLAN", "WATCH"], default="HEALTHY")
    return pd.Series(states, index=values.index, name="health_state")


def assign_health_state(rul: float, thresholds: dict = HEALTH_THRESHOLDS) -> str:
    """Single-value version of assign_health_states (same rules, same code path)."""
    return assign_health_states(pd.Series([rul]), thresholds).iloc[0]


def add_health_states(predictions_df: pd.DataFrame, thresholds: dict = HEALTH_THRESHOLDS) -> pd.DataFrame:
    """Add predicted/actual health state, severities and state_error to a predictions frame.

    Expects `actual_rul` and `predicted_rul`. Returns a new DataFrame.
    Columns added: health_state (predicted), actual_health_state,
    predicted_severity, actual_severity, state_error.
    """
    result = predictions_df.copy()
    result["health_state"] = assign_health_states(result["predicted_rul"], thresholds)
    result["actual_health_state"] = assign_health_states(result["actual_rul"], thresholds)
    result["predicted_severity"] = result["health_state"].map(STATE_SEVERITY)
    result["actual_severity"] = result["actual_health_state"].map(STATE_SEVERITY)
    result["state_error"] = result["predicted_severity"] - result["actual_severity"]
    return result


def state_distribution(states: pd.Series) -> pd.Series:
    """Count of rows per state, in severity order, including states with zero rows."""
    return states.value_counts().reindex(HEALTH_STATES, fill_value=0)


def state_confusion_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Rows = actual state, columns = predicted state, always all four states."""
    matrix = pd.crosstab(df["actual_health_state"], df["health_state"])
    matrix = matrix.reindex(index=HEALTH_STATES, columns=HEALTH_STATES, fill_value=0)
    matrix.index.name = "actual_state"
    matrix.columns.name = "predicted_state"
    return matrix


def _ratio(numerator: int, denominator: int) -> float:
    """Undefined ratios (0/0) are NaN, not 0 - "no observations" is not "0% correct"."""
    return numerator / denominator if denominator else float("nan")


def state_metrics(df: pd.DataFrame) -> dict:
    """State-level metrics that look at the dangerous cells, not just accuracy.

    - action_recall: of rows that truly belong in ACTION, share the model put in ACTION
    - action_precision: of rows the model put in ACTION, share that truly are ACTION
    - plan_or_action_recall: of rows truly in PLAN or ACTION, share predicted PLAN or ACTION
    - pct_actual_action_missed_as_healthy_or_watch: dangerous misses
    - pct_actual_healthy_flagged_action: the most extreme false alarms
    """
    actual, predicted = df["actual_health_state"], df["health_state"]

    actual_action = actual == "ACTION"
    predicted_action = predicted == "ACTION"
    actual_plan_or_action = actual.isin(["PLAN", "ACTION"])
    predicted_plan_or_action = predicted.isin(["PLAN", "ACTION"])
    actual_healthy = actual == "HEALTHY"

    n_actual_action = int(actual_action.sum())
    n_actual_healthy = int(actual_healthy.sum())

    return {
        "state_accuracy": float((actual == predicted).mean()),
        "action_recall": _ratio(int((actual_action & predicted_action).sum()), n_actual_action),
        "action_precision": _ratio(
            int((actual_action & predicted_action).sum()), int(predicted_action.sum())
        ),
        "plan_or_action_recall": _ratio(
            int((actual_plan_or_action & predicted_plan_or_action).sum()),
            int(actual_plan_or_action.sum()),
        ),
        "pct_actual_action_missed_as_healthy_or_watch": 100
        * _ratio(
            int((actual_action & predicted.isin(["HEALTHY", "WATCH"])).sum()), n_actual_action
        ),
        "pct_actual_healthy_flagged_action": 100
        * _ratio(int((actual_healthy & predicted_action).sum()), n_actual_healthy),
    }


DISPLAY_COLUMNS = [
    "unit_number",
    "time_cycles",
    "actual_rul",
    "predicted_rul",
    "actual_health_state",
    "health_state",
    "state_error",
]


def serious_under_classifications(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Rows where the model is less concerned than reality, largest severity gap first.

    Ties in state_error are broken by the RUL error (largest over-prediction first).
    """
    under = df[df["state_error"] < 0].copy()
    under = under.sort_values(["state_error", "error"], ascending=[True, False])
    return under[DISPLAY_COLUMNS].head(n).reset_index(drop=True)


def conservative_over_classifications(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Rows where the model is more concerned than reality, largest severity gap first."""
    over = df[df["state_error"] > 0].copy()
    over = over.sort_values(["state_error", "error"], ascending=[False, True])
    return over[DISPLAY_COLUMNS].head(n).reset_index(drop=True)
