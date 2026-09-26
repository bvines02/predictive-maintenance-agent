"""Stage 14h - deterministic maintenance decision engine (V2).

    PREDICTION      predicted RUL                        (ML model)
    INTERPRETATION  health state                         (src/health_state.py)
    DECISION        recommended maintenance action       (THIS MODULE)

Responsible for:
- Combining predicted RUL, health state and asset context into ONE action
  from a closed set (MaintenanceAction), with a full rule-by-rule trace
- Making every context effect explicit, ID'd, configurable and testable

Not responsible for:
- Producing free text, calling an LLM, creating work orders, touching a CMMS
- Deriving criticality / redundancy / lead time / confidence (they are inputs)
- Optimising any threshold (all numbers here are placeholders)

IMPORTANT - the asset context is SYNTHETIC. NASA C-MAPSS contains no asset
criticality, redundancy, spares lead time, consequence or safety data.
The context values used in this project are invented to teach the
architecture. In a real system they come from other systems of record:
    criticality       asset register / criticality assessment / FMEA
    redundancy        system design, P&ID, single-line diagram, knowledge graph
    lead time         CMMS / ERP / materials system
    confidence        the ML inference pipeline (ensemble spread, prediction
                      intervals, residual distributions, OOD checks, sensor
                      quality). Here it is a manually supplied label.

Pipeline (each step may append to the trace; rule IDs are stable):

    BASE_01   health state -> base action
    LEAD_01   predicted RUL <= lead time            -> INTERVENE_NOW
    LEAD_02   lead time < RUL <= lead time + buffer -> at least SCHEDULE_MAINTENANCE
    CRIT_01   HIGH/CRITICAL + WATCH                 -> at least INSPECT
    CRIT_02   CRITICAL + PLAN                       -> at least SCHEDULE_MAINTENANCE
    CRIT_03   CRITICAL + ACTION                     -> INTERVENE_NOW
    RED_01    FULL redundancy (guarded)             -> one level LESS urgent
    RED_02    (note) redundancy credit withheld by a guardrail
    CONF_01   LOW confidence (guarded)              -> INSPECT (seek confirmation)
    CONF_02   (note) LOW confidence overridden: escalation kept

Why lead time (engineering view): work takes `lead time` cycles from decision
to completion. A decision made when predicted RUL == lead time finishes
exactly at predicted failure, with zero margin - and the RUL estimate itself
is uncertain (Step 7 showed critical-region over-prediction of up to ~14
cycles). So the decision must be taken BEFORE RUL crosses the lead time; the
buffer (LEAD_02) is that margin. Inside the lead time (LEAD_01) an orderly
plan can no longer finish before predicted failure.

Why redundancy is a consequence rule, not a condition rule: a standby does
not make the engine any healthier - the model still says it is degraded.
Redundancy changes what its failure does to the SYSTEM. So it can only
relieve context-driven escalation, and only where consequence is modest.

Governance principle: context rules ESCALATE by default. The only lowering
rules are RED_01 and CONF_01, each explicit, guarded and tested. A PLAN or
ACTION state can never produce less than INSPECT (the warning is never
ignored).
"""

import math
import numbers
from dataclasses import dataclass, field
from enum import Enum, IntEnum

import pandas as pd

from src.config import HEALTH_THRESHOLDS
from src.health_state import HEALTH_STATES, assign_health_state

CONTEXT_SOURCE_SYNTHETIC = "SYNTHETIC_LEARNING_EXERCISE"


class Criticality(IntEnum):
    """Consequence of failure. Ordered, so `criticality >= HIGH` is meaningful."""

    LOW = 0  # minor operational inconvenience
    MEDIUM = 1  # meaningful production / operating impact
    HIGH = 2  # major production, cost or reliability impact
    CRITICAL = 3  # severe safety, environmental, regulatory or system consequence


class Redundancy(Enum):
    FULL = "FULL"  # standby / alternative path maintains function
    PARTIAL = "PARTIAL"  # some function maintained, reduced resilience/capacity
    NONE = "NONE"  # failure directly removes required function


class Confidence(Enum):
    """Confidence in the RUL prediction. Supplied as input in this version."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class MaintenanceAction(IntEnum):
    """Closed set of allowed actions, ordered by urgency (the integer IS the severity)."""

    NO_ACTION = 0
    MONITOR = 1
    INSPECT = 2
    PLAN_MAINTENANCE = 3
    SCHEDULE_MAINTENANCE = 4
    INTERVENE_NOW = 5


ACTION_MEANINGS = {
    MaintenanceAction.NO_ACTION: "Continue normal operation.",
    MaintenanceAction.MONITOR: "Increase observation or review the trend at normal planning cadence.",
    MaintenanceAction.INSPECT: "Perform targeted inspection or diagnostic verification.",
    MaintenanceAction.PLAN_MAINTENANCE: "Start defining scope, spares and execution requirements.",
    MaintenanceAction.SCHEDULE_MAINTENANCE: "Commit the intervention into an executable maintenance window.",
    MaintenanceAction.INTERVENE_NOW: "Immediate operational / maintenance response is required.",
}

# BASE_01: the health-state baseline, before any context is applied.
BASE_ACTIONS = {
    "HEALTHY": MaintenanceAction.NO_ACTION,
    "WATCH": MaintenanceAction.MONITOR,
    "PLAN": MaintenanceAction.PLAN_MAINTENANCE,
    "ACTION": MaintenanceAction.SCHEDULE_MAINTENANCE,
}


def _coerce(enum_cls, value, field_name: str):
    """Accept an enum member or its (case-insensitive) name; reject everything else."""
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls[value.strip().upper()]
        except KeyError:
            pass
    allowed = [member.name for member in enum_cls]
    raise ValueError(f"{field_name} must be one of {allowed}, got {value!r}.")


def _check_real(value, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or math.isnan(value):
        raise ValueError(f"{field_name} must be a real number, got {value!r}.")
    return float(value)


@dataclass(frozen=True)
class AssetContext:
    """Operational context for one asset. SYNTHETIC in this project (see module docstring)."""

    criticality: Criticality
    redundancy: Redundancy
    maintenance_lead_time_cycles: float
    confidence: Confidence
    source: str = CONTEXT_SOURCE_SYNTHETIC

    def __post_init__(self):
        object.__setattr__(self, "criticality", _coerce(Criticality, self.criticality, "criticality"))
        object.__setattr__(self, "redundancy", _coerce(Redundancy, self.redundancy, "redundancy"))
        object.__setattr__(self, "confidence", _coerce(Confidence, self.confidence, "confidence"))
        lead = _check_real(self.maintenance_lead_time_cycles, "maintenance_lead_time_cycles")
        if lead < 0:
            raise ValueError("maintenance_lead_time_cycles cannot be negative.")
        object.__setattr__(self, "maintenance_lead_time_cycles", lead)


@dataclass(frozen=True)
class DecisionConfig:
    """Tunable policy numbers. Placeholders - not optimised, not physical truths."""

    lead_time_buffer_cycles: float = 5
    # RED_01 credit is only given to assets at or below this criticality.
    redundancy_credit_max_criticality: Criticality = Criticality.MEDIUM
    # CONF_02: at/above this criticality, and inside the lead time, low
    # confidence does NOT downgrade the response.
    low_confidence_override_min_criticality: Criticality = Criticality.HIGH
    health_thresholds: dict = field(default_factory=lambda: dict(HEALTH_THRESHOLDS))


DEFAULT_DECISION_CONFIG = DecisionConfig()


@dataclass(frozen=True)
class RuleStep:
    """One entry in the audit trace."""

    rule_id: str
    kind: str  # "base" | "escalate" | "de_escalate" | "guardrail"
    action_before: MaintenanceAction | None
    action_after: MaintenanceAction
    reason: str


@dataclass(frozen=True)
class MaintenanceDecision:
    recommended_action: MaintenanceAction
    base_action: MaintenanceAction
    predicted_rul: float
    health_state: str
    criticality: Criticality
    redundancy: Redundancy
    maintenance_lead_time_cycles: float
    confidence: Confidence
    escalation_reasons: tuple[str, ...]
    de_escalation_reasons: tuple[str, ...]
    guardrail_notes: tuple[str, ...]
    rule_ids: tuple[str, ...]
    requires_human_review: bool
    context_source: str
    trace: tuple[RuleStep, ...]

    def to_dict(self) -> dict:
        return {
            "recommended_action": self.recommended_action.name,
            "base_action": self.base_action.name,
            "predicted_rul": self.predicted_rul,
            "health_state": self.health_state,
            "criticality": self.criticality.name,
            "redundancy": self.redundancy.name,
            "maintenance_lead_time_cycles": self.maintenance_lead_time_cycles,
            "confidence": self.confidence.name,
            "escalation_reasons": list(self.escalation_reasons),
            "de_escalation_reasons": list(self.de_escalation_reasons),
            "guardrail_notes": list(self.guardrail_notes),
            "rule_ids": list(self.rule_ids),
            "requires_human_review": self.requires_human_review,
            "context_source": self.context_source,
        }


class _Trace:
    """Holds the current action and records every change with its rule ID."""

    def __init__(self, base_action: MaintenanceAction, health_state: str):
        self.action = base_action
        self.steps = [
            RuleStep("BASE_01", "base", None, base_action, f"Health state {health_state} maps to {base_action.name}.")
        ]

    def escalate_to(self, floor: MaintenanceAction, rule_id: str, reason: str) -> None:
        if floor > self.action:
            self.steps.append(RuleStep(rule_id, "escalate", self.action, floor, reason))
            self.action = floor

    def lower_to(self, new_action: MaintenanceAction, rule_id: str, reason: str) -> None:
        if new_action < self.action:
            self.steps.append(RuleStep(rule_id, "de_escalate", self.action, new_action, reason))
            self.action = new_action

    def note(self, rule_id: str, reason: str) -> None:
        self.steps.append(RuleStep(rule_id, "guardrail", self.action, self.action, reason))


def get_base_action(health_state: str) -> MaintenanceAction:
    """BASE_01: the health-state baseline, before context modifies it."""
    if health_state not in BASE_ACTIONS:
        raise ValueError(f"Unknown health_state {health_state!r}. Expected one of {HEALTH_STATES}.")
    return BASE_ACTIONS[health_state]


def apply_lead_time_rules(trace: _Trace, predicted_rul: float, context: AssetContext, config: DecisionConfig) -> None:
    """LEAD_01 / LEAD_02: the decision must precede the time needed to execute the work."""
    rul = max(predicted_rul, 0.0)
    lead = context.maintenance_lead_time_cycles
    buffer = config.lead_time_buffer_cycles

    if rul <= lead:
        trace.escalate_to(
            MaintenanceAction.INTERVENE_NOW,
            "LEAD_01",
            f"Predicted RUL {rul:g} <= maintenance lead time {lead:g}: work started now may not finish before predicted failure.",
        )
    elif rul <= lead + buffer:
        trace.escalate_to(
            MaintenanceAction.SCHEDULE_MAINTENANCE,
            "LEAD_02",
            f"Predicted RUL {rul:g} is within {buffer:g} cycles of maintenance lead time {lead:g}: commit the work now to keep a margin.",
        )


def apply_criticality_rules(trace: _Trace, health_state: str, context: AssetContext) -> None:
    """CRIT_01..03: higher consequence of failure raises the floor for concerning states."""
    criticality = context.criticality

    if criticality >= Criticality.HIGH and health_state == "WATCH":
        trace.escalate_to(
            MaintenanceAction.INSPECT,
            "CRIT_01",
            f"Asset criticality is {criticality.name}: an early WATCH warrants verification, not just monitoring.",
        )
    if criticality == Criticality.CRITICAL and health_state == "PLAN":
        trace.escalate_to(
            MaintenanceAction.SCHEDULE_MAINTENANCE,
            "CRIT_02",
            "Asset criticality is CRITICAL: PLAN state requires a committed maintenance window.",
        )
    if criticality == Criticality.CRITICAL and health_state == "ACTION":
        trace.escalate_to(
            MaintenanceAction.INTERVENE_NOW,
            "CRIT_03",
            "Asset criticality is CRITICAL: ACTION state requires an immediate response.",
        )


def apply_redundancy_rules(
    trace: _Trace, base_action: MaintenanceAction, health_state: str, context: AssetContext, config: DecisionConfig
) -> None:
    """RED_01 / RED_02: full redundancy can relieve context-driven escalation, with guardrails.

    Guardrails - redundancy credit is NOT given when:
    - redundancy is NONE or PARTIAL (no reduction; PARTIAL keeps resilience reduced)
    - the health state is ACTION (the equipment is close to failure regardless)
    - criticality is above `redundancy_credit_max_criticality`
    And when given, it is at most ONE level, and never below the base action.
    """
    if context.redundancy != Redundancy.FULL or trace.action <= base_action:
        return

    if health_state == "ACTION" or context.criticality > config.redundancy_credit_max_criticality:
        trace.note(
            "RED_02",
            "Full redundancy exists but credit is withheld "
            f"(health state {health_state}, criticality {context.criticality.name}).",
        )
        return

    relieved = max(base_action, MaintenanceAction(trace.action - 1))
    trace.lower_to(
        relieved,
        "RED_01",
        "Full redundancy: system consequence is reduced, so context-driven urgency is lowered one level (never below the base action).",
    )


def apply_confidence_rules(
    trace: _Trace, health_state: str, predicted_rul: float, context: AssetContext, config: DecisionConfig
) -> None:
    """CONF_01 / CONF_02: low confidence means 'seek confirming evidence', not 'ignore'."""
    if context.confidence != Confidence.LOW or health_state not in ("PLAN", "ACTION"):
        return
    if trace.action <= MaintenanceAction.INSPECT:
        return

    inside_lead_time = max(predicted_rul, 0.0) <= context.maintenance_lead_time_cycles
    if context.criticality >= config.low_confidence_override_min_criticality and inside_lead_time:
        trace.note(
            "CONF_02",
            f"Confidence is LOW but criticality is {context.criticality.name} and predicted RUL is inside the "
            "maintenance lead time: escalation kept; verify condition in parallel.",
        )
        return

    trace.lower_to(
        MaintenanceAction.INSPECT,
        "CONF_01",
        "Confidence is LOW: seek confirming evidence (inspection) before an intrusive action.",
    )


def _check_invariants(decision: MaintenanceDecision) -> None:
    """Guardrails that must hold for every input. A failure here is a bug in the rules."""
    if decision.health_state in ("PLAN", "ACTION") and decision.recommended_action < MaintenanceAction.INSPECT:
        raise RuntimeError("Invariant violated: a PLAN/ACTION state produced less than INSPECT.")
    if decision.recommended_action < decision.base_action and "CONF_01" not in decision.rule_ids:
        raise RuntimeError("Invariant violated: action fell below base without the explicit CONF_01 rule.")


def decide_maintenance(
    predicted_rul: float,
    health_state: str | None,
    context: AssetContext,
    config: DecisionConfig = DEFAULT_DECISION_CONFIG,
) -> MaintenanceDecision:
    """Turn a prediction + interpretation + asset context into one traceable decision.

    `health_state` may be None (derived from predicted_rul) or supplied. If
    supplied it must agree with predicted_rul under config.health_thresholds -
    otherwise a caller could pair a calm state with a low RUL and steer the
    outcome, so the mismatch raises ValueError.
    """
    rul = _check_real(predicted_rul, "predicted_rul")
    derived_state = assign_health_state(rul, config.health_thresholds)

    if health_state is None:
        health_state = derived_state
    else:
        health_state = str(health_state).strip().upper()
        if health_state not in HEALTH_STATES:
            raise ValueError(f"Unknown health_state {health_state!r}. Expected one of {HEALTH_STATES}.")
        if health_state != derived_state:
            raise ValueError(
                f"health_state {health_state} is inconsistent with predicted_rul {rul:g} "
                f"(thresholds give {derived_state})."
            )

    base_action = get_base_action(health_state)
    trace = _Trace(base_action, health_state)

    apply_lead_time_rules(trace, rul, context, config)
    apply_criticality_rules(trace, health_state, context)
    apply_redundancy_rules(trace, base_action, health_state, context, config)
    apply_confidence_rules(trace, health_state, rul, context, config)

    steps = tuple(trace.steps)
    decision = MaintenanceDecision(
        recommended_action=trace.action,
        base_action=base_action,
        predicted_rul=rul,
        health_state=health_state,
        criticality=context.criticality,
        redundancy=context.redundancy,
        maintenance_lead_time_cycles=context.maintenance_lead_time_cycles,
        confidence=context.confidence,
        escalation_reasons=tuple(s.reason for s in steps if s.kind == "escalate"),
        de_escalation_reasons=tuple(s.reason for s in steps if s.kind == "de_escalate"),
        guardrail_notes=tuple(s.reason for s in steps if s.kind == "guardrail"),
        rule_ids=tuple(s.rule_id for s in steps),
        # Human-in-the-loop (CLAUDE.md): committing or forcing intervention needs a person.
        requires_human_review=trace.action >= MaintenanceAction.SCHEDULE_MAINTENANCE,
        context_source=context.source,
        trace=steps,
    )
    _check_invariants(decision)
    return decision


def decide_for_dataframe(
    df: pd.DataFrame, context: AssetContext, config: DecisionConfig = DEFAULT_DECISION_CONFIG
) -> pd.DataFrame:
    """Apply one asset context to every row (needs predicted_rul and health_state columns).

    The ML predictions in `df` are untouched - only decision columns are added,
    which is the point: same predictions, different context, different decision.
    """
    result = df.copy()
    decisions = [
        decide_maintenance(rul, state, context, config)
        for rul, state in zip(result["predicted_rul"], result["health_state"])
    ]
    result["base_action"] = [d.base_action.name for d in decisions]
    result["recommended_action"] = [d.recommended_action.name for d in decisions]
    result["action_severity"] = [int(d.recommended_action) for d in decisions]
    result["rule_ids"] = [";".join(d.rule_ids) for d in decisions]
    result["requires_human_review"] = [d.requires_human_review for d in decisions]
    return result


def decide_for_dataframe_with_confidence(
    df: pd.DataFrame,
    base_context: AssetContext,
    confidence_column: str = "model_confidence",
    config: DecisionConfig = DEFAULT_DECISION_CONFIG,
) -> pd.DataFrame:
    """Like decide_for_dataframe, but confidence comes from `confidence_column`, per row.

    Step 10 (src/uncertainty.py) derives `model_confidence` from Random
    Forest tree agreement instead of it being supplied manually. This
    function is the ONLY change that integration makes: criticality,
    redundancy and lead time still come from one shared `base_context`
    (unchanged), and CONF_01/CONF_02 - the rules that actually use
    confidence - are untouched. A new AssetContext is built per row purely
    to carry that row's own confidence value through the existing pipeline.
    """
    result = df.copy()
    decisions = [
        decide_maintenance(
            rul,
            state,
            AssetContext(
                base_context.criticality,
                base_context.redundancy,
                base_context.maintenance_lead_time_cycles,
                confidence,
                source=base_context.source,
            ),
            config,
        )
        for rul, state, confidence in zip(
            result["predicted_rul"], result["health_state"], result[confidence_column]
        )
    ]
    result["base_action"] = [d.base_action.name for d in decisions]
    result["recommended_action"] = [d.recommended_action.name for d in decisions]
    result["action_severity"] = [int(d.recommended_action) for d in decisions]
    result["rule_ids"] = [";".join(d.rule_ids) for d in decisions]
    result["requires_human_review"] = [d.requires_human_review for d in decisions]
    return result
