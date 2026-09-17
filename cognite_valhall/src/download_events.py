"""Download and characterise events for the selected hierarchy.

DESIGN NOTE — events are not automatically "work orders"
--------------------------------------------------------
CDF's ``Event`` is a deliberately generic container: an id, a type/subtype pair,
a description, a start and end time, asset links and a free-form metadata dict.
What it *represents* depends entirely on what the source system pushed into it.
In different projects the same resource type holds maintenance notifications,
SAP work orders, alarm and trip records, shift-log entries, operator comments,
process-phase markers, or synthetic events created by an analytics job.

So rather than asserting a meaning, this module *measures* it: it profiles the
distinct ``type``/``subtype`` values, the metadata keys present, the duration
distribution, and how many events carry asset links. Those measurements are what
the classifier below reasons over, and the report states its confidence.

This matters for the downstream agent. A recommendation engine that treats alarm
records as completed maintenance work will confidently tell an operator that a
bearing was replaced last month when in fact a high-temperature alarm merely
fired. Knowing what the records *are* is a prerequisite, not a nicety.
"""

from __future__ import annotations

import logging
from collections import Counter

import pandas as pd
from cognite.client import CogniteClient
from cognite.client.data_classes import Event
from cognite.client.exceptions import CogniteAPIError

from . import config
from .storage import json_or_none, ms_to_utc, write_table

logger = logging.getLogger(__name__)


def discover_events(
    client: CogniteClient, asset_ids: list[int], subtree_root_id: int | None = None
) -> list[Event]:
    """Fetch events linked to the selected assets.

    We query two ways and merge, because event-to-asset linkage in real projects
    is inconsistent:

    * ``asset_ids`` — events explicitly tagged with one of our assets.
    * ``asset_subtree_ids`` — events tagged anywhere under the target's root
      equipment, which catches events attached at a level of the tree we did not
      explicitly select (very common: a work order tagged to the train rather
      than to the specific machine).

    Merging on id makes the double-query safe.
    """
    found: dict[int, Event] = {}

    chunk_size = 100
    for offset in range(0, len(asset_ids), chunk_size):
        chunk = asset_ids[offset : offset + chunk_size]
        try:
            for event in client.events.list(asset_ids=chunk, limit=config.LIST_ALL):
                if event.id is not None:
                    found[event.id] = event
        except CogniteAPIError as exc:
            logger.warning("events.list(asset_ids=...) failed for a chunk: %s", exc)

    if subtree_root_id is not None:
        try:
            for event in client.events.list(
                asset_subtree_ids=[subtree_root_id], limit=config.LIST_ALL
            ):
                if event.id is not None:
                    found[event.id] = event
        except CogniteAPIError as exc:
            logger.warning("events.list(asset_subtree_ids=...) failed: %s", exc)

    result = sorted(found.values(), key=lambda e: (e.start_time or 0, e.id or 0))
    logger.info("Discovered %d distinct events", len(result))
    return result


def events_to_dataframe(
    events: list[Event], asset_lookup: dict[int, str] | None = None
) -> pd.DataFrame:
    """Flatten events, preserving every field including the metadata blob."""
    asset_lookup = asset_lookup or {}
    rows = []
    for event in events:
        asset_ids = list(event.asset_ids or [])
        start = ms_to_utc(event.start_time)
        end = ms_to_utc(event.end_time)
        duration_h = None
        if start is not None and end is not None:
            duration_h = (end - start).total_seconds() / 3600.0
        rows.append(
            {
                "event_id": event.id,
                "external_id": event.external_id,
                "type": event.type,
                "subtype": event.subtype,
                "description": event.description,
                "start_time": start,
                "end_time": end,
                "duration_hours": duration_h,
                "asset_ids": json_or_none(asset_ids),
                "asset_external_ids": json_or_none(
                    [asset_lookup[a] for a in asset_ids if a in asset_lookup]
                ),
                "n_assets": len(asset_ids),
                "source": event.source,
                "data_set_id": event.data_set_id,
                "metadata": json_or_none(event.metadata),
                "metadata_keys": ",".join(sorted((event.metadata or {}).keys())),
                "created_time": ms_to_utc(event.created_time),
                "last_updated_time": ms_to_utc(event.last_updated_time),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("start_time", na_position="last").reset_index(drop=True)
    return df


def profile_events(df: pd.DataFrame) -> dict[str, object]:
    """Measure what the event records actually look like."""
    if df.empty:
        return {"n_events": 0}

    type_counts = Counter(df["type"].dropna().astype(str))
    subtype_counts = Counter(df["subtype"].dropna().astype(str))

    metadata_keys: Counter[str] = Counter()
    for keys in df["metadata_keys"].dropna():
        for key in str(keys).split(","):
            if key:
                metadata_keys[key] += 1

    durations = df["duration_hours"].dropna()
    return {
        "n_events": len(df),
        "types": dict(type_counts.most_common(25)),
        "subtypes": dict(subtype_counts.most_common(25)),
        "metadata_keys": dict(metadata_keys.most_common(30)),
        "n_with_assets": int((df["n_assets"] > 0).sum()),
        "n_with_end_time": int(df["end_time"].notna().sum()),
        "median_duration_hours": float(durations.median()) if len(durations) else None,
        "min_start": df["start_time"].min(),
        "max_start": df["start_time"].max(),
        "sources": dict(Counter(df["source"].dropna().astype(str)).most_common(10)),
    }


# Vocabulary used only to *interpret* what we found. Each entry is a signal, and
# we require corroboration from more than one field before asserting a category.
_CATEGORY_SIGNALS: dict[str, tuple[str, ...]] = {
    "maintenance work": (
        "work order", "workorder", "maintenance", "notification", "wo_",
        "repair", "overhaul", "replace", "service", "pm ", "preventive",
    ),
    "failure / breakdown": (
        "failure", "fail", "breakdown", "trip", "fault", "defect", "malfunction",
    ),
    "inspection": ("inspection", "inspect", "survey", "test", "check", "audit"),
    "operational event": (
        "alarm", "alert", "start", "stop", "shutdown", "startup", "setpoint",
        "state", "mode", "phase", "batch", "operation",
    ),
}


def classify_event_nature(df: pd.DataFrame, profile: dict[str, object]) -> dict[str, object]:
    """Infer, with stated evidence and confidence, what these events represent.

    The output is intentionally hedged. We return the evidence we matched so a
    human can overrule us, and we return ``"unknown / project-specific"`` when
    nothing matches rather than forcing a guess into a familiar category.
    """
    if df.empty:
        return {
            "verdict": "no events found",
            "confidence": "n/a",
            "evidence": [],
            "caveat": "The selected assets have no linked events in this project.",
        }

    # Build a searchable corpus from the structural fields, not the free text of
    # individual descriptions -- type/subtype/metadata keys are the schema-level
    # signal, and are far more reliable than one description happening to
    # contain the word "check".
    corpus_parts: list[str] = []
    for field in ("type", "subtype"):
        corpus_parts.extend(str(v).lower() for v in df[field].dropna().unique())
    corpus_parts.extend(str(k).lower() for k in (profile.get("metadata_keys") or {}))
    structural_corpus = " | ".join(corpus_parts)

    # Descriptions are used as weaker, secondary evidence.
    description_corpus = " | ".join(
        str(v).lower() for v in df["description"].dropna().head(500).unique()
    )

    scores: dict[str, dict[str, object]] = {}
    for category, signals in _CATEGORY_SIGNALS.items():
        structural_hits = [s for s in signals if s in structural_corpus]
        description_hits = [s for s in signals if s in description_corpus]
        # Structural evidence is worth more than text evidence.
        score = 2 * len(structural_hits) + len(description_hits)
        if score:
            scores[category] = {
                "score": score,
                "structural_matches": structural_hits,
                "description_matches": description_hits[:6],
            }

    if not scores:
        return {
            "verdict": "unknown / project-specific",
            "confidence": "low",
            "evidence": [],
            "caveat": (
                "No recognisable maintenance, failure, inspection or operational "
                "vocabulary appears in the event types, subtypes or metadata keys. "
                f"Observed types: {list((profile.get('types') or {}).keys())[:10]}. "
                "Inspect data/events/events.csv directly before assigning meaning."
            ),
        }

    ranked = sorted(scores.items(), key=lambda kv: -kv[1]["score"])  # type: ignore[index]
    best_category, best = ranked[0]
    best_score = int(best["score"])  # type: ignore[arg-type]
    has_structural = bool(best["structural_matches"])  # type: ignore[index]

    if has_structural and best_score >= 4:
        confidence = "high"
    elif has_structural:
        confidence = "medium"
    else:
        confidence = "low (text evidence only -- descriptions matched, but the "
        confidence += "event type/subtype schema does not corroborate it)"

    duration = profile.get("median_duration_hours")
    caveats = []
    if not has_structural:
        caveats.append(
            "The verdict rests on description text alone; type/subtype carry no "
            "corroborating vocabulary, so treat it as a hypothesis."
        )
    if profile.get("n_with_end_time", 0) == 0:
        caveats.append(
            "No event has an end_time, so these are instants rather than "
            "work intervals -- more consistent with alarms/log entries than "
            "with executed work orders."
        )
    elif isinstance(duration, (int, float)) and duration is not None and duration < 0.5:
        caveats.append(
            f"Median duration is {duration:.2f} h, which is short for maintenance "
            "execution and more typical of alarms or state changes."
        )
    if profile.get("n_with_assets", 0) < len(df) * 0.5:
        caveats.append(
            "Fewer than half of the events carry asset links, which limits how "
            "reliably they can be attributed to specific equipment."
        )

    return {
        "verdict": best_category,
        "confidence": confidence,
        "evidence": ranked[:3],
        "caveat": " ".join(caveats) if caveats else "",
    }


def print_event_assessment(profile: dict[str, object], assessment: dict[str, object]) -> None:
    print()
    print("=" * 78)
    print("EVENT / MAINTENANCE-HISTORY ASSESSMENT")
    print("=" * 78)
    print(f"  Events found            : {profile.get('n_events', 0)}")
    if profile.get("n_events"):
        print(f"  With asset links        : {profile.get('n_with_assets')}")
        print(f"  With an end_time        : {profile.get('n_with_end_time')}")
        print(f"  Time span               : {profile.get('min_start')} -> {profile.get('max_start')}")
        print(f"  Distinct types          : {list((profile.get('types') or {}).keys())}")
        print(f"  Distinct subtypes       : {list((profile.get('subtypes') or {}).keys())[:12]}")
        print(f"  Metadata keys present   : {list((profile.get('metadata_keys') or {}).keys())[:15]}")
        median = profile.get("median_duration_hours")
        if median is not None:
            print(f"  Median duration (hours) : {median:.3f}")
    print()
    print(f"  ASSESSMENT              : {assessment['verdict']}")
    print(f"  CONFIDENCE              : {assessment['confidence']}")
    if assessment.get("caveat"):
        print(f"  CAVEAT                  : {assessment['caveat']}")
    print("=" * 78)
    print()


def save_events(
    client: CogniteClient, asset_ids: list[int], asset_lookup: dict[int, str],
    subtree_root_id: int | None = None,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    events = discover_events(client, asset_ids, subtree_root_id)
    df = events_to_dataframe(events, asset_lookup)
    write_table(
        df,
        config.EVENTS_DIR / "events.parquet",
        config.EVENTS_DIR / "events.csv",
        label="events",
    )
    profile = profile_events(df)
    assessment = classify_event_nature(df, profile)
    print_event_assessment(profile, assessment)
    return df, profile, assessment
