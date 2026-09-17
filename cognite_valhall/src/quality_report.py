"""Generate a data-quality report from what actually landed on disk.

DESIGN NOTE — why the report reads the files, not the API responses
-------------------------------------------------------------------
It would be cheaper to accumulate statistics during the download. We
deliberately re-read the Parquet files instead, because the question this
report answers is *"is the dataset on disk fit to model from?"* — not *"what
did the API say?"*. Those differ whenever a write is truncated, a schema is
inconsistent, or a chunk silently failed. Re-reading is an independent check of
the artefact you are about to hand to a model.

DESIGN NOTE — which quality checks matter for condition monitoring
------------------------------------------------------------------
The four checks below are not generic data hygiene; each one corresponds to a
failure mode that silently destroys a predictive-maintenance model:

* **Missing-data rate / gaps.** A resampled feature over a gap is fabricated
  data. An anomaly detector trained across a two-day historian outage learns
  the outage, then flags every normal restart.
* **Duplicate timestamps.** Usually a double-ingest. They bias any windowed
  aggregate towards the duplicated instant and break ``set_index`` /
  ``reindex`` operations that assume a unique index.
* **Frozen / constant sensors.** A transmitter that has failed high, failed
  low, or is reading its last good value is the single most common way a
  "model" ends up predicting from a dead input. Variance-based feature
  selection will happily keep a constant channel, and it will look stable and
  trustworthy right up until you rely on it.
* **Units.** Mixing bar and barg, or degC and K, across tags that a model
  treats as interchangeable produces physically nonsensical thresholds. Missing
  units are a blocker for any physics-informed reasoning downstream.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from . import config

logger = logging.getLogger(__name__)

# A channel whose range is below this fraction of its own magnitude over the
# whole window is treated as suspiciously flat rather than genuinely steady.
FROZEN_RELATIVE_RANGE = 1e-9


def _unit(value: Any) -> str:
    """Render a unit cell, treating NaN/None/empty as "no unit declared".

    ``float("nan")`` is truthy, so the obvious ``value or "-"`` idiom prints the
    literal string "nan" into the report. A small thing, but a generated
    document that says "nan" where it means "unknown" loses the reader's trust.
    """
    if value is None:
        return "\u2014"
    if isinstance(value, float) and math.isnan(value):
        return "\u2014"
    if str(value).strip() in {"", "nan", "None", "<NA>"}:
        return "\u2014"
    return str(value)


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def analyse_datapoint_files(directory: Path) -> pd.DataFrame:
    """Compute per-tag quality statistics from the Parquet datapoint files."""
    rows: list[dict[str, Any]] = []
    files = sorted(directory.glob("*.parquet"))
    logger.info("Analysing %d datapoint file(s) for quality ...", len(files))

    for path in files:
        try:
            df = pd.read_parquet(path, columns=["timestamp", "value", "external_id", "unit"])
        except Exception as exc:  # noqa: BLE001 - a corrupt file must not abort the report
            logger.error("Could not read %s: %s", path.name, exc)
            rows.append({"file": path.name, "external_id": None, "readable": False,
                         "error": str(exc), "n_points": 0})
            continue

        if df.empty:
            rows.append({"file": path.name, "external_id": None, "readable": True,
                         "n_points": 0, "error": None})
            continue

        xid = str(df["external_id"].iloc[0])
        unit = df["unit"].iloc[0] if "unit" in df else None
        ts = pd.to_datetime(df["timestamp"], utc=True).sort_values()

        n_points = len(df)
        n_dup = int(ts.duplicated().sum())
        first, last = ts.iloc[0], ts.iloc[-1]
        span_s = (last - first).total_seconds()

        # Interval statistics. The *median* interval is the right estimator of
        # the nominal sample rate: the mean is dragged upward by a single long
        # outage, which would understate the missing-data rate precisely when it
        # matters most.
        deltas = ts.diff().dropna().dt.total_seconds()
        median_interval = float(deltas.median()) if len(deltas) else None
        max_gap = float(deltas.max()) if len(deltas) else None

        expected = None
        missing_rate = None
        if median_interval and median_interval > 0 and span_s > 0:
            expected = int(span_s / median_interval) + 1
            missing_rate = max(0.0, 1.0 - (n_points / expected)) if expected else None

        # Frozen-sensor detection on numeric channels only.
        # Two distinct things, previously conflated: a genuinely null value, and
        # a value that is simply not numeric (a string-type series). Keep them
        # apart -- a null is a data-quality defect, a string value is a type fact.
        n_null = int(df["value"].isna().sum())
        numeric = pd.to_numeric(df["value"], errors="coerce").dropna()
        n_non_numeric = int(len(df) - n_null - len(numeric))
        is_frozen = False
        value_range = None
        n_unique = None
        if len(numeric):
            value_range = float(numeric.max() - numeric.min())
            n_unique = int(numeric.nunique())
            magnitude = max(abs(float(numeric.mean())), 1.0)
            is_frozen = n_unique <= 1 or (value_range / magnitude) < FROZEN_RELATIVE_RANGE

        rows.append({
            "file": path.name,
            "external_id": xid,
            "unit": unit,
            "readable": True,
            "error": None,
            "n_points": n_points,
            "n_duplicate_timestamps": n_dup,
            "n_null_values": n_null,
            "n_non_numeric_values": n_non_numeric,
            "first_timestamp": first,
            "last_timestamp": last,
            "span_days": span_s / 86400.0 if span_s else 0.0,
            "median_interval_seconds": median_interval,
            "max_gap_seconds": max_gap,
            "expected_points": expected,
            "missing_rate": missing_rate,
            "n_unique_values": n_unique,
            "value_range": value_range,
            "is_frozen": is_frozen,
            "bytes": path.stat().st_size,
        })

    return pd.DataFrame(rows)


def build_report(
    assets_df: pd.DataFrame,
    timeseries_df: pd.DataFrame,
    events_df: pd.DataFrame,
    event_profile: dict[str, Any],
    event_assessment: dict[str, Any],
    files_df: pd.DataFrame,
    downloaded_df: pd.DataFrame,
    relationships: dict[str, pd.DataFrame],
    run_meta: dict[str, Any],
) -> str:
    """Assemble the markdown report."""
    dp_stats = analyse_datapoint_files(config.DATAPOINTS_DIR)

    lines: list[str] = []
    add = lines.append

    add("# Data Quality Report — Valhall First-Stage Compressor Subset")
    add("")
    add(f"Generated: `{datetime.now(timezone.utc).isoformat()}`")
    add("")
    add("| Run parameter | Value |")
    add("| --- | --- |")
    for key, value in run_meta.items():
        add(f"| {key} | `{value}` |")
    add("")

    # ---------------- assets ----------------
    add("## 1. Assets")
    add("")
    add(f"- Assets selected: **{len(assets_df)}**")
    if not assets_df.empty:
        add(f"- Hierarchy depth range: {int(assets_df['depth'].min())} – {int(assets_df['depth'].max())}")
        roles = assets_df["role"].value_counts().to_dict()
        add(f"- By structural role: {roles}")
        labelled = assets_df[assets_df["subsystem_labels"].astype(str) != ""]
        add(f"- Assets carrying a subsystem label: {len(labelled)} / {len(assets_df)}")
        add("")
        add("### Subsystem coverage")
        add("")
        add("| Subsystem | Assets matched |")
        add("| --- | --- |")
        counts: dict[str, int] = {k: 0 for k in config.SUBSYSTEM_KEYWORDS}
        for raw in assets_df["subsystem_labels"].astype(str):
            for label in raw.split(","):
                if label:
                    counts[label] = counts.get(label, 0) + 1
        for subsystem in config.SUBSYSTEM_KEYWORDS:
            n = counts.get(subsystem, 0)
            note = "" if n else " _(not present in this subtree)_"
            add(f"| {subsystem} | {n}{note} |")
    add("")

    # ---------------- time series ----------------
    add("## 2. Time series")
    add("")
    add(f"- Time series linked to selected assets: **{len(timeseries_df)}**")
    if not timeseries_df.empty:
        n_string = int(timeseries_df["is_string"].fillna(False).sum())
        n_step = int(timeseries_df["is_step"].fillna(False).sum())
        add(f"- Numeric: {len(timeseries_df) - n_string} | String: {n_string}")
        add(f"- Step series (must be forward-filled, not interpolated): {n_step}")
        n_unlinked = int(timeseries_df["asset_id"].isna().sum())
        add(f"- Series with no asset link: {n_unlinked}")
        add("")
        add("### Units present")
        add("")
        units = timeseries_df["unit"].fillna("(no unit)").value_counts()
        add("| Unit | Series |")
        add("| --- | --- |")
        for unit, count in units.items():
            add(f"| `{unit}` | {count} |")
        n_missing_unit = int(timeseries_df["unit"].isna().sum())
        if n_missing_unit:
            add("")
            add(f"> **{n_missing_unit} series have no unit declared.** Any physics-based "
                "threshold or cross-tag comparison involving these is unsafe until the "
                "unit is established from the tag description or the source system.")
    add("")

    # ---------------- telemetry quality ----------------
    add("## 3. Telemetry coverage and quality")
    add("")
    if dp_stats.empty:
        add("_No datapoint files were written (datapoint download skipped or empty)._")
    else:
        with_data = dp_stats[dp_stats["n_points"] > 0]
        add(f"- Datapoint files written: **{len(dp_stats)}**")
        add(f"- Files containing data: **{len(with_data)}**")
        add(f"- Total datapoints: **{int(dp_stats['n_points'].sum()):,}**")
        add(f"- Total size on disk: **{dp_stats['bytes'].sum() / 1024 / 1024:.1f} MB**")
        if not with_data.empty:
            add(f"- Time coverage: `{with_data['first_timestamp'].min()}` → "
                f"`{with_data['last_timestamp'].max()}` "
                f"({with_data['span_days'].max():.1f} days max span)")
            add("")
            add("### Quality flags")
            add("")
            n_dup = int((with_data["n_duplicate_timestamps"] > 0).sum())
            n_frozen = int(with_data["is_frozen"].sum())
            n_gappy = int((with_data["missing_rate"].fillna(0) > 0.05).sum())
            n_nulls = int((with_data["n_null_values"] > 0).sum())
            add(f"- Series with duplicate timestamps: **{n_dup}** "
                f"(total duplicates: {int(with_data['n_duplicate_timestamps'].sum()):,})")
            add(f"- Series with >5% missing data vs their own median rate: **{n_gappy}**")
            add(f"- Series with null values: **{n_nulls}**")
            add(f"- **Frozen / constant series: {n_frozen}** "
                "— exclude these from modelling until the transmitter is verified")
            add("")
            if n_frozen:
                add("#### Frozen or constant channels")
                add("")
                add("| Tag | Points | Unique values | Range |")
                add("| --- | --- | --- | --- |")
                for _, row in with_data[with_data["is_frozen"]].iterrows():
                    add(f"| `{row['external_id']}` | {int(row['n_points']):,} | "
                        f"{_fmt(row['n_unique_values'])} | {_fmt(row['value_range'], 6)} |")
                add("")
            add("### Per-tag detail")
            add("")
            add("| Tag | Unit | Points | Median interval (s) | Max gap (s) | Missing | Dups | Frozen |")
            add("| --- | --- | --- | --- | --- | --- | --- | --- |")
            for _, row in with_data.sort_values("external_id").iterrows():
                missing = row["missing_rate"]
                missing_str = f"{missing * 100:.1f}%" if missing is not None and not pd.isna(missing) else "n/a"
                add(
                    f"| `{row['external_id']}` | {_unit(row['unit'])} | {int(row['n_points']):,} | "
                    f"{_fmt(row['median_interval_seconds'])} | {_fmt(row['max_gap_seconds'])} | "
                    f"{missing_str} | {int(row['n_duplicate_timestamps'])} | "
                    f"{'YES' if row['is_frozen'] else 'no'} |"
                )
        empty_files = dp_stats[dp_stats["n_points"] == 0]
        if not empty_files.empty:
            add("")
            add(f"> {len(empty_files)} series returned no datapoints in the requested window. "
                "This is expected for tags that were decommissioned or not yet "
                "commissioned during the window — widen `--start`/`--end` to check.")
    add("")

    # ---------------- events ----------------
    add("## 4. Events / maintenance context")
    add("")
    add(f"- Events found: **{event_profile.get('n_events', 0)}**")
    if event_profile.get("n_events"):
        add(f"- With asset links: {event_profile.get('n_with_assets')}")
        add(f"- With an end_time: {event_profile.get('n_with_end_time')}")
        add(f"- Span: `{event_profile.get('min_start')}` → `{event_profile.get('max_start')}`")
        median = event_profile.get("median_duration_hours")
        if median is not None:
            add(f"- Median duration: {median:.3f} hours")
        add(f"- Types observed: `{list((event_profile.get('types') or {}).keys())}`")
        add(f"- Subtypes observed: `{list((event_profile.get('subtypes') or {}).keys())[:15]}`")
        add(f"- Metadata keys observed: `{list((event_profile.get('metadata_keys') or {}).keys())[:20]}`")
    add("")
    add("### What do these records represent?")
    add("")
    add(f"- **Assessment:** {event_assessment.get('verdict')}")
    add(f"- **Confidence:** {event_assessment.get('confidence')}")
    if event_assessment.get("caveat"):
        add(f"- **Caveat:** {event_assessment['caveat']}")
    evidence = event_assessment.get("evidence") or []
    if evidence:
        add("- **Evidence:**")
        for category, detail in evidence:
            add(f"  - `{category}` — score {detail['score']}, "
                f"schema matches: {detail['structural_matches']}, "
                f"text matches: {detail['description_matches']}")
    add("")

    # ---------------- files ----------------
    add("## 5. Engineering files and P&IDs")
    add("")
    add(f"- Documents catalogued: **{len(files_df)}**")
    if not files_df.empty:
        add(f"- Positively scored (relevant): **{int((files_df['relevance_score'] > 0).sum())}**")
        add(f"- **Likely P&IDs / engineering drawings: {int(files_df['likely_engineering_drawing'].sum())}**")
        add(f"- Linked to a selected asset: {int((files_df['n_linked_selected_assets'] > 0).sum())}")
        add(f"- Tag appears in filename: {int(files_df['tag_in_filename'].sum())}")
        mimes = files_df["mime_type"].fillna("(unknown)").value_counts().head(10).to_dict()
        add(f"- MIME types: {mimes}")
    if downloaded_df is not None and not downloaded_df.empty and "downloaded" in downloaded_df:
        n_ok = int(downloaded_df["downloaded"].fillna(False).sum())
        add(f"- Downloaded to `data/files/documents/`: **{n_ok} / {len(downloaded_df)}**")
        drawings = downloaded_df[
            downloaded_df["likely_engineering_drawing"] & downloaded_df["downloaded"].fillna(False)
        ]
        if not drawings.empty:
            add("")
            add("#### Drawings downloaded")
            add("")
            add("| File | MIME | Matched via |")
            add("| --- | --- | --- |")
            for _, row in drawings.iterrows():
                add(f"| `{row['filename']}` | {row['mime_type']} | {row['match_reasons']} |")
    else:
        add("- No documents downloaded (skipped, or no positively scored candidates).")
    add("")

    # ---------------- relationships ----------------
    add("## 6. Relationships / graph readiness")
    add("")
    explicit = relationships.get("explicit", pd.DataFrame())
    implicit = relationships.get("implicit", pd.DataFrame())
    models = relationships.get("data_models", pd.DataFrame())
    add(f"- Explicit CDF Relationship edges: **{len(explicit)}**")
    add(f"- Data-modelling spaces / models visible: **{len(models)}**")
    add(f"- Implicit edges derived from foreign keys: **{len(implicit)}**")
    if not implicit.empty:
        add("")
        add("| Edge type | Count | Derived from |")
        add("| --- | --- | --- |")
        grouped = implicit.groupby(["relationship", "origin"]).size()
        for (rel, origin), count in grouped.items():
            add(f"| `{rel}` | {count} | `{origin}` |")
    if explicit.empty:
        add("")
        add("> No explicit Relationship rows exist for these assets in this project. "
            "The knowledge graph is still fully constructible from "
            "`relationships/implicit_edges.parquet`. What CDF metadata cannot give "
            "you is **process topology** (which stream flows into which vessel) — "
            "that has to come from annotating the P&IDs.")
    add("")

    # ---------------- verdict ----------------
    add("## 7. Fitness for modelling")
    add("")
    blockers: list[str] = []
    warnings: list[str] = []

    if timeseries_df.empty:
        blockers.append("No time series were found for the selected assets.")
    if not dp_stats.empty:
        with_data = dp_stats[dp_stats["n_points"] > 0]
        if with_data.empty:
            blockers.append("No datapoints landed on disk for the requested window.")
        else:
            n_frozen = int(with_data["is_frozen"].sum())
            if n_frozen:
                warnings.append(
                    f"{n_frozen} channel(s) are frozen/constant and must be excluded "
                    "from feature sets."
                )
            n_gappy = int((with_data["missing_rate"].fillna(0) > 0.05).sum())
            if n_gappy:
                warnings.append(
                    f"{n_gappy} channel(s) are missing more than 5% of their expected "
                    "samples; resample with an explicit gap policy rather than "
                    "interpolating blindly."
                )
            if int(with_data["n_duplicate_timestamps"].sum()):
                warnings.append(
                    "Duplicate timestamps present; de-duplicate before setting a "
                    "DatetimeIndex."
                )
    if not timeseries_df.empty and int(timeseries_df["unit"].isna().sum()):
        warnings.append(
            f"{int(timeseries_df['unit'].isna().sum())} series lack units; "
            "physics-based thresholds are unsafe for those."
        )
    if event_profile.get("n_events", 0) == 0:
        warnings.append(
            "No events found, so there is no maintenance/failure history to label "
            "anomalies against. Supervised failure prediction is not possible on "
            "this subset; unsupervised anomaly detection is."
        )
    if not files_df.empty and int(files_df["likely_engineering_drawing"].sum()) == 0:
        warnings.append("No document matched the drawing heuristics — P&ID context is absent.")

    if blockers:
        add("**Blockers**")
        add("")
        for item in blockers:
            add(f"- ❌ {item}")
        add("")
    add("**Warnings**")
    add("")
    if warnings:
        for item in warnings:
            add(f"- ⚠️ {item}")
    else:
        add("- None.")
    add("")
    if not blockers:
        add("**Verdict:** the subset is usable for unsupervised anomaly detection on the "
            "numeric channels, subject to the warnings above.")
    else:
        add("**Verdict:** not yet usable — resolve the blockers above.")
    add("")

    return "\n".join(lines)


def save_report(content: str) -> Path:
    path = config.DATA_DIR / "data_quality_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    logger.info("Wrote data quality report -> %s (%.1f KB)", path, path.stat().st_size / 1024)
    return path
