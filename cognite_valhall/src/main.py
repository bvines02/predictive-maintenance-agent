"""Pipeline entry point: ``python -m src.main``.

DESIGN NOTE — why the stages are sequential functions, not a framework
----------------------------------------------------------------------
This pipeline is a linear dependency chain: assets define the telemetry scope,
which defines the quality report. There is no fan-out to parallelise and no
partial-failure topology to manage, so a DAG framework (Airflow, Dagster,
Prefect) would add operational weight and buy nothing. Plain functions with
explicit arguments are easier to read, test and debug.

What would change in production
-------------------------------
The shape changes once the pipeline becomes *incremental and scheduled* rather
than a one-shot extraction:

* **Orchestration.** Each stage becomes a task with retries and alerting, so a
  transient failure at 03:00 does not silently leave you with yesterday's data.
* **Watermarks.** Instead of ``--start/--end``, you persist the last successful
  timestamp per tag and resume from it. That is the single most important
  change, and it is why every datapoint file here carries explicit
  ``external_id`` and UTC timestamps: those are exactly the columns a watermark
  table joins on.
* **Streaming.** For live monitoring you would subscribe to CDF's datapoint
  subscriptions rather than polling windows.
* **Storage.** Local Parquet becomes object storage with a table format (Delta
  or Iceberg) so that concurrent readers, schema evolution and time travel are
  handled for you.
* **Validation as a gate.** The quality report becomes an assertion suite that
  *fails the run* rather than a document a human reads.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

from . import config
from .cognite_client import get_verified_client
from .discover_assets import discover, print_hierarchy
from .download_assets import save_assets
from .download_events import save_events
from .download_files import save_files_catalogue
from .download_relationships import save_relationships
from .download_timeseries import (
    discover_timeseries,
    download_datapoints,
    print_coverage,
    probe_coverage,
    resolve_window,
    save_timeseries_metadata,
)
from .quality_report import build_report, save_report

logger = logging.getLogger("valhall")


def setup_logging(verbose: bool = False) -> None:
    """Configure logging once, for the whole process.

    Progress goes to stderr so that stdout carries only the human-facing
    reports. That separation means ``python -m src.main > summary.txt`` captures
    the readable output while logs still stream to the terminal.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    # The SDK's own logger is chatty at INFO during large downloads.
    logging.getLogger("cognite").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("msal").setLevel(logging.WARNING)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description=(
            "Download a focused, correctly-contextualised subset of the Cognite / "
            "Aker BP Open Industrial Data set for the Valhall first-stage gas "
            "compressor train."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # default: 60 days of 1-minute averages around 23-KA-9101
  python -m src.main

  # explicit window, raw resolution (WARNING: much larger)
  python -m src.main --start 2024-01-01 --end 2024-03-01 --raw

  # a different machine
  python -m src.main --asset 23-KA-9102

  # metadata and structure only, no telemetry or document bytes
  python -m src.main --skip-datapoints --skip-files

  # expand to two years of hourly averages
  python -m src.main --start 2023-01-01 --granularity 1h --yes
""",
    )
    parser.add_argument("--asset", default=config.DEFAULT_ASSET,
                        help=f"target asset tag or external id (default: {config.DEFAULT_ASSET})")
    parser.add_argument("--start", default=None,
                        help="window start, ISO 8601 (e.g. 2024-01-01 or 2024-01-01T00:00:00Z)")
    parser.add_argument("--end", default=None,
                        help="window end, ISO 8601. Defaults to the latest datapoint present.")
    parser.add_argument("--lookback-days", type=int, default=config.DEFAULT_LOOKBACK_DAYS,
                        help=f"days before --end when --start is omitted (default: {config.DEFAULT_LOOKBACK_DAYS})")
    parser.add_argument("--granularity", default="1m",
                        help="aggregate granularity, e.g. 1s/1m/15m/1h/1d (default: 1m)")
    parser.add_argument("--aggregate", default="average",
                        help="server-side aggregate to fetch (default: average)")
    parser.add_argument("--raw", action="store_true",
                        help="fetch raw datapoints instead of aggregates (much larger)")
    parser.add_argument("--skip-datapoints", action="store_true",
                        help="discover telemetry metadata but do not download values")
    parser.add_argument("--skip-files", action="store_true",
                        help="catalogue documents but do not download bytes")
    parser.add_argument("--max-files", type=int, default=config.MAX_FILES_TO_DOWNLOAD,
                        help=f"cap on documents downloaded (default: {config.MAX_FILES_TO_DOWNLOAD})")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="proceed without confirmation on a large estimated download")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser.parse_args(argv)


def confirm_large_download(estimated_points: int | None, assume_yes: bool) -> bool:
    """Gate a large download behind an explicit confirmation.

    Deliberately a *hard stop* rather than a warning that scrolls past. The
    failure mode we are preventing is someone running the default command,
    walking away, and returning to a saturated disk and a rate-limited token.
    """
    if estimated_points is None or estimated_points < config.LARGE_DOWNLOAD_THRESHOLD:
        return True
    print()
    print("!" * 78)
    print(f"LARGE DOWNLOAD: estimated ~{estimated_points:,} datapoints "
          f"(threshold {config.LARGE_DOWNLOAD_THRESHOLD:,}).")
    print("Consider narrowing --start/--end, or using a coarser --granularity.")
    print("!" * 78)
    if assume_yes:
        print("--yes supplied; proceeding.")
        return True
    try:
        answer = input("Proceed anyway? [y/N] ").strip().lower()
    except EOFError:
        answer = "n"
    return answer in {"y", "yes"}


def print_final_summary(
    asset_tag: str,
    assets_df: pd.DataFrame,
    timeseries_df: pd.DataFrame,
    dp_summary: pd.DataFrame,
    events_df: pd.DataFrame,
    event_assessment: dict[str, object],
    files_df: pd.DataFrame,
    downloaded_df: pd.DataFrame,
    relationships: dict[str, pd.DataFrame],
    window: tuple[datetime, datetime],
) -> None:
    start, end = window
    days = (end - start).total_seconds() / 86400

    n_tags_with_data = 0
    total_points = 0
    if dp_summary is not None and not dp_summary.empty and "rows" in dp_summary:
        n_tags_with_data = int((dp_summary["rows"] > 0).sum())
        total_points = int(dp_summary["rows"].sum())

    n_drawings = int(files_df["likely_engineering_drawing"].sum()) if not files_df.empty else 0
    n_downloaded = 0
    if downloaded_df is not None and not downloaded_df.empty and "downloaded" in downloaded_df:
        n_downloaded = int(downloaded_df["downloaded"].fillna(False).sum())

    explicit = len(relationships.get("explicit", pd.DataFrame()))
    implicit = len(relationships.get("implicit", pd.DataFrame()))

    print()
    print("COMPRESSOR SYSTEM")
    print(f"├── Asset hierarchy         : {len(assets_df)} assets "
          f"(target {asset_tag}, depth "
          f"{int(assets_df['depth'].min()) if not assets_df.empty else 0}"
          f"-{int(assets_df['depth'].max()) if not assets_df.empty else 0})")
    print("├── Telemetry")
    print(f"│   ├── {len(timeseries_df)} tags discovered, {n_tags_with_data} with data")
    print(f"│   └── {days:.0f} days of history "
          f"({start.date()} → {end.date()}), {total_points:,} datapoints")
    print(f"├── Events / maintenance    : {len(events_df)} records "
          f"— assessed as: {event_assessment.get('verdict')} "
          f"(confidence: {event_assessment.get('confidence')})")
    print(f"├── P&IDs / engineering     : {len(files_df)} catalogued, "
          f"{n_drawings} likely drawings, {n_downloaded} downloaded")
    print(f"└── Relationships           : {explicit} explicit CDF edges, "
          f"{implicit} implicit edges derived from foreign keys")
    print()
    print(f"Artefacts written to: {config.DATA_DIR}")
    print(f"Quality report      : {config.DATA_DIR / 'data_quality_report.md'}")
    print()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    config.ensure_dirs()

    logger.info("=" * 60)
    logger.info("STAGE 1/9  Authenticate")
    logger.info("=" * 60)
    client = get_verified_client()

    logger.info("=" * 60)
    logger.info("STAGE 2/9  Discover compressor hierarchy")
    logger.info("=" * 60)
    try:
        result = discover(client, args.asset)
    except LookupError as exc:
        logger.error("%s", exc)
        return 3

    logger.info("=" * 60)
    logger.info("STAGE 3/9  Show what was found")
    logger.info("=" * 60)
    print_hierarchy(result)

    logger.info("=" * 60)
    logger.info("STAGE 4/9  Save asset metadata")
    logger.info("=" * 60)
    assets_df = save_assets(result)
    asset_ids = result.selected_ids
    asset_lookup = {
        a.id: a.external_id for a in result.selected if a.id is not None and a.external_id
    }
    asset_external_ids = [x for x in asset_lookup.values() if x]

    logger.info("=" * 60)
    logger.info("STAGE 5/9  Discover telemetry")
    logger.info("=" * 60)
    series = discover_timeseries(client, asset_ids)
    timeseries_df = save_timeseries_metadata(series, asset_lookup)

    # Probe before committing: we need the latest-known timestamp to anchor the
    # default window, so the probe runs against a provisional window first.
    provisional_end = datetime.now(timezone.utc)
    provisional_start = provisional_end - timedelta(days=args.lookback_days)
    estimate = probe_coverage(client, series, provisional_start, provisional_end)

    try:
        window_start, window_end = resolve_window(
            args.start, args.end, args.lookback_days, latest_known=estimate.latest
        )
    except ValueError as exc:
        logger.error("%s", exc)
        return 4

    # Re-estimate volume against the window we actually resolved.
    estimate.window_start, estimate.window_end = window_start, window_end
    if estimate.median_interval_seconds and estimate.n_numeric:
        per_series = (window_end - window_start).total_seconds() / estimate.median_interval_seconds
        estimate.estimated_points = int(per_series * estimate.n_numeric)
    print_coverage(estimate, n_assets=len(assets_df))

    logger.info("=" * 60)
    logger.info("STAGE 6/9  Download telemetry")
    logger.info("=" * 60)
    dp_summary = pd.DataFrame()
    if args.skip_datapoints:
        logger.info("--skip-datapoints set; skipping datapoint download")
    else:
        aggregate = None if args.raw else args.aggregate
        # Raw mode is what actually risks a huge download; aggregates are bounded
        # by granularity, so we only gate the raw path on the estimate.
        if aggregate is None and not confirm_large_download(estimate.estimated_points, args.yes):
            logger.warning("Aborted by user at the download confirmation prompt")
            return 5
        dp_summary = download_datapoints(
            client, series, window_start, window_end,
            aggregate=aggregate, granularity=args.granularity,
        )

    logger.info("=" * 60)
    logger.info("STAGE 7/9  Download events / maintenance history")
    logger.info("=" * 60)
    root_id = result.ancestors[0].id if result.ancestors else result.target.id
    events_df, event_profile, event_assessment = save_events(
        client, asset_ids, asset_lookup, subtree_root_id=root_id
    )

    logger.info("=" * 60)
    logger.info("STAGE 8/9  Find and download engineering documents")
    logger.info("=" * 60)
    files_df, downloaded_df = save_files_catalogue(
        client,
        asset_ids,
        asset_lookup,
        target_tag=args.asset,
        extra_tags=[result.target.name] if result.target.name != args.asset else None,
        max_files=args.max_files,
        skip_download=args.skip_files,
    )

    logger.info("=" * 60)
    logger.info("STAGE 8b/9 Discover relationships / graph context")
    logger.info("=" * 60)
    relationships = save_relationships(
        client, asset_external_ids, assets_df, timeseries_df, events_df, files_df
    )

    logger.info("=" * 60)
    logger.info("STAGE 9/9  Produce data quality report")
    logger.info("=" * 60)
    run_meta = {
        "target asset": args.asset,
        "resolved target": f"{result.target.name} (id={result.target.id}, "
                           f"xid={result.target.external_id})",
        "CDF project": config.COGNITE_PROJECT,
        "base url": config.COGNITE_BASE_URL,
        "window start (UTC)": window_start.isoformat(),
        "window end (UTC)": window_end.isoformat(),
        "mode": "raw datapoints" if args.raw else f"{args.aggregate} @ {args.granularity}",
        "datapoints skipped": args.skip_datapoints,
        "files skipped": args.skip_files,
    }
    report = build_report(
        assets_df=assets_df,
        timeseries_df=timeseries_df,
        events_df=events_df,
        event_profile=event_profile,
        event_assessment=event_assessment,
        files_df=files_df,
        downloaded_df=downloaded_df,
        relationships=relationships,
        run_meta=run_meta,
    )
    save_report(report)

    print_final_summary(
        asset_tag=args.asset,
        assets_df=assets_df,
        timeseries_df=timeseries_df,
        dp_summary=dp_summary,
        events_df=events_df,
        event_assessment=event_assessment,
        files_df=files_df,
        downloaded_df=downloaded_df,
        relationships=relationships,
        window=(window_start, window_end),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
