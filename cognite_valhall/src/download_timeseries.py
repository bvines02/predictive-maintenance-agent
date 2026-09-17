"""Discover time-series metadata and download datapoints.

DESIGN NOTE — how SCADA tags relate to assets in CDF
----------------------------------------------------
In a real plant the chain is:

  physical sensor -> DCS/SCADA tag -> historian series -> CDF time series

A CDF ``TimeSeries`` is the *series*, not the sensor. It carries an
``asset_id`` foreign key pointing at the equipment it measures. That link is
many-to-one: one compressor has dozens of series (suction pressure, discharge
pressure, discharge temperature, vibration, speed, lube-oil pressure...), and
the link is what lets us ask "give me every measurement on this machine"
instead of guessing from tag-name conventions.

Crucially, the link is *optional and imperfect* in real deployments. Tags get
ingested before the hierarchy is built, or attached to the wrong level (the
train rather than the machine). So we report unlinked-but-name-matching series
rather than pretending the ``asset_id`` link is complete. In production you
would reconcile this with a tag-to-equipment mapping from the DCS
configuration, and treat disagreements as a data-quality ticket.

DESIGN NOTE — how CDF pagination works
--------------------------------------
Two different mechanisms, often confused:

1. **Resource listing** (assets/time series/events/files) is *cursor*-based.
   Each response carries a ``nextCursor``; you pass it back to get the next
   page. The SDK hides this: ``limit=None`` means "keep following cursors until
   exhausted". Offset pagination is deliberately not offered, because with
   concurrent writes offsets skip and duplicate rows. For large listings you
   can also pass ``partitions=N``, which splits the keyspace so N workers
   paginate disjoint slices in parallel.

2. **Datapoint retrieval** is *time-window*-based, not cursor-based. One
   request returns at most 100,000 raw points (or 10,000 aggregate rows) per
   series, and the window you asked for is truncated. To continue you re-query
   from the last timestamp you received. The SDK does this internally and also
   parallelises across series and sub-windows. We still chunk by time
   ourselves, for the memory reason below.

DESIGN NOTE — how batching works, and why we chunk twice
--------------------------------------------------------
The datapoints endpoint accepts up to 100 series per request. Requesting 100
series in one call rather than 100 sequential calls is the single biggest
performance lever: it amortises TLS, auth and HTTP overhead, and lets the
SDK's thread pool saturate the connection.

But a batch that is wide (100 series) *and* long (2 years at 1 Hz) would
materialise ~6 billion points in RAM. So we chunk on both axes:

* **width**: ``DATAPOINTS_TS_PER_REQUEST`` series per request (API limit),
* **length**: ``RAW_CHUNK_DAYS`` days per request window.

Each (width x length) block is fetched, converted, appended to its per-tag
writer, and released. Peak memory is bounded by one block regardless of how
long a history you ask for.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from cognite.client import CogniteClient
from cognite.client.data_classes import Datapoints, TimeSeries
from cognite.client.exceptions import CogniteAPIError

from . import config
from .storage import json_or_none, ms_to_utc, safe_filename, write_table

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Metadata discovery
# --------------------------------------------------------------------------
def discover_timeseries(client: CogniteClient, asset_ids: list[int]) -> list[TimeSeries]:
    """List every time series linked to any of the selected assets.

    We filter on ``asset_ids`` (the explicit list) rather than
    ``asset_subtree_ids``. Both would work here, but the explicit list makes the
    extraction *exactly* reproducible from the saved assets table: whatever is
    in ``assets.parquet`` defines the telemetry scope, with no hidden server-side
    traversal that could pick up assets added to the subtree after the fact.

    ``asset_ids`` has a per-request cap, so we chunk it. This is the listing
    pagination described in the module docstring: ``limit=None`` follows cursors
    to exhaustion within each chunk.
    """
    if not asset_ids:
        return []

    found: dict[int, TimeSeries] = {}
    chunk_size = 100
    for offset in range(0, len(asset_ids), chunk_size):
        chunk = asset_ids[offset : offset + chunk_size]
        logger.info(
            "Listing time series for assets %d-%d of %d ...",
            offset + 1,
            min(offset + chunk_size, len(asset_ids)),
            len(asset_ids),
        )
        series = client.time_series.list(asset_ids=chunk, limit=config.LIST_ALL)
        for ts in series:
            if ts.id is not None:
                found[ts.id] = ts

    result = sorted(found.values(), key=lambda t: (t.external_id or "", t.id or 0))
    logger.info("Discovered %d distinct time series", len(result))
    return result


def timeseries_to_dataframe(
    series: list[TimeSeries], asset_lookup: dict[int, str] | None = None
) -> pd.DataFrame:
    """Flatten time-series metadata, keeping every field the agent may need."""
    asset_lookup = asset_lookup or {}
    rows = []
    for ts in series:
        rows.append(
            {
                "timeseries_id": ts.id,
                "external_id": ts.external_id,
                "name": ts.name,
                "description": ts.description,
                "unit": ts.unit,
                "unit_external_id": ts.unit_external_id,
                "is_string": ts.is_string,
                # is_step matters for interpolation: a step series (e.g. a valve
                # position setpoint) must be forward-filled, while a continuous
                # series (e.g. a pressure) may be linearly interpolated.
                # Getting this wrong quietly corrupts resampled features.
                "is_step": ts.is_step,
                "asset_id": ts.asset_id,
                "asset_external_id": asset_lookup.get(ts.asset_id) if ts.asset_id else None,
                "data_set_id": ts.data_set_id,
                "security_categories": json_or_none(ts.security_categories),
                "metadata": json_or_none(ts.metadata),
                "created_time": ms_to_utc(ts.created_time),
                "last_updated_time": ms_to_utc(ts.last_updated_time),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["asset_external_id", "external_id"], na_position="last")
        df = df.reset_index(drop=True)
    return df


def save_timeseries_metadata(
    series: list[TimeSeries], asset_lookup: dict[int, str] | None = None
) -> pd.DataFrame:
    df = timeseries_to_dataframe(series, asset_lookup)
    write_table(
        df,
        config.TIMESERIES_DIR / "timeseries_metadata.parquet",
        config.TIMESERIES_DIR / "timeseries_metadata.csv",
        label="timeseries metadata",
    )
    return df


# --------------------------------------------------------------------------
# Coverage probing
# --------------------------------------------------------------------------
@dataclass
class CoverageEstimate:
    """What we learned about the extent and density of the data, before download."""

    n_series: int
    n_numeric: int
    n_string: int
    earliest: pd.Timestamp | None
    latest: pd.Timestamp | None
    median_interval_seconds: float | None
    estimated_points: int | None
    window_start: datetime
    window_end: datetime


def probe_coverage(
    client: CogniteClient,
    series: list[TimeSeries],
    window_start: datetime,
    window_end: datetime,
    sample_size: int = 12,
) -> CoverageEstimate:
    """Estimate extent and volume *before* committing to a download.

    Three cheap probes, deliberately in this order:

    1. ``retrieve_latest`` across all series — one request, gives the newest
       timestamp per tag. This tells us whether the data is live.
    2. ``retrieve`` with ``start=0, limit=1`` per sampled series — the oldest
       point, which establishes how far history goes back.
    3. A ``count`` aggregate over a short recent window on a sample — this is
       the key trick. Rather than downloading data to find out how much there
       is, we ask the server to count it. ``count`` with a 1-day granularity
       returns one number per day per series, so the density estimate costs
       almost nothing.

    We sample rather than probe all series because sampling is sufficient to
    choose a sensible default window, and an exhaustive probe on a large tag set
    is itself a slow operation. In production you would cache these per-tag
    statistics in a catalogue table and refresh them on a schedule.
    """
    numeric = [ts for ts in series if not ts.is_string and ts.external_id]
    strings = [ts for ts in series if ts.is_string]

    earliest: pd.Timestamp | None = None
    latest: pd.Timestamp | None = None
    median_interval: float | None = None
    estimated_points: int | None = None

    if not numeric:
        return CoverageEstimate(
            n_series=len(series), n_numeric=0, n_string=len(strings),
            earliest=None, latest=None, median_interval_seconds=None,
            estimated_points=None, window_start=window_start, window_end=window_end,
        )

    # --- probe 1: newest point across every numeric series (single request) ---
    # NOTE on SDK versions: in cognite-sdk 8.x, retrieve_latest returns
    # LatestDatapoint objects whose `timestamp` is a single datetime (or None
    # when the series is empty). Older majors returned Datapoints objects whose
    # `timestamp` was a *list* of epoch-ms ints. We normalise both shapes so the
    # probe does not silently break on an SDK upgrade.
    try:
        latest_points = client.time_series.data.retrieve_latest(
            external_id=[ts.external_id for ts in numeric],
            ignore_unknown_ids=True,
        )
        stamps: list[pd.Timestamp] = []
        for dp in latest_points or []:
            raw = getattr(dp, "timestamp", None)
            if raw is None:
                continue
            if isinstance(raw, (list, tuple)):
                raw = raw[0] if raw else None
            if raw is None:
                continue
            stamps.append(
                ms_to_utc(raw) if isinstance(raw, (int, float))
                else pd.Timestamp(raw).tz_convert("UTC")
                if pd.Timestamp(raw).tzinfo
                else pd.Timestamp(raw).tz_localize("UTC")
            )
        stamps = [s for s in stamps if s is not None]
        if stamps:
            latest = max(stamps)
    except CogniteAPIError as exc:
        logger.warning("Could not probe latest datapoints: %s", exc)

    # --- probe 2: oldest point on a sample (start=0 means the epoch) ---
    sample = numeric[:sample_size]
    oldest_stamps: list[int] = []
    for ts in sample:
        try:
            dps = client.time_series.data.retrieve(
                external_id=ts.external_id, start=0, end="now", limit=1
            )
            if dps is not None and len(dps) > 0:
                oldest_stamps.append(dps.timestamp[0])
        except CogniteAPIError as exc:
            logger.debug("Oldest-point probe failed for %s: %s", ts.external_id, exc)
    if oldest_stamps:
        earliest = ms_to_utc(min(oldest_stamps))

    # --- probe 3: server-side count over a recent slice, to get density ---
    probe_days = 3
    probe_start = window_end - timedelta(days=probe_days)
    try:
        counts = client.time_series.data.retrieve_dataframe(
            external_id=[ts.external_id for ts in sample],
            start=probe_start,
            end=window_end,
            aggregates="count",
            granularity="1d",
            ignore_unknown_ids=True,
            include_aggregate_name=False,
        )
        if counts is not None and not counts.empty:
            per_series_per_day = counts.sum(axis=0) / max(probe_days, 1)
            mean_per_day = float(per_series_per_day.mean())
            if mean_per_day > 0:
                median_interval = 86400.0 / mean_per_day
                window_days = max((window_end - window_start).total_seconds() / 86400.0, 0)
                estimated_points = int(mean_per_day * window_days * len(numeric))
    except CogniteAPIError as exc:
        logger.warning("Could not probe datapoint density: %s", exc)

    return CoverageEstimate(
        n_series=len(series),
        n_numeric=len(numeric),
        n_string=len(strings),
        earliest=earliest,
        latest=latest,
        median_interval_seconds=median_interval,
        estimated_points=estimated_points,
        window_start=window_start,
        window_end=window_end,
    )


def print_coverage(estimate: CoverageEstimate, n_assets: int) -> None:
    """Show the user what is about to be downloaded, before it happens."""
    print()
    print("=" * 78)
    print("TELEMETRY SCOPE")
    print("=" * 78)
    print(f"  Selected assets            : {n_assets}")
    print(f"  Associated time series     : {estimate.n_series} "
          f"({estimate.n_numeric} numeric, {estimate.n_string} string)")
    print(f"  Earliest available point   : {estimate.earliest or 'unknown (probe failed)'}")
    print(f"  Latest available point     : {estimate.latest or 'unknown (probe failed)'}")
    if estimate.median_interval_seconds:
        print(f"  Typical sample interval    : ~{estimate.median_interval_seconds:.1f} s")
    print(f"  Requested window           : {estimate.window_start.isoformat()} "
          f"-> {estimate.window_end.isoformat()}")
    window_days = (estimate.window_end - estimate.window_start).total_seconds() / 86400
    print(f"  Requested window length    : {window_days:.1f} days")
    if estimate.estimated_points is not None:
        mb = estimate.estimated_points * 12 / 1024 / 1024  # ~12 B/point in Parquet
        print(f"  Estimated raw datapoints   : ~{estimate.estimated_points:,}")
        print(f"  Estimated Parquet size     : ~{mb:,.0f} MB (compressed)")
    else:
        print("  Estimated raw datapoints   : unknown (density probe failed)")
    print("=" * 78)
    print()


# --------------------------------------------------------------------------
# Datapoint download
# --------------------------------------------------------------------------
def _time_chunks(
    start: datetime, end: datetime, chunk_days: int
) -> list[tuple[datetime, datetime]]:
    """Split a window into consecutive half-open sub-windows."""
    chunks: list[tuple[datetime, datetime]] = []
    cursor = start
    step = timedelta(days=max(chunk_days, 1))
    while cursor < end:
        chunk_end = min(cursor + step, end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end
    return chunks


class _TagWriter:
    """Streams one tag's datapoints into a single Parquet file, chunk by chunk.

    STORAGE LAYOUT DECISION
    -----------------------
    We write **one Parquet file per time series**, named after its external_id:

        data/timeseries/datapoints/<external_id>.parquet

    Why this rather than one big table, or a Hive-partitioned dataset?

    * The access pattern dominates. Every downstream consumer — anomaly
      detection, feature engineering, plotting one tag — starts by asking for
      *specific tags over a time range*. One file per tag makes that a single
      file open with no filtering at all. A monolithic long-format table would
      require scanning or filtering on every read.
    * Tags are heterogeneous. Pressure in bar, temperature in degC, vibration in
      mm/s and a string status tag do not belong in one typed value column
      without either upcasting everything to float64 (losing the string tags) or
      carrying nullable columns per type. Per-tag files let each file have the
      schema its data actually needs.
    * Incremental extension is trivial. Extending history for one tag rewrites
      one small file. In a shared table it is a read-modify-write of the whole
      thing.
    * Row-group statistics still give us time pushdown *inside* each file,
      because we append in chronological order, so a query for one week skips
      the row groups outside it.

    The main alternative — ``.../datapoints/external_id=<tag>/year=<y>/part.parquet``
    Hive partitioning — is the better choice once a single tag's history stops
    fitting comfortably in memory or you move to Spark/Athena, because the
    partition keys get pruned before any file is opened. At this subsystem scale
    (tens of tags, weeks-to-months of history) it only adds path complexity, so
    we prefer the flat layout and document the upgrade path.

    Implementation note: we hold an open ``pyarrow.parquet.ParquetWriter`` and
    write one row group per time chunk. This is what keeps memory bounded — the
    naive version (concatenate every chunk, then ``to_parquet`` once) defeats
    the entire point of time-chunking.
    """

    def __init__(self, path: Path, external_id: str, timeseries_id: int | None, unit: str | None):
        self.path = path
        self.external_id = external_id
        self.timeseries_id = timeseries_id
        self.unit = unit
        self._writer: pq.ParquetWriter | None = None
        self._schema: pa.Schema | None = None
        self.rows_written = 0

    def write(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        table = pa.Table.from_pandas(df, preserve_index=False)
        if self._writer is None:
            self._schema = table.schema
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(self.path, self._schema, compression="snappy")
        elif not table.schema.equals(self._schema):
            # Can happen if an all-null chunk infers a different type. Cast to
            # the schema established by the first chunk rather than failing.
            table = table.cast(self._schema)
        self._writer.write_table(table)
        self.rows_written += len(df)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def _datapoints_to_frame(
    dps, external_id: str, timeseries_id: int | None, unit: str | None
) -> pd.DataFrame:
    """Convert one SDK ``Datapoints`` object into our on-disk schema.

    The schema is deliberately self-describing: every row carries its
    ``external_id`` and ``timeseries_id``. That is redundant within a per-tag
    file, but it means any single file — or any concatenation of files — can be
    interpreted with no reference to the metadata table. Dictionary encoding
    makes the repeated string column nearly free on disk.
    """
    timestamps = list(getattr(dps, "timestamp", []) or [])
    if not timestamps:
        return pd.DataFrame()

    values = list(getattr(dps, "value", []) or [])
    if len(values) != len(timestamps):
        values = values + [None] * (len(timestamps) - len(values))

    return pd.DataFrame(
        {
            # UTC, tz-aware, millisecond precision -- matching CDF's own storage.
            "timestamp": pd.to_datetime(timestamps, unit="ms", utc=True),
            "value": values,
            "external_id": external_id,
            "timeseries_id": timeseries_id,
            "unit": unit,
        }
    )


def download_datapoints(
    client: CogniteClient,
    series: list[TimeSeries],
    start: datetime,
    end: datetime,
    aggregate: str | None = None,
    granularity: str = "1m",
) -> pd.DataFrame:
    """Download datapoints for every series, streaming to per-tag Parquet files.

    Set ``aggregate`` (e.g. ``"average"``) to fetch server-side aggregates
    instead of raw points. That is often the *right* choice for a first
    modelling pass: a 1-minute average over 60 days is ~86k rows per tag rather
    than ~5M, trains an anomaly detector just as well for slow thermal and
    process drift, and downloads in seconds. Raw resolution matters when you are
    chasing fast phenomena — surge cycles, vibration transients, trip sequences.
    """
    string_series = [ts for ts in series if ts.is_string]
    if aggregate is None:
        # Raw retrieval works for string series too, so keep them: a status or
        # mode tag ("RUNNING"/"STOPPED") is valuable troubleshooting context.
        downloadable = [ts for ts in series if ts.external_id]
        if string_series:
            logger.info(
                "Including %d string time series (raw mode supports them)",
                len(string_series),
            )
    else:
        # Numeric aggregates are undefined for string series, so the API cannot
        # serve them here. They are reported, not silently dropped.
        downloadable = [ts for ts in series if ts.external_id and not ts.is_string]
        if string_series:
            logger.info(
                "Excluding %d string time series: numeric aggregates (%s) are not "
                "defined for them. Re-run with --raw to capture their values.",
                len(string_series),
                aggregate,
            )

    if not downloadable:
        logger.warning("No downloadable time series")
        return pd.DataFrame()

    writers: dict[str, _TagWriter] = {}
    for ts in downloadable:
        assert ts.external_id is not None
        filename = safe_filename(ts.external_id) + ".parquet"
        writers[ts.external_id] = _TagWriter(
            config.DATAPOINTS_DIR / filename, ts.external_id, ts.id, ts.unit
        )

    by_xid = {ts.external_id: ts for ts in downloadable}
    chunks = _time_chunks(start, end, config.RAW_CHUNK_DAYS)
    batches = [
        downloadable[i : i + config.DATAPOINTS_TS_PER_REQUEST]
        for i in range(0, len(downloadable), config.DATAPOINTS_TS_PER_REQUEST)
    ]

    total_requests = len(chunks) * len(batches)
    logger.info(
        "Downloading %s for %d series over %d time chunk(s) x %d batch(es) = %d request group(s)",
        f"{aggregate}/{granularity} aggregates" if aggregate else "raw datapoints",
        len(downloadable),
        len(chunks),
        len(batches),
        total_requests,
    )

    done = 0
    failures: list[tuple[str, str]] = []
    try:
        for chunk_start, chunk_end in chunks:
            for batch in batches:
                done += 1
                xids = [ts.external_id for ts in batch]
                logger.info(
                    "  [%d/%d] %s -> %s  (%d series)",
                    done,
                    total_requests,
                    chunk_start.date(),
                    chunk_end.date(),
                    len(xids),
                )
                try:
                    # ignore_unknown_ids keeps one deleted/renamed tag from
                    # aborting the whole batch -- important for long runs.
                    kwargs = dict(
                        external_id=xids,
                        start=chunk_start,
                        end=chunk_end,
                        ignore_unknown_ids=True,
                        limit=None,
                    )
                    if aggregate:
                        kwargs["aggregates"] = aggregate
                        kwargs["granularity"] = granularity
                    result = client.time_series.data.retrieve(**kwargs)
                except CogniteAPIError as exc:
                    # The SDK already retried 429/5xx with backoff. Reaching here
                    # means a persistent failure, so we record it and continue:
                    # a partial dataset plus an explicit failure list beats
                    # losing hours of completed work to one bad window.
                    logger.error(
                        "  request group %d failed (%s); continuing", done, exc
                    )
                    failures.extend((x, str(exc)) for x in xids)
                    continue

                if result is None:
                    continue
                # IMPORTANT: DatapointsList is a collections.UserList, NOT a
                # builtin list, so `isinstance(result, list)` is False for it.
                # Testing for the singular type instead is the robust check --
                # getting this backwards silently treats a whole batch as one
                # series and throws AttributeError on .external_id.
                items = [result] if isinstance(result, Datapoints) else list(result)
                for dps in items:
                    xid = getattr(dps, "external_id", None)
                    if xid is None or xid not in writers:
                        continue
                    ts_meta = by_xid[xid]
                    if aggregate:
                        # Aggregate results expose the values under the aggregate
                        # name (e.g. dps.average) rather than dps.value.
                        agg_values = getattr(dps, aggregate, None)
                        frame = pd.DataFrame()
                        if agg_values is not None and len(agg_values) > 0:
                            frame = pd.DataFrame(
                                {
                                    "timestamp": pd.to_datetime(
                                        list(dps.timestamp), unit="ms", utc=True
                                    ),
                                    "value": list(agg_values),
                                    "external_id": xid,
                                    "timeseries_id": ts_meta.id,
                                    "unit": ts_meta.unit,
                                }
                            )
                    else:
                        frame = _datapoints_to_frame(dps, xid, ts_meta.id, ts_meta.unit)
                    writers[xid].write(frame)
    finally:
        # Always close writers, even on Ctrl-C: an unclosed ParquetWriter leaves
        # a file with no footer, which is unreadable. This turns an interrupted
        # run into a valid partial dataset.
        for writer in writers.values():
            writer.close()

    summary = pd.DataFrame(
        [
            {
                "external_id": w.external_id,
                "timeseries_id": w.timeseries_id,
                "unit": w.unit,
                "rows": w.rows_written,
                "file": w.path.name if w.rows_written else None,
                "bytes": w.path.stat().st_size if w.path.exists() else 0,
            }
            for w in writers.values()
        ]
    )

    non_empty = int((summary["rows"] > 0).sum()) if not summary.empty else 0
    total_rows = int(summary["rows"].sum()) if not summary.empty else 0
    logger.info(
        "Datapoint download complete: %d rows across %d/%d series with data",
        total_rows,
        non_empty,
        len(writers),
    )
    if failures:
        logger.warning("%d series had at least one failed request window", len({f[0] for f in failures}))

    if not summary.empty:
        write_table(
            summary.sort_values("external_id"),
            config.TIMESERIES_DIR / "datapoints_summary.parquet",
            config.TIMESERIES_DIR / "datapoints_summary.csv",
            label="datapoints summary",
        )
    return summary


def resolve_window(
    start: str | None, end: str | None, lookback_days: int, latest_known: pd.Timestamp | None = None
) -> tuple[datetime, datetime]:
    """Turn CLI strings into a concrete UTC window.

    Anchoring the default window on the *latest datapoint actually present*
    rather than on wall-clock now is a small detail that matters: if ingestion
    has paused, "the last 60 days from now" can return nothing at all, which
    looks like a broken pipeline rather than a stalled feed.
    """
    end_dt = (
        pd.to_datetime(end, utc=True).to_pydatetime()
        if end
        else ((latest_known.to_pydatetime() if latest_known is not None else None)
              or datetime.now(timezone.utc))
    )
    if start:
        start_dt = pd.to_datetime(start, utc=True).to_pydatetime()
    else:
        start_dt = end_dt - timedelta(days=lookback_days)
    if start_dt >= end_dt:
        raise ValueError(f"--start ({start_dt}) must be before --end ({end_dt})")
    return start_dt, end_dt
