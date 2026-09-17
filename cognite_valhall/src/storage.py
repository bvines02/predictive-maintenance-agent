"""Shared persistence helpers.

DESIGN NOTE — why Parquet is the primary format and CSV is only a courtesy
-------------------------------------------------------------------------
Both are written for every table, but they serve different purposes: Parquet is
the dataset, CSV is a debugging aid you open in a spreadsheet.

For telemetry specifically, Parquet wins decisively:

* **Typed schema.** A timestamp stays a timestamp and a float stays a float.
  CSV has exactly one type — text — so every reload re-parses and re-guesses.
  That is how a tag whose values happen to look like ``1-2`` silently becomes a
  date, and how an ``int64`` id above 2^53 gets mangled by a float round-trip.
* **Size and speed.** Columnar storage plus per-column compression typically
  gives 5-20x smaller files than CSV for sensor data, because a column of
  slowly-varying floats compresses far better than interleaved rows. Reads are
  faster again because you can load two columns out of twenty.
* **Predicate and column pushdown.** Engines skip row groups that cannot match
  a filter using per-row-group min/max statistics. Asking for one week out of
  two years of history touches a fraction of the file.
* **Null handling.** Parquet distinguishes "missing" from "empty string" from
  "zero". In CSV all three are often an empty field — and in condition
  monitoring, the difference between a sensor reading 0.0 and a sensor not
  reporting is the difference between a tripped machine and a dead transmitter.

We keep CSV because being able to eyeball a file with ``head`` or Excel during
development is genuinely valuable, and these metadata tables are small. We do
*not* write CSV for datapoints, where the size penalty is real.

DESIGN NOTE — why metadata becomes a JSON string
------------------------------------------------
CDF resources carry a free-form ``metadata`` dict (``dict[str, str]``). Parquet
can represent that as a MAP type, but support across pandas/Polars/DuckDB/Spark
versions is uneven, and CSV cannot represent it at all. Serialising to a single
JSON text column is lossless, portable, and trivially re-inflated with
``json.loads``. The alternative — exploding metadata into one column per key —
produces a wide, sparse, unstable schema that changes whenever the source
system adds a field.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def json_or_none(value: Any) -> str | None:
    """Serialise a dict/list to compact JSON, preserving None as null."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def ms_to_utc(value: Any) -> pd.Timestamp | None:
    """Convert CDF's epoch-milliseconds integers to tz-aware UTC timestamps.

    CDF stores every timestamp as milliseconds since the Unix epoch, UTC. We
    convert on write and always keep the timezone attached. Naive timestamps are
    a liability in industrial data: the moment one analyst assumes local Bergen
    time and another assumes UTC, your correlation between a vibration spike and
    a maintenance record is off by one or two hours and the conclusion is wrong.
    """
    if value is None or pd.isna(value):
        return None
    return pd.to_datetime(int(value), unit="ms", utc=True)


def write_table(
    df: pd.DataFrame,
    parquet_path: Path,
    csv_path: Path | None = None,
    label: str = "table",
) -> None:
    """Write a DataFrame to Parquet (and optionally CSV), logging the result."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        # Still write the file. An empty-but-present file with the right schema
        # is a much better signal to downstream code (and to a human reading
        # the directory) than a missing file, which is ambiguous between "no
        # data" and "this stage never ran".
        logger.warning("%s: no rows — writing empty file with schema only", label)

    df.to_parquet(parquet_path, index=False)
    size_kb = parquet_path.stat().st_size / 1024
    logger.info("%s: %d rows -> %s (%.1f KB)", label, len(df), parquet_path.name, size_kb)

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        logger.info("%s: also wrote %s for inspection", label, csv_path.name)


def safe_filename(value: str, max_length: int = 150) -> str:
    """Make a string safe to use as a filename on any common filesystem.

    Industrial tags contain characters that are illegal or awkward in paths:
    ``VAL_23-KA-9101_ASP:VALUE`` has a colon (illegal on Windows, awkward
    everywhere). We substitute rather than strip so that distinct tags cannot
    collide into the same filename, and we keep the original external_id inside
    the file as a column so the mapping is never lost.
    """
    replacements = {":": "__", "/": "_", "\\": "_", " ": "_", "*": "_", "?": "_",
                    "<": "_", ">": "_", "|": "_", '"': "_"}
    out = value
    for bad, good in replacements.items():
        out = out.replace(bad, good)
    out = out.strip(". ")
    if len(out) > max_length:
        # Keep a hash suffix so truncation cannot cause two tags to collide.
        import hashlib

        digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
        out = out[: max_length - 9] + "_" + digest
    return out or "unnamed"
