"""Persist the selected asset hierarchy to disk.

The output table is intentionally *denormalised*: alongside each asset's own
fields we store its parent external_id and its full root-to-leaf path. The
hierarchy in CDF is stored as parent pointers, which is the right shape for an
API but an awkward shape for analysis — answering "give me everything under the
first-stage train" from parent pointers alone requires a recursive join.

Materialising ``path_external_ids`` turns those recursive questions into string
prefix matches, which pandas, Polars, DuckDB and SQL all do efficiently. This is
the standard closure-table/materialised-path trade-off: we accept a little
redundancy on write in exchange for cheap ancestor queries on read. It is also
exactly the column a knowledge-graph builder wants later, since each path is a
ready-made chain of ``PART_OF`` edges.
"""

from __future__ import annotations

import logging

import pandas as pd
from cognite.client.data_classes import Asset

from . import config
from .discover_assets import DiscoveryResult
from .storage import json_or_none, ms_to_utc, write_table

logger = logging.getLogger(__name__)


def assets_to_dataframe(result: DiscoveryResult) -> pd.DataFrame:
    """Flatten the discovery result into one row per asset."""
    selected = result.selected
    roles = result.role_of

    # Index everything we hold so we can resolve parent links locally instead of
    # issuing one API call per asset.
    by_id: dict[int, Asset] = {a.id: a for a in selected if a.id is not None}

    # Reverse the keyword classification into asset_id -> [subsystem labels].
    labels: dict[int, list[str]] = {}
    for subsystem, hits in result.keyword_matches.items():
        for asset in hits:
            if asset.id is not None:
                labels.setdefault(asset.id, []).append(subsystem)

    def ancestor_chain(asset: Asset) -> list[Asset]:
        """Walk parent pointers upward using only assets already in memory."""
        chain: list[Asset] = []
        current = asset
        seen: set[int] = set()
        while current.parent_id is not None and current.parent_id in by_id:
            if current.parent_id in seen:  # defensive: malformed cyclic hierarchy
                break
            seen.add(current.parent_id)
            current = by_id[current.parent_id]
            chain.append(current)
        chain.reverse()
        return chain

    rows = []
    for asset in selected:
        chain = ancestor_chain(asset)
        parent = by_id.get(asset.parent_id) if asset.parent_id is not None else None
        rows.append(
            {
                "asset_id": asset.id,
                "external_id": asset.external_id,
                "name": asset.name,
                "description": asset.description,
                "parent_id": asset.parent_id,
                "parent_external_id": parent.external_id if parent else asset.parent_external_id,
                "root_id": asset.root_id,
                "depth": len(chain),
                # Materialised path: root -> ... -> this asset.
                "path_external_ids": " / ".join(
                    [a.external_id or str(a.id) for a in chain]
                    + [asset.external_id or str(asset.id)]
                ),
                "path_names": " / ".join([a.name or "" for a in chain] + [asset.name or ""]),
                "role": roles.get(asset.id or -1, "context"),
                "subsystem_labels": ",".join(sorted(labels.get(asset.id or -1, []))),
                "labels": json_or_none(
                    [lbl.external_id for lbl in (asset.labels or [])] if asset.labels else None
                ),
                "source": asset.source,
                "data_set_id": asset.data_set_id,
                "metadata": json_or_none(asset.metadata),
                "created_time": ms_to_utc(asset.created_time),
                "last_updated_time": ms_to_utc(asset.last_updated_time),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        # Stable ordering: hierarchy order, so the file reads like the tree.
        df = df.sort_values(["depth", "path_external_ids"]).reset_index(drop=True)
    return df


def save_assets(result: DiscoveryResult) -> pd.DataFrame:
    """Write ``data/assets/assets.parquet`` (and the CSV twin)."""
    df = assets_to_dataframe(result)
    write_table(
        df,
        config.ASSETS_DIR / "assets.parquet",
        config.ASSETS_DIR / "assets.csv",
        label="assets",
    )
    return df
