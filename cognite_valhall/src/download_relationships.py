"""Probe CDF for explicit relationships and data-model edges.

DESIGN NOTE — three generations of "context" in CDF, and why we probe all three
------------------------------------------------------------------------------
Cognite has shipped three different ways to express that two things are
related, and which ones a given project uses depends on when and how it was
built:

1. **Asset hierarchy** (``parent_id``). Always present. Expresses exactly one
   relation: physical/functional containment.
2. **Relationships API** (``/relationships``). A typed, labelled edge between
   any two resources — asset↔asset, asset↔time series, file↔asset — with an
   optional confidence and validity interval. This is where "flows to",
   "is documented by" or "drives" would live.
3. **Data modelling / FDM** (instances and edges in spaces and views). The
   current graph-native approach, where edges are first-class typed instances.

Open Industrial Data is a long-lived demo project whose backbone is the classic
asset hierarchy. It may expose few or no Relationship rows. That is not a
failure — but it *is* something the pipeline must establish empirically rather
than assume, because the answer changes what the knowledge-graph build looks
like. So we probe each source, record what exists, and when nothing does we
write an explicit inventory of the implicit edges we can construct instead.

DESIGN NOTE — turning this into a NetworkX or Neo4j graph later
---------------------------------------------------------------
Everything this pipeline writes is already edge-shaped. The mapping is direct:

  nodes:
    Asset        <- assets.parquet             (key: external_id)
    TimeSeries   <- timeseries_metadata.parquet (key: external_id)
    Event        <- events.parquet              (key: event_id/external_id)
    Document     <- files_metadata.parquet      (key: file_id/external_id)

  edges:
    (Asset)-[:PART_OF]->(Asset)          from parent_external_id
    (TimeSeries)-[:MEASURES]->(Asset)    from timeseries.asset_external_id
    (Event)-[:AFFECTS]->(Asset)          from events.asset_external_ids (explode)
    (Document)-[:DOCUMENTS]->(Asset)     from files.asset_external_ids (explode)
    (Asset)-[:<label>]->(Any)            from relationships/*.parquet, if present

In NetworkX that is a handful of ``add_edge`` loops over those columns — see
``notebooks/`` for a worked example. For Neo4j the same frames become
``LOAD CSV`` or ``neo4j-admin import`` inputs, with ``external_id`` as the node
key and a uniqueness constraint on it. The reason ``external_id`` is the key
everywhere in this dataset, rather than the integer ``id``, is precisely this:
a graph keyed on project-internal integers cannot be merged with anything else,
while a graph keyed on engineering tags joins straight onto SAP, the DCS and
the P&ID annotations.

The genuinely hard part, which this pipeline does *not* do, is extracting
process topology (what flows into what) from the P&IDs themselves. That needs
drawing annotation — CDF has a P&ID contextualisation service, or you run your
own symbol/line detection — and it is what upgrades the graph from "things
belonging to things" to "a process you can reason about causally".
"""

from __future__ import annotations

import logging

import pandas as pd
from cognite.client import CogniteClient
from cognite.client.exceptions import CogniteAPIError

from . import config
from .storage import json_or_none, ms_to_utc, write_table

logger = logging.getLogger(__name__)

# Explicit schemas for the empty case. An empty DataFrame with no columns
# writes a Parquet file with no schema, which forces every downstream consumer
# to special-case "file exists but has no columns". Declaring the columns means
# a zero-row file is still a valid, self-describing table you can filter and
# concatenate without branching.
RELATIONSHIP_COLUMNS = [
    "relationship_external_id", "source_external_id", "source_type",
    "target_external_id", "target_type", "confidence", "labels",
    "data_set_id", "start_time", "end_time", "created_time",
]
DATA_MODEL_COLUMNS = ["kind", "identifier", "name", "description"]
IMPLICIT_EDGE_COLUMNS = [
    "source_external_id", "source_type", "relationship",
    "target_external_id", "target_type", "origin",
]


def probe_relationships(
    client: CogniteClient, asset_external_ids: list[str]
) -> pd.DataFrame:
    """Query the Relationships API for edges touching our assets, either end.

    Relationships are directional, so an asset can appear as ``source`` or as
    ``target``. Querying only one side is a classic half-empty-graph bug, so we
    query both and merge on ``external_id`` (which for relationships is the
    edge's own identifier).
    """
    if not asset_external_ids:
        return pd.DataFrame(columns=RELATIONSHIP_COLUMNS)

    edges: dict[str, object] = {}
    chunk_size = 100

    for label, kwarg in (("source", "source_external_ids"), ("target", "target_external_ids")):
        for offset in range(0, len(asset_external_ids), chunk_size):
            chunk = asset_external_ids[offset : offset + chunk_size]
            try:
                found = client.relationships.list(**{kwarg: chunk}, limit=config.LIST_ALL)
            except CogniteAPIError as exc:
                # A 403 here means the token lacks relationshipsAcl:READ, which
                # is a capability problem, not an empty-graph finding. We
                # distinguish the two in the log because the remedy differs.
                logger.warning(
                    "relationships.list(%s=...) failed (%s). If this is a 403, the "
                    "token lacks relationshipsAcl:READ.", kwarg, exc,
                )
                continue
            for rel in found:
                if rel.external_id:
                    edges[rel.external_id] = rel
            logger.debug("  %s-side probe: cumulative %d edges", label, len(edges))

    if not edges:
        logger.info("Relationships API returned no edges for the selected assets")
        return pd.DataFrame(columns=RELATIONSHIP_COLUMNS)

    rows = []
    for rel in edges.values():
        rows.append(
            {
                "relationship_external_id": rel.external_id,
                "source_external_id": rel.source_external_id,
                "source_type": rel.source_type,
                "target_external_id": rel.target_external_id,
                "target_type": rel.target_type,
                "confidence": rel.confidence,
                "labels": json_or_none(
                    [lbl.external_id for lbl in (rel.labels or [])] if rel.labels else None
                ),
                "data_set_id": rel.data_set_id,
                "start_time": ms_to_utc(rel.start_time),
                "end_time": ms_to_utc(rel.end_time),
                "created_time": ms_to_utc(rel.created_time),
            }
        )
    df = pd.DataFrame(rows)
    logger.info("Relationships API returned %d distinct edges", len(df))
    return df


def probe_data_models(client: CogniteClient) -> pd.DataFrame:
    """Check whether the project exposes data-model (FDM) spaces and views.

    We only inventory what exists; we do not traverse instances. Knowing
    *whether* a graph-native layer is present tells you which API a later
    knowledge-graph build should read from, and that is the question worth
    answering cheaply here.
    """
    rows = []
    try:
        spaces = client.data_modeling.spaces.list(limit=100)
        for space in spaces:
            rows.append({"kind": "space", "identifier": space.space,
                         "name": getattr(space, "name", None),
                         "description": getattr(space, "description", None)})
    except (CogniteAPIError, AttributeError) as exc:
        logger.info("Data-modelling spaces not readable (%s)", exc)
        return pd.DataFrame(columns=DATA_MODEL_COLUMNS)

    try:
        models = client.data_modeling.data_models.list(limit=100)
        for model in models:
            rows.append({"kind": "data_model", "identifier": f"{model.space}:{model.external_id}",
                         "name": getattr(model, "name", None),
                         "description": getattr(model, "description", None)})
    except (CogniteAPIError, AttributeError) as exc:
        logger.info("Data models not readable (%s)", exc)

    df = pd.DataFrame(rows, columns=DATA_MODEL_COLUMNS) if rows else pd.DataFrame(columns=DATA_MODEL_COLUMNS)
    logger.info("Data-modelling inventory: %d entries", len(df))
    return df


def _missing(value: object) -> bool:
    """True if a DataFrame cell holds no usable value.

    This exists because ``if row["parent_external_id"]:`` is WRONG on a pandas
    frame. A missing object-column value arrives as ``float("nan")``, which is
    truthy, so the naive check happily emits an edge pointing at the string
    "nan" -- a phantom node that silently corrupts the graph. (The root asset
    has no parent, so it hit exactly this path.)
    """
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() in {"", "nan", "None", "<NA>", "NaT"}


def build_implicit_edges(
    assets_df: pd.DataFrame,
    timeseries_df: pd.DataFrame,
    events_df: pd.DataFrame,
    files_df: pd.DataFrame,
) -> pd.DataFrame:
    """Derive the graph edges implied by the foreign keys we already hold.

    This is the deliverable that makes the knowledge graph buildable whether or
    not the project exposes explicit Relationships. Every edge here is a real
    link present in CDF — we are only reshaping foreign keys into an edge list.
    """
    import json

    edges: list[dict[str, object]] = []

    # (Asset)-[:PART_OF]->(Asset)
    if not assets_df.empty and "parent_external_id" in assets_df:
        for _, row in assets_df.iterrows():
            if not _missing(row.get("parent_external_id")) and not _missing(row.get("external_id")):
                edges.append({
                    "source_external_id": str(row["external_id"]),
                    "source_type": "asset",
                    "relationship": "PART_OF",
                    "target_external_id": str(row["parent_external_id"]),
                    "target_type": "asset",
                    "origin": "asset.parent_id",
                })

    # (TimeSeries)-[:MEASURES]->(Asset)
    if not timeseries_df.empty and "asset_external_id" in timeseries_df:
        for _, row in timeseries_df.iterrows():
            if not _missing(row.get("asset_external_id")) and not _missing(row.get("external_id")):
                edges.append({
                    "source_external_id": str(row["external_id"]),
                    "source_type": "timeseries",
                    "relationship": "MEASURES",
                    "target_external_id": str(row["asset_external_id"]),
                    "target_type": "asset",
                    "origin": "timeseries.asset_id",
                })

    def explode_links(df: pd.DataFrame, key: str, src_type: str, rel: str, origin: str) -> None:
        if df.empty or "asset_external_ids" not in df:
            return
        for _, row in df.iterrows():
            raw = row.get("asset_external_ids")
            if not raw:
                continue
            try:
                targets = json.loads(raw) if isinstance(raw, str) else list(raw)
            except (ValueError, TypeError):
                continue
            # Node keys are ALWAYS strings. Events and documents fall back to
            # their integer CDF id when they have no external_id, and mixing
            # int and str in one column both breaks the Arrow/Parquet write and
            # makes the column useless as a graph node key. We prefer
            # external_id (portable) and stringify the numeric fallback.
            source_key = row.get("external_id")
            if _missing(source_key):
                source_key = row.get(key)
            if _missing(source_key):
                continue
            for target in targets or []:
                if _missing(target):
                    continue
                edges.append({
                    "source_external_id": str(source_key),
                    "source_type": src_type,
                    "relationship": rel,
                    "target_external_id": str(target),
                    "target_type": "asset",
                    "origin": origin,
                })

    # (Event)-[:AFFECTS]->(Asset) and (Document)-[:DOCUMENTS]->(Asset)
    explode_links(events_df, "event_id", "event", "AFFECTS", "event.asset_ids")
    explode_links(files_df, "file_id", "document", "DOCUMENTS", "file.asset_ids")

    df = pd.DataFrame(edges, columns=IMPLICIT_EDGE_COLUMNS) if edges else pd.DataFrame(columns=IMPLICIT_EDGE_COLUMNS)
    if not df.empty:
        # Enforce the string contract on write, so a future edge source cannot
        # reintroduce a mixed-type column.
        for col in ("source_external_id", "target_external_id", "source_type",
                    "target_type", "relationship", "origin"):
            df[col] = df[col].astype("string")
        df = df.drop_duplicates().reset_index(drop=True)
    logger.info("Derived %d implicit edges from foreign keys", len(df))
    return df


def save_relationships(
    client: CogniteClient,
    asset_external_ids: list[str],
    assets_df: pd.DataFrame,
    timeseries_df: pd.DataFrame,
    events_df: pd.DataFrame,
    files_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Run the full relationship/context discovery stage."""
    explicit = probe_relationships(client, asset_external_ids)
    write_table(
        explicit,
        config.RELATIONSHIPS_DIR / "relationships.parquet",
        config.RELATIONSHIPS_DIR / "relationships.csv",
        label="explicit relationships",
    )

    models = probe_data_models(client)
    write_table(
        models,
        config.RELATIONSHIPS_DIR / "data_model_inventory.parquet",
        config.RELATIONSHIPS_DIR / "data_model_inventory.csv",
        label="data model inventory",
    )

    implicit = build_implicit_edges(assets_df, timeseries_df, events_df, files_df)
    write_table(
        implicit,
        config.RELATIONSHIPS_DIR / "implicit_edges.parquet",
        config.RELATIONSHIPS_DIR / "implicit_edges.csv",
        label="implicit edges",
    )

    print()
    print("=" * 78)
    print("RELATIONSHIP / CONTEXT DISCOVERY")
    print("=" * 78)
    print(f"  Explicit Relationship edges : {len(explicit)}")
    print(f"  Data-modelling entries      : {len(models)}")
    print(f"  Implicit edges from FKs     : {len(implicit)}")
    if not implicit.empty:
        for rel, count in implicit["relationship"].value_counts().items():
            print(f"      {rel:<12} {count}")
    if explicit.empty:
        print()
        print("  NOTE: this project exposes no explicit Relationship rows for these")
        print("  assets. The knowledge graph is still fully buildable from the")
        print("  foreign keys in implicit_edges.parquet (PART_OF / MEASURES /")
        print("  AFFECTS / DOCUMENTS). What is NOT available from CDF metadata is")
        print("  process topology -- which stream flows into which vessel. That")
        print("  requires annotating the P&IDs themselves.")
    print("=" * 78)
    print()

    return {"explicit": explicit, "data_models": models, "implicit": implicit}
