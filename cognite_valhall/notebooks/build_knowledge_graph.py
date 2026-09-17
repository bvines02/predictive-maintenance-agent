"""Build a NetworkX knowledge graph from the extracted Parquet dataset.

Run from the project root, after ``python -m src.main``:

    python notebooks/build_knowledge_graph.py

This is the bridge between the extraction pipeline and the reasoning layer. It
is intentionally short, because the extraction already did the hard part: every
table it wrote is keyed on ``external_id``, so the graph build is a few loops
over columns rather than an entity-resolution project.

WHY external_id IS THE NODE KEY
-------------------------------
A graph keyed on CDF's internal integer ids is a dead end: those ids are unique
only inside one CDF project, so the graph cannot be merged with SAP, the DCS,
or a P&ID annotation run. Engineering tags (``23-KA-9101``) are the identifier
every one of those systems already shares, so they are the join key.

GOING TO NEO4J INSTEAD
----------------------
The same frames load directly. Create the key constraint first, then MERGE:

    CREATE CONSTRAINT asset_xid IF NOT EXISTS
      FOR (a:Asset) REQUIRE a.external_id IS UNIQUE;

    LOAD CSV WITH HEADERS FROM 'file:///implicit_edges.csv' AS row
    MATCH (s {external_id: row.source_external_id})
    MATCH (t {external_id: row.target_external_id})
    CALL apoc.merge.relationship(s, row.relationship, {}, {}, t) YIELD rel
    RETURN count(rel);

Load the node tables first (one ``LOAD CSV`` per label, MERGE on
``external_id``), then the edge table. For anything above a few million edges,
use ``neo4j-admin database import`` rather than ``LOAD CSV``.
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"


def load(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def build_graph() -> nx.MultiDiGraph:
    assets = load(DATA / "assets" / "assets.parquet")
    series = load(DATA / "timeseries" / "timeseries_metadata.parquet")
    events = load(DATA / "events" / "events.parquet")
    files = load(DATA / "files" / "files_metadata.parquet")
    edges = load(DATA / "relationships" / "implicit_edges.parquet")
    explicit = load(DATA / "relationships" / "relationships.parquet")

    # MultiDiGraph, deliberately: two nodes can be connected by more than one
    # kind of relationship (an asset both contains and is measured by things),
    # and direction carries meaning (PART_OF is not symmetric).
    graph = nx.MultiDiGraph()

    for _, row in assets.iterrows():
        graph.add_node(
            row["external_id"], label="Asset", name=row["name"],
            description=row["description"], depth=int(row["depth"]),
            role=row["role"], subsystems=row["subsystem_labels"],
        )

    for _, row in series.iterrows():
        graph.add_node(
            row["external_id"], label="TimeSeries", name=row["name"],
            description=row["description"], unit=row["unit"],
            is_step=bool(row["is_step"]) if pd.notna(row["is_step"]) else False,
        )

    for _, row in events.iterrows():
        key = row["external_id"] if pd.notna(row["external_id"]) else str(row["event_id"])
        graph.add_node(
            key, label="Event", type=row["type"], subtype=row["subtype"],
            description=row["description"], start_time=str(row["start_time"]),
        )

    for _, row in files.iterrows():
        key = row["external_id"] if pd.notna(row["external_id"]) else str(row["file_id"])
        graph.add_node(
            key, label="Document", filename=row["filename"],
            mime_type=row["mime_type"],
            is_drawing=bool(row["likely_engineering_drawing"]),
        )

    for _, row in edges.iterrows():
        graph.add_edge(
            row["source_external_id"], row["target_external_id"],
            key=row["relationship"], relationship=row["relationship"],
            origin=row["origin"],
        )

    for _, row in explicit.iterrows():
        labels = json.loads(row["labels"]) if isinstance(row.get("labels"), str) else None
        rel = (labels[0] if labels else "RELATED_TO")
        graph.add_edge(
            row["source_external_id"], row["target_external_id"],
            key=rel, relationship=rel, confidence=row.get("confidence"),
            origin="relationships_api",
        )

    return graph


def context_for_tag(graph: nx.MultiDiGraph, tag: str) -> dict[str, object]:
    """The query a troubleshooting agent actually needs.

    Given an anomalous *tag*, return the equipment it measures, that
    equipment's place in the hierarchy, its sibling measurements, the events
    that touched it, and the drawings that document it. This one function is
    why the graph exists: it turns "pi:163661 looks wrong" into "the discharge
    temperature on the first-stage compressor looks wrong, here are the other
    signals on that machine, here is what happened to it recently, and here is
    the P&ID".
    """
    if tag not in graph:
        return {"error": f"{tag} not in graph"}

    # The asset this tag measures.
    assets = [
        t for _, t, d in graph.out_edges(tag, data=True)
        if d.get("relationship") == "MEASURES"
    ]
    asset = assets[0] if assets else None

    ancestors: list[str] = []
    node = asset
    while node is not None:
        parents = [
            t for _, t, d in graph.out_edges(node, data=True)
            if d.get("relationship") == "PART_OF"
        ]
        if not parents:
            break
        ancestors.append(parents[0])
        node = parents[0]

    sibling_tags = []
    events = []
    documents = []
    if asset is not None:
        for source, _, data in graph.in_edges(asset, data=True):
            rel = data.get("relationship")
            if rel == "MEASURES" and source != tag:
                sibling_tags.append(source)
            elif rel == "AFFECTS":
                events.append(source)
            elif rel == "DOCUMENTS":
                documents.append(source)

    return {
        "tag": tag,
        "tag_description": graph.nodes[tag].get("description"),
        "unit": graph.nodes[tag].get("unit"),
        "measures_asset": asset,
        "asset_description": graph.nodes[asset].get("description") if asset else None,
        "hierarchy_upward": ancestors,
        "sibling_measurements": sorted(sibling_tags),
        "related_events": sorted(events),
        "documents": sorted(documents),
    }


def main() -> None:
    graph = build_graph()
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    by_label: dict[str, int] = {}
    for _, data in graph.nodes(data=True):
        by_label[data.get("label", "Unknown")] = by_label.get(data.get("label", "Unknown"), 0) + 1
    print(f"Nodes by label: {by_label}")

    by_rel: dict[str, int] = {}
    for _, _, data in graph.edges(data=True):
        rel = data.get("relationship", "?")
        by_rel[rel] = by_rel.get(rel, 0) + 1
    print(f"Edges by relationship: {by_rel}")

    # Demonstrate the troubleshooting query on a real tag from the dataset.
    series = load(DATA / "timeseries" / "timeseries_metadata.parquet")
    if series.empty:
        print("\nNo time series found -- run `python -m src.main` first.")
        return
    tag = series["external_id"].iloc[0]
    print(f"\nTroubleshooting context for {tag}:")
    for key, value in context_for_tag(graph, tag).items():
        print(f"  {key:<22} {value}")


if __name__ == "__main__":
    main()
