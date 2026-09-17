"""Discover the compressor asset and its surrounding process context.

DESIGN NOTE — asset IDs vs external IDs
---------------------------------------
Every CDF resource has two identifiers and confusing them is the single most
common source of broken industrial pipelines:

* ``id`` is an **internal**, CDF-generated 64-bit integer. It is unique only
  within one CDF project. If the same data is re-ingested into a new project,
  every ``id`` changes. Most *filter* parameters in the API take ids
  (``asset_ids``, ``asset_subtree_ids``), so you need them at query time.
* ``external_id`` is a **source-system** identifier supplied by whoever loaded
  the data — typically the engineering tag from SAP, the DCS, or the P&ID.
  It is stable across projects and re-ingests, and it is the thing a human
  recognises: ``23-KA-9101`` is a tag an operator can point to on a drawing.

The rule we follow throughout: **query with ids, persist with both, and join on
external_id.** Anything written to disk keeps both columns, because in six
months the parquet files may be loaded against a rebuilt project where the
integer ids no longer mean anything — but the tags still do.

DESIGN NOTE — why the asset hierarchy matters
---------------------------------------------
For a predictive-maintenance agent, the hierarchy is the reasoning substrate,
not decoration. It answers:

* **Localisation.** A vibration alarm on a bearing sensor is meaningless in
  isolation. The hierarchy says that sensor belongs to the compressor's
  drive-end bearing, which belongs to 23-KA-9101, which belongs to the
  first-stage train. That chain is what turns an anomaly into a fault
  hypothesis.
* **Correlation scope.** When you look for *other* signals that moved at the
  same time, the hierarchy tells you which signals are physically plausible
  neighbours. Correlating a compressor bearing with an unrelated water-injection
  pump is noise; correlating it with its own lube-oil supply pressure is signal.
* **Aggregation.** Maintenance history, documents and events are attached at
  different levels of the tree. Rolling up a subtree is how you collect "all
  the context for this machine".

Discovery strategy
------------------
We do not assume ``23-KA-9101`` is an external_id, or a name, or that any
particular subsystem word exists. We try identifier lookups in order of
specificity, then walk the hierarchy *structurally* (parents and children),
and only afterwards apply keyword labels to whatever we actually found.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from cognite.client import CogniteClient
from cognite.client.data_classes import Asset, AssetList

from . import config

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryResult:
    """Everything we learned about the target asset and its neighbourhood."""

    target: Asset
    ancestors: list[Asset] = field(default_factory=list)
    descendants: list[Asset] = field(default_factory=list)
    siblings: list[Asset] = field(default_factory=list)
    keyword_matches: dict[str, list[Asset]] = field(default_factory=dict)

    @property
    def selected(self) -> list[Asset]:
        """The de-duplicated set of assets this run will extract data for.

        Ordered deterministically (by external_id, falling back to id) so that
        repeated runs produce byte-comparable outputs — which matters when you
        want to diff two extractions.
        """
        by_id: dict[int, Asset] = {}
        groups = [[self.target], self.ancestors, self.descendants, self.siblings]
        for group in groups:
            for asset in group:
                if asset.id is not None:
                    by_id[asset.id] = asset
        for matches in self.keyword_matches.values():
            for asset in matches:
                if asset.id is not None:
                    by_id[asset.id] = asset
        return sorted(by_id.values(), key=lambda a: (a.external_id or "", a.id or 0))

    @property
    def selected_ids(self) -> list[int]:
        return [a.id for a in self.selected if a.id is not None]

    @property
    def role_of(self) -> dict[int, str]:
        """Map asset id -> structural role, for labelling the saved table."""
        roles: dict[int, str] = {}
        for asset in self.siblings:
            if asset.id is not None:
                roles[asset.id] = "sibling"
        for asset in self.descendants:
            if asset.id is not None:
                roles[asset.id] = "descendant"
        for asset in self.ancestors:
            if asset.id is not None:
                roles[asset.id] = "ancestor"
        if self.target.id is not None:
            roles[self.target.id] = "target"
        return roles


def find_target_asset(client: CogniteClient, identifier: str) -> Asset:
    """Locate the target asset without assuming which identifier was given.

    Order of attempts, most to least precise:

    1. ``retrieve(external_id=...)`` — exact external-id hit, one request.
    2. ``list(name=...)`` — exact-name filter (CDF names are not guaranteed
       unique, so we take the shallowest match, i.e. the one closest to the
       root, which is the containing equipment rather than a sub-part).
    3. ``search(query=...)`` — fuzzy full-text search, used only as a last
       resort and logged loudly, because fuzzy matching is exactly how you end
       up silently extracting the wrong machine.
    """
    logger.info("Locating asset %r ...", identifier)

    asset = client.assets.retrieve(external_id=identifier)
    if asset is not None:
        logger.info("  matched by external_id -> id=%s name=%r", asset.id, asset.name)
        return asset

    exact = client.assets.list(name=identifier, limit=config.LIST_ALL)
    if len(exact) == 1:
        asset = exact[0]
        logger.info("  matched by exact name -> id=%s external_id=%r", asset.id, asset.external_id)
        return asset
    if len(exact) > 1:
        # CDF asset names are not unique. There is no server-side "depth", so we
        # tie-break deterministically on the lowest id (earliest ingested) and
        # log every candidate loudly -- picking the wrong machine silently is
        # far worse than making the ambiguity impossible to miss.
        asset = min(exact, key=lambda a: a.id or 0)
        logger.warning(
            "  %d assets share the name %r; selecting id=%s external_id=%r. "
            "Candidates: %s",
            len(exact),
            identifier,
            asset.id,
            asset.external_id,
            [(a.id, a.external_id) for a in exact],
        )
        return asset

    fuzzy = client.assets.search(query=identifier, limit=25)
    if len(fuzzy) == 0:
        raise LookupError(
            f"No asset matching {identifier!r} in project {config.COGNITE_PROJECT!r}. "
            "Check the tag, or list root assets with: "
            "client.assets.list(root=True, limit=None)"
        )
    asset = fuzzy[0]
    logger.warning(
        "  NO exact match for %r. Falling back to best fuzzy hit: id=%s name=%r external_id=%r. "
        "Verify this is the intended machine before trusting the extraction.",
        identifier,
        asset.id,
        asset.name,
        asset.external_id,
    )
    return asset


def walk_ancestors(client: CogniteClient, asset: Asset, max_depth: int) -> list[Asset]:
    """Walk parent links to the root.

    CDF gives each asset a ``parent_id``, so ancestry is a pointer chase: one
    request per level. There is no bulk "get ancestors" endpoint, but the chain
    is short (a plant hierarchy is rarely deeper than ~8), so this is cheap.
    ``max_depth`` is a guard against a malformed hierarchy containing a cycle.
    """
    ancestors: list[Asset] = []
    current = asset
    for _ in range(max_depth):
        parent_id = current.parent_id
        if parent_id is None:
            break
        parent = client.assets.retrieve(id=parent_id)
        if parent is None:
            logger.warning("  parent_id=%s could not be retrieved; stopping walk", parent_id)
            break
        ancestors.append(parent)
        current = parent
    # Root-first ordering reads naturally when printed as a tree.
    ancestors.reverse()
    logger.info("  ancestors: %d", len(ancestors))
    return ancestors


def collect_descendants(client: CogniteClient, asset: Asset) -> list[Asset]:
    """Fetch the whole subtree below the target.

    ``retrieve_subtree`` is a server-side traversal: one call returns every
    descendant, and the SDK paginates through the result set for us. Doing this
    client-side (recursively listing children level by level) would be N+1
    requests and is the classic performance mistake here.
    """
    subtree: AssetList = client.assets.retrieve_subtree(id=asset.id)
    descendants = [a for a in subtree if a.id != asset.id]
    logger.info("  descendants: %d", len(descendants))
    return descendants


def collect_siblings(client: CogniteClient, asset: Asset) -> list[Asset]:
    """Fetch assets sharing the target's parent.

    Why bother? Because process troubleshooting is rarely confined to one tag.
    If the compressor's discharge temperature is rising, the answer may live in
    the aftercooler or the suction scrubber — which are the compressor's
    siblings in the train, not its children. Including one level of siblings is
    a cheap, bounded way to capture process-adjacent equipment without
    ballooning into the whole platform.
    """
    if asset.parent_id is None:
        logger.info("  siblings: 0 (target is a root asset)")
        return []
    siblings = client.assets.list(parent_ids=[asset.parent_id], limit=config.LIST_ALL)
    result = [a for a in siblings if a.id != asset.id]
    logger.info("  siblings: %d", len(result))
    return result


def classify_subsystems(assets: list[Asset]) -> dict[str, list[Asset]]:
    """Label discovered assets against the subsystem vocabulary.

    This is *post-hoc labelling of real data*, which is the important
    distinction. We never query CDF for "the gearbox" and report a gearbox if
    the query returned nothing. We collect whatever the hierarchy actually
    contains and then ask which of our subsystem words describe it. Keywords
    that match nothing are reported as absent, which is itself a useful finding
    about the dataset.
    """
    matches: dict[str, list[Asset]] = {}
    for subsystem, keywords in config.SUBSYSTEM_KEYWORDS.items():
        hits = []
        for asset in assets:
            haystack = " ".join(
                str(x).lower()
                for x in (asset.name, asset.description, asset.external_id)
                if x
            )
            if any(kw.lower() in haystack for kw in keywords):
                hits.append(asset)
        matches[subsystem] = hits
    return matches


def discover(client: CogniteClient, identifier: str) -> DiscoveryResult:
    """Run the full discovery sequence for one target asset."""
    target = find_target_asset(client, identifier)

    ancestors = walk_ancestors(client, target, config.ANCESTOR_DEPTH)
    descendants = collect_descendants(client, target)
    siblings = collect_siblings(client, target) if config.INCLUDE_SIBLINGS else []

    result = DiscoveryResult(
        target=target,
        ancestors=ancestors,
        descendants=descendants,
        siblings=siblings,
    )
    # Classify over the structural selection, so labels describe what we hold.
    result.keyword_matches = classify_subsystems(result.selected)
    return result


def print_hierarchy(result: DiscoveryResult) -> None:
    """Print a human-inspectable tree of what was discovered.

    Deliberately printed to stdout rather than the logger: this is a report for
    a person to read, not a log line for a machine to parse.
    """
    target = result.target
    print()
    print("=" * 78)
    print(f"DISCOVERED HIERARCHY for {target.name!r} (external_id={target.external_id!r})")
    print("=" * 78)

    indent = 0
    for ancestor in result.ancestors:
        print(f"{'  ' * indent}└─ {_label(ancestor)}")
        indent += 1

    print(f"{'  ' * indent}└─ ** {_label(target)} **   <-- TARGET")
    target_depth = indent + 1

    # Render descendants as a real tree using parent_id links.
    children_by_parent: dict[int | None, list[Asset]] = {}
    for asset in result.descendants:
        children_by_parent.setdefault(asset.parent_id, []).append(asset)

    def render(parent_id: int | None, depth: int) -> None:
        for child in sorted(
            children_by_parent.get(parent_id, []), key=lambda a: (a.name or "")
        ):
            print(f"{'  ' * depth}└─ {_label(child)}")
            render(child.id, depth + 1)

    render(target.id, target_depth)

    if result.siblings:
        parent_name = result.ancestors[-1].name if result.ancestors else "(root)"
        print()
        print(f"PROCESS-ADJACENT ASSETS (siblings under {parent_name!r}):")
        for sibling in sorted(result.siblings, key=lambda a: (a.name or "")):
            print(f"  ├─ {_label(sibling)}")

    print()
    print("SUBSYSTEM KEYWORD COVERAGE (labels applied to discovered assets only):")
    for subsystem, hits in result.keyword_matches.items():
        if hits:
            names = ", ".join(sorted({h.name or str(h.id) for h in hits})[:6])
            more = f" (+{len(hits) - 6} more)" if len(hits) > 6 else ""
            print(f"  [{len(hits):3d}] {subsystem:<16} {names}{more}")
        else:
            print(f"  [  0] {subsystem:<16} -- no asset in this subtree matches these terms")

    print()
    print(f"TOTAL SELECTED ASSETS: {len(result.selected)}")
    print("=" * 78)
    print()


def _label(asset: Asset) -> str:
    desc = (asset.description or "").strip()
    if len(desc) > 58:
        desc = desc[:55] + "..."
    suffix = f"  |  {desc}" if desc else ""
    return f"{asset.name}  (id={asset.id}, xid={asset.external_id}){suffix}"
