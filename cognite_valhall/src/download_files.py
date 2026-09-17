"""Find, catalogue and selectively download engineering documents.

DESIGN NOTE — catalogue first, download second
----------------------------------------------
The public dataset contains a large document corpus. Downloading it wholesale
would be slow, wasteful, and would bury the handful of drawings that actually
describe our compressor. So this module is strictly two-phase:

  Phase 1 (cheap, metadata only): find candidates by asset link and by search
  term, score them, write the full catalogue to Parquet, and *print* what we
  believe the P&IDs are.

  Phase 2 (expensive, bytes): download only the ranked shortlist, capped by
  ``--max-files``.

This is the same principle as the datapoint density probe: establish the shape
of what you are about to fetch before fetching it. In a production system this
phase separation is what lets you review and approve an extraction scope before
it runs against a metered egress bill.

DESIGN NOTE — why asset links alone are not enough
--------------------------------------------------
Two complementary discovery routes, because document-to-asset linkage is the
least complete link in most CDF projects:

* ``files.list(asset_ids=...)`` — documents explicitly linked to our equipment.
  High precision, often low recall: linking is frequently done by an ML
  "P&ID contextualisation" job that only annotated a subset of drawings.
* ``files.search(name=...)`` plus tag-substring matching — catches the drawing
  named ``23-KA-9101_PID_REV3.pdf`` that nobody ever linked. Lower precision,
  so every hit records *why* it matched and is scored accordingly.

We keep the provenance of each match in the catalogue (``match_reasons``), which
means a later knowledge-graph build can weight an explicit asset link higher
than a filename coincidence.
"""

from __future__ import annotations

import logging
import re

import pandas as pd
from cognite.client import CogniteClient
from cognite.client.data_classes import FileMetadata
from cognite.client.exceptions import CogniteAPIError

from . import config
from .storage import json_or_none, ms_to_utc, write_table

logger = logging.getLogger(__name__)


def _tag_variants(tag: str) -> set[str]:
    """Generate the ways an engineering tag is spelled in filenames.

    ``23-KA-9101`` shows up as ``23KA9101``, ``23_KA_9101``, ``23-ka-9101`` and
    so on, because filenames pass through many systems with different
    conventions. Normalising to a compact form catches all of them.
    """
    tag = tag.strip()
    compact = re.sub(r"[^A-Za-z0-9]", "", tag)
    return {v for v in {tag, tag.upper(), tag.lower(), compact, compact.upper(), compact.lower()} if v}


def discover_files(
    client: CogniteClient,
    asset_ids: list[int],
    target_tag: str,
    extra_tags: list[str] | None = None,
) -> tuple[dict[int, FileMetadata], dict[int, list[str]]]:
    """Find candidate documents, recording why each one matched.

    Returns ``(files_by_id, reasons_by_id)``.
    """
    files: dict[int, FileMetadata] = {}
    reasons: dict[int, list[str]] = {}

    def record(file: FileMetadata, reason: str) -> None:
        if file.id is None:
            return
        files[file.id] = file
        reasons.setdefault(file.id, [])
        if reason not in reasons[file.id]:
            reasons[file.id].append(reason)

    # --- route 1: explicit asset links -----------------------------------
    chunk_size = 100
    for offset in range(0, len(asset_ids), chunk_size):
        chunk = asset_ids[offset : offset + chunk_size]
        try:
            for file in client.files.list(asset_ids=chunk, limit=config.LIST_ALL):
                record(file, "asset_link")
        except CogniteAPIError as exc:
            logger.warning("files.list(asset_ids=...) failed for a chunk: %s", exc)

    logger.info("Files linked to selected assets: %d", len(files))

    # --- route 2: name search on the tag and its spelling variants --------
    search_tags = [target_tag] + list(extra_tags or [])
    for tag in search_tags:
        for variant in sorted(_tag_variants(tag)):
            try:
                hits = client.files.search(name=variant, limit=100)
            except CogniteAPIError as exc:
                logger.warning("files.search(name=%r) failed: %s", variant, exc)
                continue
            for file in hits:
                record(file, f"name_search:{variant}")

    # --- route 3: engineering-drawing vocabulary --------------------------
    # Broad by design, so it is only ever used to *flag* documents; the
    # relevance score below decides what is actually downloaded.
    for term in config.PID_SEARCH_TERMS:
        try:
            hits = client.files.search(name=term, limit=100)
        except CogniteAPIError as exc:
            logger.warning("files.search(name=%r) failed: %s", term, exc)
            continue
        for file in hits:
            record(file, f"drawing_term:{term}")

    logger.info("Total candidate documents after all routes: %d", len(files))
    return files, reasons


def _looks_like_drawing(file: FileMetadata) -> bool:
    """Heuristic: does this file look like an engineering drawing?"""
    haystack = " ".join(
        str(x).lower()
        for x in (file.name, file.external_id, file.directory, json_or_none(file.metadata))
        if x
    )
    if any(hint in haystack for hint in config.PID_FILENAME_HINTS):
        return True
    # Vector formats and DWG are drawing-native; a PDF may be anything.
    return (file.mime_type or "").lower() in {
        "application/dwg", "image/vnd.dwg", "application/dxf", "image/svg+xml",
    }


def files_to_dataframe(
    files: dict[int, FileMetadata],
    reasons: dict[int, list[str]],
    asset_lookup: dict[int, str],
    selected_asset_ids: set[int],
    target_tag: str,
) -> pd.DataFrame:
    """Build the document catalogue, with a relevance score per document.

    The score is a small, explicit, auditable ranking rather than a learned
    model, because at this scale an engineer needs to be able to read the rule
    and disagree with it:

      +5  explicitly linked to one of our selected assets
      +4  the target tag appears in the filename/metadata
      +2  looks like an engineering drawing (P&ID/PFD/ISO/DWG hints)
      +1  matched a generic drawing search term

    Only positively-scored documents are download candidates.
    """
    rows = []
    variants = _tag_variants(target_tag)

    for file_id, file in files.items():
        file_reasons = reasons.get(file_id, [])
        asset_ids = list(file.asset_ids or [])
        linked_selected = [a for a in asset_ids if a in selected_asset_ids]

        haystack = " ".join(
            str(x).lower()
            for x in (file.name, file.external_id, file.directory, json_or_none(file.metadata))
            if x
        )
        tag_in_name = any(v.lower() in haystack for v in variants)
        is_drawing = _looks_like_drawing(file)

        score = 0
        if linked_selected:
            score += 5
        if tag_in_name:
            score += 4
        if is_drawing:
            score += 2
        if any(r.startswith("drawing_term:") for r in file_reasons):
            score += 1

        rows.append(
            {
                "file_id": file.id,
                "external_id": file.external_id,
                "filename": file.name,
                "mime_type": file.mime_type,
                "source": file.source,
                "directory": file.directory,
                "uploaded": file.uploaded,
                "asset_ids": json_or_none(asset_ids),
                "asset_external_ids": json_or_none(
                    [asset_lookup[a] for a in asset_ids if a in asset_lookup]
                ),
                "n_linked_selected_assets": len(linked_selected),
                "data_set_id": file.data_set_id,
                "labels": json_or_none(
                    [lbl.external_id for lbl in (file.labels or [])] if file.labels else None
                ),
                "metadata": json_or_none(file.metadata),
                "match_reasons": ",".join(file_reasons),
                "search_terms": ",".join(
                    sorted({r.split(":", 1)[1] for r in file_reasons if ":" in r})
                ),
                "tag_in_filename": tag_in_name,
                "likely_engineering_drawing": is_drawing,
                "relevance_score": score,
                "source_created_time": ms_to_utc(file.source_created_time),
                "source_modified_time": ms_to_utc(file.source_modified_time),
                "uploaded_time": ms_to_utc(file.uploaded_time),
                "created_time": ms_to_utc(file.created_time),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["relevance_score", "n_linked_selected_assets", "filename"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
    return df


def print_file_candidates(df: pd.DataFrame, max_files: int) -> pd.DataFrame:
    """Print the shortlist and return the rows we intend to download."""
    print()
    print("=" * 78)
    print("ENGINEERING DOCUMENT CANDIDATES")
    print("=" * 78)
    if df.empty:
        print("  No candidate documents found.")
        print("=" * 78)
        print()
        return df

    shortlist = df[df["relevance_score"] > 0].head(max_files)
    drawings = shortlist[shortlist["likely_engineering_drawing"]]

    print(f"  Catalogued candidates      : {len(df)}")
    print(f"  Positively scored          : {int((df['relevance_score'] > 0).sum())}")
    print(f"  Likely drawings / P&IDs    : {int(df['likely_engineering_drawing'].sum())}")
    print(f"  Linked to selected assets  : {int((df['n_linked_selected_assets'] > 0).sum())}")
    print(f"  Will download (cap {max_files:>3d})     : {len(shortlist)}")
    print()
    print("  LIKELY P&IDs / ENGINEERING DRAWINGS:")
    if drawings.empty:
        print("    (none of the shortlisted documents match drawing heuristics)")
    else:
        for _, row in drawings.head(30).iterrows():
            print(
                f"    [score {row['relevance_score']:>2}] {row['filename']}"
                f"  ({row['mime_type']})  via {row['match_reasons']}"
            )
    print()
    print("  OTHER SHORTLISTED DOCUMENTS:")
    others = shortlist[~shortlist["likely_engineering_drawing"]]
    if others.empty:
        print("    (none)")
    else:
        for _, row in others.head(20).iterrows():
            print(f"    [score {row['relevance_score']:>2}] {row['filename']}  ({row['mime_type']})")
    print("=" * 78)
    print()
    return shortlist


def _record_path(path) -> str:
    """Render a saved path relative to the project root when possible.

    ``Path.relative_to`` raises when the target is not under the given root,
    and the output directories ARE overridable via environment variables (and
    are redirected wholesale under test). Falling back to the absolute path
    keeps the catalogue correct instead of crashing a completed download at the
    bookkeeping step.
    """
    from pathlib import Path

    path = Path(path)
    try:
        return str(path.relative_to(config.PROJECT_ROOT))
    except ValueError:
        return str(path)


def download_documents(client: CogniteClient, shortlist: pd.DataFrame) -> pd.DataFrame:
    """Download the shortlisted documents, one at a time with per-file errors.

    We use ``download_to_path`` per file rather than the bulk ``download``
    helper, deliberately. Bulk download is faster, but a single unavailable
    file raises and you lose the batch; per-file lets us record exactly which
    documents landed and which failed, which is the information you need when
    reconciling a dataset later. For the tens-of-files scale here the
    throughput difference is irrelevant.
    """
    if shortlist.empty:
        logger.info("No documents to download")
        return shortlist.assign(downloaded=False, local_path=None) if not shortlist.empty else shortlist

    config.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for i, (_, row) in enumerate(shortlist.iterrows(), start=1):
        name = str(row["filename"] or f"file_{row['file_id']}")
        from .storage import safe_filename

        local = config.DOCUMENTS_DIR / safe_filename(name)
        if local.exists() and local.stat().st_size > 0:
            logger.info("  [%d/%d] already present: %s", i, len(shortlist), local.name)
            results.append({"file_id": row["file_id"], "downloaded": True,
                            "local_path": _record_path(local), "error": None})
            continue
        try:
            logger.info("  [%d/%d] downloading %s ...", i, len(shortlist), name)
            client.files.download_to_path(path=local, id=int(row["file_id"]))
            results.append({"file_id": row["file_id"], "downloaded": True,
                            "local_path": _record_path(local), "error": None})
        except (CogniteAPIError, OSError) as exc:
            logger.error("  failed to download %s: %s", name, exc)
            results.append({"file_id": row["file_id"], "downloaded": False,
                            "local_path": None, "error": str(exc)})

    outcome = pd.DataFrame(results)
    merged = shortlist.merge(outcome, on="file_id", how="left")
    n_ok = int(merged["downloaded"].fillna(False).sum())
    logger.info("Downloaded %d/%d documents", n_ok, len(merged))
    return merged


def save_files_catalogue(
    client: CogniteClient,
    asset_ids: list[int],
    asset_lookup: dict[int, str],
    target_tag: str,
    extra_tags: list[str] | None = None,
    max_files: int = config.MAX_FILES_TO_DOWNLOAD,
    skip_download: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full files stage. Returns ``(catalogue, downloaded)``."""
    files, reasons = discover_files(client, asset_ids, target_tag, extra_tags)
    df = files_to_dataframe(files, reasons, asset_lookup, set(asset_ids), target_tag)
    write_table(
        df,
        config.FILES_DIR / "files_metadata.parquet",
        config.FILES_DIR / "files_metadata.csv",
        label="files catalogue",
    )
    shortlist = print_file_candidates(df, max_files)
    if skip_download or shortlist.empty:
        if skip_download:
            logger.info("--skip-files set: catalogue written, no bytes downloaded")
        return df, shortlist.head(0)
    downloaded = download_documents(client, shortlist)
    write_table(
        downloaded,
        config.FILES_DIR / "files_downloaded.parquet",
        config.FILES_DIR / "files_downloaded.csv",
        label="downloaded documents",
    )
    return df, downloaded
