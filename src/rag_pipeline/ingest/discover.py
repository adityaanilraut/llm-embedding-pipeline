from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

from rag_pipeline.ingest.parsers import SUPPORTED_EXTS, parse
from rag_pipeline.schema import Chunk, Document
from rag_pipeline.state import StateStore


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class DiscoveryResult:
    new_or_changed: list[Document]  # need re-parse + re-embed
    deleted_doc_ids: list[str]  # need tombstone
    unchanged_count: int


def discover(raw_dir: Path, state: StateStore) -> DiscoveryResult:
    """Walk raw_dir, diff against state, return docs needing work + tombstones."""
    seen_paths: set[str] = set()
    new_or_changed: list[Document] = []

    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        path_str = str(path.resolve())
        seen_paths.add(path_str)

        mtime = path.stat().st_mtime
        prior = state.known_file(path_str)
        doc_id = file_sha256(path)

        if prior is not None and prior[0] == doc_id:
            continue  # unchanged

        stype, text = parse(path)
        doc = Document(
            doc_id=doc_id,
            source_path=path_str,
            source_type=stype,
            text=text,
            metadata={"filename": path.name, "mtime": mtime},
        )
        new_or_changed.append(doc)
        state.upsert_file(path_str, doc_id, mtime, time.time())

    deleted_doc_ids: list[str] = []
    for stale in state.all_paths() - seen_paths:
        gone = state.delete_file(stale)
        if gone:
            deleted_doc_ids.append(gone)

    unchanged = len(seen_paths) - len(new_or_changed)
    return DiscoveryResult(
        new_or_changed=new_or_changed,
        deleted_doc_ids=deleted_doc_ids,
        unchanged_count=unchanged,
    )


def diff_chunks(state: StateStore, doc_id: str, chunks: list[Chunk]) -> list[Chunk]:
    """Return only the chunks whose content hash changed vs state. Updates state."""
    prior = state.known_chunks_for_doc(doc_id)
    current = {c.chunk_id: c.content_hash for c in chunks}
    to_embed = [c for c in chunks if prior.get(c.chunk_id) != c.content_hash]
    state.replace_chunks(doc_id, current)
    return to_embed
