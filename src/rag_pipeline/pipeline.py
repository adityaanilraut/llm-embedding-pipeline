"""End-to-end ingestion pipeline callable from CLI, Airflow, or tests."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from rag_pipeline.config import Settings, get_settings
from rag_pipeline.embed.ray_embed import embed_chunks_distributed
from rag_pipeline.ingest.chunker import chunk_document
from rag_pipeline.ingest.discover import diff_chunks, discover
from rag_pipeline.io_parquet import (
    read_all_chunks,
    read_all_embeddings,
    write_chunks,
    write_embeddings,
)
from rag_pipeline.retrieve.bm25 import BM25Index
from rag_pipeline.schema import Chunk, EmbeddedChunk
from rag_pipeline.state import StateStore
from rag_pipeline.store.factory import get_store

log = logging.getLogger(__name__)


@dataclass
class RunReport:
    docs_changed: int
    docs_unchanged: int
    docs_deleted: int
    chunks_embedded: int
    chunks_skipped: int
    elapsed_s: float


def run_pipeline(
    settings: Settings | None = None, backends: tuple[str, ...] = ("faiss",)
) -> RunReport:
    settings = settings or get_settings()
    settings.chunks_dir.mkdir(parents=True, exist_ok=True)
    settings.embeddings_dir.mkdir(parents=True, exist_ok=True)
    state = StateStore(settings.state_db)
    t0 = time.time()

    # 1. Discover changed / deleted files
    disc = discover(settings.raw_dir, state)
    log.info(
        "discover: %d changed, %d unchanged, %d deleted",
        len(disc.new_or_changed),
        disc.unchanged_count,
        len(disc.deleted_doc_ids),
    )

    # 2. Chunk changed docs; diff against per-chunk hashes
    all_chunks: list[Chunk] = []
    chunks_to_embed: list[Chunk] = []
    chunks_skipped = 0
    for doc in disc.new_or_changed:
        doc_chunks = chunk_document(doc, settings.chunk_size, settings.chunk_overlap)
        all_chunks.extend(doc_chunks)
        deltas = diff_chunks(state, doc.doc_id, doc_chunks)
        chunks_to_embed.extend(deltas)
        chunks_skipped += len(doc_chunks) - len(deltas)

    if all_chunks:
        write_chunks(all_chunks, settings.chunks_dir)

    # 3. Embed
    embedded: list[EmbeddedChunk] = []
    if chunks_to_embed:
        embedded = embed_chunks_distributed(
            chunks_to_embed,
            model_name=settings.embed_model,
            batch_size=settings.embed_batch_size,
            num_workers=settings.ray_num_workers,
        )
        write_embeddings(embedded, settings.embeddings_dir)

    # 4. Index — push to each requested backend
    for backend in backends:
        store = get_store(backend, settings)
        # tombstone deleted
        if disc.deleted_doc_ids:
            store.delete_by_doc(disc.deleted_doc_ids)
        # upsert new embeddings
        if embedded:
            import numpy as np

            vecs = np.asarray([e.embedding for e in embedded], dtype="float32")
            payloads = [
                {
                    "doc_id": e.doc_id,
                    "chunk_index": e.index,
                    "source_path": e.source_path,
                    "source_type": e.source_type,
                    "text": e.text,
                }
                for e in embedded
            ]
            store.upsert([e.chunk_id for e in embedded], vecs, payloads)
        store.persist()

    # 5. Rebuild BM25 from the union of all chunks on disk (sparse index is cheap)
    chunks_table = read_all_chunks(settings.chunks_dir)
    if chunks_table.num_rows > 0:
        cids = chunks_table.column("chunk_id").to_pylist()
        texts = chunks_table.column("text").to_pylist()
        # de-duplicate by chunk_id (later writes win)
        latest: dict[str, str] = {}
        latest_path: dict[str, str] = {}
        latest_stype: dict[str, str] = {}
        latest_doc: dict[str, str] = {}
        paths = chunks_table.column("source_path").to_pylist()
        stypes = chunks_table.column("source_type").to_pylist()
        dids = chunks_table.column("doc_id").to_pylist()
        for cid, t, p, st, d in zip(cids, texts, paths, stypes, dids, strict=False):
            latest[cid] = t
            latest_path[cid] = p
            latest_stype[cid] = st
            latest_doc[cid] = d

        bm25 = BM25Index(settings.bm25_dir / "bm25.pkl")
        bm25.build(
            chunk_ids=list(latest.keys()),
            texts=list(latest.values()),
            payloads=[
                {
                    "doc_id": latest_doc[cid],
                    "source_path": latest_path[cid],
                    "source_type": latest_stype[cid],
                    "text": latest[cid],
                }
                for cid in latest
            ],
        )
        bm25.persist()

    return RunReport(
        docs_changed=len(disc.new_or_changed),
        docs_unchanged=disc.unchanged_count,
        docs_deleted=len(disc.deleted_doc_ids),
        chunks_embedded=len(embedded),
        chunks_skipped=chunks_skipped,
        elapsed_s=time.time() - t0,
    )


# Stage entry points for Airflow ---------------------------------------------


def stage_discover_and_chunk(settings: Settings | None = None) -> dict:
    """Discover + chunk only. Returns a JSON-serializable summary for XCom."""
    settings = settings or get_settings()
    state = StateStore(settings.state_db)
    settings.chunks_dir.mkdir(parents=True, exist_ok=True)

    disc = discover(settings.raw_dir, state)
    all_chunks: list[Chunk] = []
    to_embed: list[Chunk] = []
    for doc in disc.new_or_changed:
        cs = chunk_document(doc, settings.chunk_size, settings.chunk_overlap)
        all_chunks.extend(cs)
        to_embed.extend(diff_chunks(state, doc.doc_id, cs))

    if all_chunks:
        write_chunks(all_chunks, settings.chunks_dir)

    # Pass the changed chunk_ids via XCom so the embed stage knows what to load
    return {
        "to_embed_chunk_ids": [c.chunk_id for c in to_embed],
        "to_embed_texts": [c.text for c in to_embed],
        "to_embed_payloads": [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "index": c.index,
                "source_path": c.source_path,
                "source_type": c.source_type,
            }
            for c in to_embed
        ],
        "deleted_doc_ids": disc.deleted_doc_ids,
        "docs_changed": len(disc.new_or_changed),
        "docs_unchanged": disc.unchanged_count,
    }


def stage_embed(payload: dict, settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    cids = payload.get("to_embed_chunk_ids", [])
    texts = payload.get("to_embed_texts", [])
    meta = payload.get("to_embed_payloads", [])
    if not cids:
        return {"chunks_embedded": 0, "deleted_doc_ids": payload.get("deleted_doc_ids", [])}

    chunks = [
        Chunk(
            chunk_id=m["chunk_id"],
            doc_id=m["doc_id"],
            index=m["index"],
            text=t,
            content_hash="",
            source_path=m["source_path"],
            source_type=m["source_type"],
        )
        for m, t in zip(meta, texts, strict=False)
    ]
    embedded = embed_chunks_distributed(
        chunks,
        model_name=settings.embed_model,
        batch_size=settings.embed_batch_size,
        num_workers=settings.ray_num_workers,
    )
    write_embeddings(embedded, settings.embeddings_dir)
    return {
        "chunks_embedded": len(embedded),
        "deleted_doc_ids": payload.get("deleted_doc_ids", []),
    }


def stage_index(payload: dict, settings: Settings | None = None, backends=("faiss",)) -> dict:
    settings = settings or get_settings()
    import numpy as np

    table = read_all_embeddings(settings.embeddings_dir)
    if table.num_rows == 0:
        return {"indexed": 0}

    # use the latest snapshot per chunk_id
    cids = table.column("chunk_id").to_pylist()
    vecs = np.asarray(table.column("embedding").to_pylist(), dtype="float32")
    payloads = [
        {
            "doc_id": d,
            "chunk_index": i,
            "source_path": sp,
            "source_type": st,
            "text": t,
        }
        for d, i, sp, st, t in zip(
            table.column("doc_id").to_pylist(),
            table.column("index").to_pylist(),
            table.column("source_path").to_pylist(),
            table.column("source_type").to_pylist(),
            table.column("text").to_pylist(),
            strict=False,
        )
    ]
    latest: dict[str, int] = {}
    for i, cid in enumerate(cids):
        latest[cid] = i  # later wins
    keep = list(latest.values())
    cids_kept = [cids[i] for i in keep]
    vecs_kept = vecs[keep]
    payloads_kept = [payloads[i] for i in keep]

    for backend in backends:
        store = get_store(backend, settings)
        if payload.get("deleted_doc_ids"):
            store.delete_by_doc(payload["deleted_doc_ids"])
        store.upsert(cids_kept, vecs_kept, payloads_kept)
        store.persist()

    # rebuild BM25 from chunks parquet
    chunks_table = read_all_chunks(settings.chunks_dir)
    if chunks_table.num_rows > 0:
        latest_text: dict[str, str] = {}
        latest_pl: dict[str, dict] = {}
        for cid, t, sp, st, d in zip(
            chunks_table.column("chunk_id").to_pylist(),
            chunks_table.column("text").to_pylist(),
            chunks_table.column("source_path").to_pylist(),
            chunks_table.column("source_type").to_pylist(),
            chunks_table.column("doc_id").to_pylist(),
            strict=False,
        ):
            latest_text[cid] = t
            latest_pl[cid] = {
                "doc_id": d,
                "source_path": sp,
                "source_type": st,
                "text": t,
            }
        bm25 = BM25Index(settings.bm25_dir / "bm25.pkl")
        bm25.build(list(latest_text.keys()), list(latest_text.values()), list(latest_pl.values()))
        bm25.persist()

    return {"indexed": len(cids_kept), "backends": list(backends)}
