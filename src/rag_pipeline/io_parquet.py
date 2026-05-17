from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from rag_pipeline.schema import Chunk, EmbeddedChunk


def _today_partition() -> str:
    return f"date={dt.date.today().isoformat()}"


def write_chunks(chunks: Iterable[Chunk], out_dir: Path) -> Path:
    rows = list(chunks)
    if not rows:
        return out_dir
    part = out_dir / _today_partition()
    part.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "chunk_id": [c.chunk_id for c in rows],
            "doc_id": [c.doc_id for c in rows],
            "index": [c.index for c in rows],
            "text": [c.text for c in rows],
            "content_hash": [c.content_hash for c in rows],
            "source_path": [c.source_path for c in rows],
            "source_type": [c.source_type for c in rows],
            "metadata": [json.dumps(c.metadata) for c in rows],
        }
    )
    # one file per batch, timestamp-named
    fname = part / f"chunks-{dt.datetime.utcnow().strftime('%H%M%S%f')}.parquet"
    pq.write_table(table, fname, compression="snappy")
    return fname


def write_embeddings(embedded: Iterable[EmbeddedChunk], out_dir: Path) -> Path:
    rows = list(embedded)
    if not rows:
        return out_dir
    part = out_dir / _today_partition()
    part.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "chunk_id": [e.chunk_id for e in rows],
            "doc_id": [e.doc_id for e in rows],
            "index": [e.index for e in rows],
            "text": [e.text for e in rows],
            "source_path": [e.source_path for e in rows],
            "source_type": [e.source_type for e in rows],
            "embedding": [list(e.embedding) for e in rows],
            "metadata": [json.dumps(e.metadata) for e in rows],
        }
    )
    fname = part / f"embeddings-{dt.datetime.utcnow().strftime('%H%M%S%f')}.parquet"
    pq.write_table(table, fname, compression="snappy")
    return fname


def read_all_embeddings(root: Path) -> pa.Table:
    files = sorted(root.rglob("*.parquet"))
    if not files:
        return pa.table({})
    return pa.concat_tables([pq.read_table(f) for f in files])


def read_all_chunks(root: Path) -> pa.Table:
    files = sorted(root.rglob("*.parquet"))
    if not files:
        return pa.table({})
    return pa.concat_tables([pq.read_table(f) for f in files])
