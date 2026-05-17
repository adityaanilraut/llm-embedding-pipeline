from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from rag_pipeline.config import get_settings
from rag_pipeline.pipeline import run_pipeline
from rag_pipeline.retrieve.bm25 import BM25Index
from rag_pipeline.retrieve.hybrid import hybrid_search
from rag_pipeline.store.factory import get_store

log = logging.getLogger("rag.api")

app = FastAPI(title="RAG Pipeline API", version="0.1.0")


class Hit(BaseModel):
    chunk_id: str
    score: float
    dense_rank: int | None
    sparse_rank: int | None
    source_path: str
    source_type: str
    text: str


class SearchResponse(BaseModel):
    query: str
    backend: str
    hits: list[Hit]


class IngestResponse(BaseModel):
    docs_changed: int
    docs_unchanged: int
    docs_deleted: int
    chunks_embedded: int
    chunks_skipped: int
    elapsed_s: float


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/ingest", response_model=IngestResponse)
def ingest(backend: str = "faiss") -> IngestResponse:
    if backend not in {"faiss", "qdrant", "both"}:
        raise HTTPException(400, "backend must be faiss, qdrant, or both")
    backends = ("faiss", "qdrant") if backend == "both" else (backend,)
    settings = get_settings()
    report = run_pipeline(settings, backends=backends)
    return IngestResponse(**report.__dict__)


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1),
    k: int = 10,
    backend: str = "faiss",
    source_type: str | None = None,
) -> SearchResponse:
    settings = get_settings()
    store = get_store(backend, settings)
    bm25 = BM25Index(settings.bm25_dir / "bm25.pkl")
    bm25.load()

    filters = {"source_type": source_type} if source_type else None
    hits = hybrid_search(
        q,
        vector_store=store,
        bm25=bm25,
        embed_model=settings.embed_model,
        k=k,
        candidate_k=max(k * 5, 25),
        k_rrf=settings.rrf_k,
        filters=filters,
    )
    return SearchResponse(
        query=q,
        backend=backend,
        hits=[
            Hit(
                chunk_id=h.chunk_id,
                score=h.score,
                dense_rank=h.dense_rank,
                sparse_rank=h.sparse_rank,
                source_path=h.payload.get("source_path", ""),
                source_type=h.payload.get("source_type", ""),
                text=h.payload.get("text", ""),
            )
            for h in hits
        ],
    )
