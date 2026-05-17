from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rag_pipeline.embed.model import embed_one
from rag_pipeline.retrieve.bm25 import BM25Index
from rag_pipeline.store.base import VectorStore


@dataclass
class HybridHit:
    chunk_id: str
    score: float  # RRF fused score
    dense_rank: int | None
    sparse_rank: int | None
    payload: dict[str, Any]


def rrf_fuse(
    dense: list[tuple[str, dict]],
    sparse: list[tuple[str, dict]],
    k_rrf: int = 60,
) -> list[HybridHit]:
    """Reciprocal Rank Fusion.

    Each hit list contributes 1 / (k_rrf + rank) to a chunk's score.
    """
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}
    dense_ranks: dict[str, int] = {}
    sparse_ranks: dict[str, int] = {}

    for r, (cid, pl) in enumerate(dense):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k_rrf + r + 1)
        payloads[cid] = pl
        dense_ranks[cid] = r
    for r, (cid, pl) in enumerate(sparse):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k_rrf + r + 1)
        payloads.setdefault(cid, pl)
        sparse_ranks[cid] = r

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [
        HybridHit(
            chunk_id=cid,
            score=s,
            dense_rank=dense_ranks.get(cid),
            sparse_rank=sparse_ranks.get(cid),
            payload=payloads[cid],
        )
        for cid, s in ranked
    ]


def hybrid_search(
    query: str,
    *,
    vector_store: VectorStore,
    bm25: BM25Index,
    embed_model: str,
    k: int = 10,
    candidate_k: int = 50,
    k_rrf: int = 60,
    filters: dict | None = None,
) -> list[HybridHit]:
    q_vec = np.asarray(embed_one(query, embed_model))
    dense_raw = vector_store.search(q_vec, k=candidate_k, filters=filters)
    sparse_raw = bm25.search(query, k=candidate_k, filters=filters)

    dense = [(h.chunk_id, h.payload) for h in dense_raw]
    sparse = [(cid, pl) for cid, _score, pl in sparse_raw]

    return rrf_fuse(dense, sparse, k_rrf=k_rrf)[:k]
