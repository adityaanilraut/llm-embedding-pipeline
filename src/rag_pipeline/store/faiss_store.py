from __future__ import annotations

import json
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rag_pipeline.store.base import SearchHit


class FaissStore:
    """FAISS-backed store using IndexIVFPQ for vector compression.

    Falls back to IndexFlatIP automatically when the corpus is smaller than the
    minimum training set for IVF-PQ (a few hundred vectors). This keeps the
    demo runnable on tiny corpora while still exercising the IVF-PQ code path
    on real data.
    """

    name = "faiss"

    def __init__(
        self,
        dim: int,
        index_dir: Path,
        nlist: int = 32,
        m: int = 16,
        nbits: int = 8,
    ):
        self.dim = dim
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "index.faiss"
        self.payload_path = self.index_dir / "payloads.pkl"
        self.meta_path = self.index_dir / "meta.json"
        self.nlist = nlist
        self.m = m
        self.nbits = nbits

        self._index = None  # lazy
        self._id_to_idx: dict[str, int] = {}
        self._idx_to_id: dict[int, str] = {}
        self._payloads: dict[str, dict[str, Any]] = {}
        self._doc_to_chunks: dict[str, set[str]] = {}
        self._next_idx = 0
        self._uses_ivfpq = False

    # --- index management ---

    def _ensure_index(self, train_vecs: np.ndarray | None = None) -> None:
        import faiss

        if self._index is not None:
            return

        if train_vecs is not None and len(train_vecs) >= max(self.nlist * 4, 256):
            quant = faiss.IndexFlatIP(self.dim)
            index = faiss.IndexIVFPQ(quant, self.dim, self.nlist, self.m, self.nbits)
            index.metric_type = faiss.METRIC_INNER_PRODUCT
            index.train(train_vecs.astype(np.float32))
            index.nprobe = min(8, self.nlist)
            self._uses_ivfpq = True
            id_map = faiss.IndexIDMap2(index)
            self._index = id_map
        else:
            base = faiss.IndexFlatIP(self.dim)
            self._index = faiss.IndexIDMap2(base)
            self._uses_ivfpq = False

    # --- VectorStore API ---

    def upsert(
        self,
        chunk_ids: Sequence[str],
        vectors: np.ndarray,
        payloads: Sequence[dict[str, Any]],
    ) -> None:
        if len(chunk_ids) == 0:
            return
        vectors = vectors.astype(np.float32)
        self._ensure_index(train_vecs=vectors)

        # delete any prior entries for these chunk_ids so upsert is idempotent
        existing = [self._id_to_idx[c] for c in chunk_ids if c in self._id_to_idx]
        if existing:
            self._index.remove_ids(np.array(existing, dtype=np.int64))
            for c in chunk_ids:
                if c in self._id_to_idx:
                    idx = self._id_to_idx.pop(c)
                    self._idx_to_id.pop(idx, None)

        new_idxs = np.array([self._allocate_idx(cid) for cid in chunk_ids], dtype=np.int64)
        self._index.add_with_ids(vectors, new_idxs)
        for cid, pl in zip(chunk_ids, payloads, strict=False):
            self._payloads[cid] = pl
            self._doc_to_chunks.setdefault(pl.get("doc_id", ""), set()).add(cid)

    def _allocate_idx(self, chunk_id: str) -> int:
        idx = self._next_idx
        self._next_idx += 1
        self._id_to_idx[chunk_id] = idx
        self._idx_to_id[idx] = chunk_id
        return idx

    def delete_by_doc(self, doc_ids: Sequence[str]) -> None:
        if not doc_ids or self._index is None:
            return

        to_remove: list[int] = []
        for d in doc_ids:
            for cid in self._doc_to_chunks.pop(d, set()):
                idx = self._id_to_idx.pop(cid, None)
                if idx is not None:
                    self._idx_to_id.pop(idx, None)
                    to_remove.append(idx)
                self._payloads.pop(cid, None)
        if to_remove:
            self._index.remove_ids(np.array(to_remove, dtype=np.int64))

    def search(
        self,
        query: np.ndarray,
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        if self._index is None or self.count() == 0:
            return []
        q = query.astype(np.float32).reshape(1, -1)
        # over-fetch when filtering since we filter post-hoc
        fetch_k = k * 4 if filters else k
        scores, ids = self._index.search(q, min(fetch_k, max(self.count(), 1)))
        hits: list[SearchHit] = []
        for s, i in zip(scores[0], ids[0], strict=False):
            if i == -1:
                continue
            cid = self._idx_to_id.get(int(i))
            if cid is None:
                continue
            pl = self._payloads.get(cid, {})
            if filters and not _matches(pl, filters):
                continue
            hits.append(SearchHit(chunk_id=cid, score=float(s), payload=pl))
            if len(hits) >= k:
                break
        return hits

    def count(self) -> int:
        return self._index.ntotal if self._index is not None else 0

    # --- persistence ---

    def persist(self) -> None:
        import faiss

        if self._index is not None:
            faiss.write_index(self._index, str(self.index_path))
        with open(self.payload_path, "wb") as f:
            pickle.dump(
                {
                    "id_to_idx": self._id_to_idx,
                    "idx_to_id": self._idx_to_id,
                    "payloads": self._payloads,
                    "doc_to_chunks": self._doc_to_chunks,
                    "next_idx": self._next_idx,
                },
                f,
            )
        with open(self.meta_path, "w") as f:
            json.dump(
                {
                    "dim": self.dim,
                    "nlist": self.nlist,
                    "m": self.m,
                    "nbits": self.nbits,
                    "uses_ivfpq": self._uses_ivfpq,
                },
                f,
            )

    def load(self) -> None:
        import faiss

        if self.index_path.exists():
            self._index = faiss.read_index(str(self.index_path))
        if self.payload_path.exists():
            with open(self.payload_path, "rb") as f:
                state = pickle.load(f)
            self._id_to_idx = state["id_to_idx"]
            self._idx_to_id = {int(k): v for k, v in state["idx_to_id"].items()}
            self._payloads = state["payloads"]
            self._doc_to_chunks = state["doc_to_chunks"]
            self._next_idx = state["next_idx"]
        if self.meta_path.exists():
            with open(self.meta_path) as f:
                self._uses_ivfpq = bool(json.load(f).get("uses_ivfpq", False))


def _matches(payload: dict[str, Any], filters: dict[str, Any]) -> bool:
    for k, v in filters.items():
        if payload.get(k) != v:
            return False
    return True
