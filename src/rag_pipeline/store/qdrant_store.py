from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from rag_pipeline.store.base import SearchHit


def _chunk_id_to_point_id(chunk_id: str) -> str:
    """Qdrant point IDs must be uint64 or UUID. We use deterministic UUID5."""
    import uuid

    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantStore:
    """Qdrant-backed store with scalar quantization (int8) for compression."""

    name = "qdrant"

    def __init__(self, dim: int, url: str, collection: str):
        from qdrant_client import QdrantClient

        self.dim = dim
        self.url = url
        self.collection = collection
        self.client = QdrantClient(url=url)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        from qdrant_client.http import models as qm

        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection in existing:
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE),
            quantization_config=qm.ScalarQuantization(
                scalar=qm.ScalarQuantizationConfig(
                    type=qm.ScalarType.INT8,
                    always_ram=True,
                )
            ),
        )

    def upsert(
        self,
        chunk_ids: Sequence[str],
        vectors: np.ndarray,
        payloads: Sequence[dict[str, Any]],
    ) -> None:
        if len(chunk_ids) == 0:
            return
        from qdrant_client.http import models as qm

        points = [
            qm.PointStruct(
                id=_chunk_id_to_point_id(cid),
                vector=vec.tolist(),
                payload={**pl, "chunk_id": cid},
            )
            for cid, vec, pl in zip(chunk_ids, vectors, payloads, strict=False)
        ]
        self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def delete_by_doc(self, doc_ids: Sequence[str]) -> None:
        if not doc_ids:
            return
        from qdrant_client.http import models as qm

        self.client.delete(
            collection_name=self.collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    should=[
                        qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=d))
                        for d in doc_ids
                    ]
                )
            ),
            wait=True,
        )

    def search(
        self,
        query: np.ndarray,
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        from qdrant_client.http import models as qm

        q_filter = None
        if filters:
            q_filter = qm.Filter(
                must=[
                    qm.FieldCondition(key=k_, match=qm.MatchValue(value=v))
                    for k_, v in filters.items()
                ]
            )
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query.astype(np.float32).tolist(),
            limit=k,
            query_filter=q_filter,
            with_payload=True,
        )
        return [
            SearchHit(
                chunk_id=r.payload["chunk_id"],
                score=float(r.score),
                payload=r.payload,
            )
            for r in results
        ]

    def count(self) -> int:
        return int(self.client.count(self.collection, exact=True).count)

    def persist(self) -> None:
        """Qdrant persists server-side; nothing to do."""

    def load(self) -> None:
        """Qdrant state lives in the server."""
