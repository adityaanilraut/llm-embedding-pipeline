from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass
class SearchHit:
    chunk_id: str
    score: float
    payload: dict[str, Any]


class VectorStore(Protocol):
    """Common interface for FAISS and Qdrant backends."""

    name: str
    dim: int

    def upsert(
        self,
        chunk_ids: Sequence[str],
        vectors: np.ndarray,
        payloads: Sequence[dict[str, Any]],
    ) -> None: ...

    def delete_by_doc(self, doc_ids: Sequence[str]) -> None: ...

    def search(
        self,
        query: np.ndarray,
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchHit]: ...

    def persist(self) -> None: ...

    def load(self) -> None: ...

    def count(self) -> int: ...
