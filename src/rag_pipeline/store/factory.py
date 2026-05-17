from __future__ import annotations

from rag_pipeline.config import Settings
from rag_pipeline.store.base import VectorStore
from rag_pipeline.store.faiss_store import FaissStore


def get_store(backend: str, settings: Settings) -> VectorStore:
    if backend == "faiss":
        store = FaissStore(
            dim=settings.embed_dim,
            index_dir=settings.faiss_dir,
            nlist=settings.faiss_nlist,
            m=settings.faiss_m,
            nbits=settings.faiss_nbits,
        )
        store.load()
        return store
    if backend == "qdrant":
        from rag_pipeline.store.qdrant_store import QdrantStore

        return QdrantStore(
            dim=settings.embed_dim,
            url=settings.qdrant_url,
            collection=settings.qdrant_collection,
        )
    raise ValueError(f"unknown backend: {backend}")
