import numpy as np
import pytest

pytest.importorskip("faiss")


def test_rrf_fusion_orders_by_combined_rank():
    from rag_pipeline.retrieve.hybrid import rrf_fuse

    dense = [("a", {}), ("b", {}), ("c", {})]
    sparse = [("c", {}), ("b", {}), ("d", {})]
    hits = rrf_fuse(dense, sparse, k_rrf=10)
    # b appears in top of both → should outrank a and d
    order = [h.chunk_id for h in hits]
    assert order.index("b") < order.index("a")
    assert order.index("b") < order.index("d")


def test_hybrid_end_to_end_in_memory(tmp_path, monkeypatch):
    """Tiny corpus → embed → FAISS + BM25 → query, assert top hit is right."""
    from rag_pipeline.embed.model import embed_texts
    from rag_pipeline.retrieve.bm25 import BM25Index
    from rag_pipeline.retrieve.hybrid import hybrid_search
    from rag_pipeline.store.faiss_store import FaissStore

    corpus = [
        ("c1", "Cats are small carnivorous mammals."),
        ("c2", "FAISS is a library for efficient similarity search."),
        ("c3", "Reciprocal Rank Fusion blends multiple ranked lists."),
        ("c4", "Apache Airflow orchestrates DAGs of tasks."),
    ]
    model = "BAAI/bge-small-en-v1.5"
    try:
        vecs = embed_texts([t for _, t in corpus], model)
    except Exception as e:
        pytest.skip(f"sentence-transformers unavailable: {e}")
    dim = vecs.shape[1]

    store = FaissStore(dim=dim, index_dir=tmp_path / "faiss")
    payloads = [
        {"doc_id": cid, "text": t, "source_type": "txt", "source_path": "/x"} for cid, t in corpus
    ]
    store.upsert([cid for cid, _ in corpus], np.asarray(vecs), payloads)

    bm25 = BM25Index(tmp_path / "bm25.pkl")
    bm25.build([cid for cid, _ in corpus], [t for _, t in corpus], payloads)

    hits = hybrid_search(
        "what is reciprocal rank fusion",
        vector_store=store,
        bm25=bm25,
        embed_model=model,
        k=2,
    )
    assert hits[0].chunk_id == "c3"
