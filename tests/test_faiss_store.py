import numpy as np
import pytest

faiss = pytest.importorskip("faiss")  # noqa: F841


def _rand_vecs(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, d)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v


def test_faiss_store_upsert_search_delete(tmp_path):
    from rag_pipeline.store.faiss_store import FaissStore

    store = FaissStore(dim=32, index_dir=tmp_path)
    vecs = _rand_vecs(10, 32)
    cids = [f"c{i}" for i in range(10)]
    payloads = [{"doc_id": f"d{i // 2}", "text": f"t{i}", "source_type": "txt"} for i in range(10)]
    store.upsert(cids, vecs, payloads)

    hits = store.search(vecs[3], k=3)
    assert hits and hits[0].chunk_id == "c3"

    store.delete_by_doc(["d1"])  # removes c2, c3
    hit_ids = [h.chunk_id for h in store.search(vecs[3], k=10)]
    assert "c2" not in hit_ids and "c3" not in hit_ids


def test_faiss_persistence_round_trip(tmp_path):
    from rag_pipeline.store.faiss_store import FaissStore

    s1 = FaissStore(dim=16, index_dir=tmp_path)
    vecs = _rand_vecs(5, 16)
    s1.upsert([f"c{i}" for i in range(5)], vecs, [{"doc_id": "d", "text": "t"}] * 5)
    s1.persist()

    s2 = FaissStore(dim=16, index_dir=tmp_path)
    s2.load()
    assert s2.count() == 5
    hits = s2.search(vecs[0], k=1)
    assert hits[0].chunk_id == "c0"
