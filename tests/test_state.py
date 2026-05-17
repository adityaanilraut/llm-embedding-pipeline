from rag_pipeline.state import StateStore


def test_state_round_trip(tmp_path):
    s = StateStore(tmp_path / "state.db")
    assert s.known_file("/a") is None
    s.upsert_file("/a", "doc1", mtime=1.0, indexed_at=1.0)
    assert s.known_file("/a") == ("doc1", 1.0)

    s.replace_chunks("doc1", {"doc1::0": "h0", "doc1::1": "h1"})
    assert s.known_chunks_for_doc("doc1") == {"doc1::0": "h0", "doc1::1": "h1"}

    # delete tombstones chunks too
    assert s.delete_file("/a") == "doc1"
    assert s.known_chunks_for_doc("doc1") == {}


def test_incremental_diff_chunks(tmp_path):
    from rag_pipeline.ingest.discover import diff_chunks
    from rag_pipeline.schema import Chunk

    s = StateStore(tmp_path / "state.db")
    chunks_v1 = [
        Chunk("doc::0", "doc", 0, "hello", "h0", "/p", "txt"),
        Chunk("doc::1", "doc", 1, "world", "h1", "/p", "txt"),
    ]
    # first run: everything is new
    out = diff_chunks(s, "doc", chunks_v1)
    assert {c.chunk_id for c in out} == {"doc::0", "doc::1"}

    # second run, identical → nothing to embed
    out = diff_chunks(s, "doc", chunks_v1)
    assert out == []

    # third run, one chunk's hash changed
    chunks_v2 = [
        Chunk("doc::0", "doc", 0, "hello!", "h0_changed", "/p", "txt"),
        Chunk("doc::1", "doc", 1, "world", "h1", "/p", "txt"),
    ]
    out = diff_chunks(s, "doc", chunks_v2)
    assert [c.chunk_id for c in out] == ["doc::0"]
