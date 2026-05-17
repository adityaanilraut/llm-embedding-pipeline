from rag_pipeline.ingest.chunker import chunk_document
from rag_pipeline.schema import Document


def test_chunker_produces_overlapping_chunks():
    text = " ".join(f"This is sentence {i}." for i in range(200))
    doc = Document(doc_id="abc", source_path="t.txt", source_type="txt", text=text)
    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=40)
    assert len(chunks) >= 2
    # each chunk hash differs from neighbors (no exact duplicates)
    assert len({c.content_hash for c in chunks}) == len(chunks)
    # chunk_id encodes doc_id + index
    assert all(c.chunk_id.startswith("abc::") for c in chunks)


def test_chunker_handles_empty_text():
    doc = Document(doc_id="empty", source_path="t.txt", source_type="txt", text="")
    assert chunk_document(doc, chunk_size=200, chunk_overlap=40) == []
