from __future__ import annotations

import hashlib

from rag_pipeline.schema import Chunk, Document


def _split(text: str, size: int, overlap: int) -> list[str]:
    """Recursive character splitter — tries paragraph, then sentence, then char boundaries.

    Implemented inline rather than pulling langchain at runtime so the package
    works without the heavyweight LC dependency tree when only chunking is needed.
    Falls back to langchain's splitter when installed.
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return [c for c in splitter.split_text(text) if c.strip()]
    except ImportError:
        # naive fallback
        chunks: list[str] = []
        step = max(size - overlap, 1)
        for i in range(0, len(text), step):
            piece = text[i : i + size].strip()
            if piece:
                chunks.append(piece)
        return chunks


def chunk_document(doc: Document, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    pieces = _split(doc.text, chunk_size, chunk_overlap)
    out: list[Chunk] = []
    for i, piece in enumerate(pieces):
        content_hash = hashlib.sha256(piece.encode("utf-8")).hexdigest()
        out.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::{i}",
                doc_id=doc.doc_id,
                index=i,
                text=piece,
                content_hash=content_hash,
                source_path=doc.source_path,
                source_type=doc.source_type,
                metadata=dict(doc.metadata),
            )
        )
    return out
