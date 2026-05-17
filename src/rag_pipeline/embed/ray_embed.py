from __future__ import annotations

from collections.abc import Iterable

from rag_pipeline.embed.model import embed_texts
from rag_pipeline.schema import Chunk, EmbeddedChunk


def _batched(seq: list[Chunk], size: int) -> Iterable[list[Chunk]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def embed_chunks_distributed(
    chunks: list[Chunk],
    model_name: str,
    batch_size: int = 64,
    num_workers: int = 2,
) -> list[EmbeddedChunk]:
    """Embed chunks in parallel via Ray. Falls back to in-process if Ray unavailable."""
    if not chunks:
        return []

    try:
        import ray
    except ImportError:
        return _embed_inline(chunks, model_name, batch_size)

    if not ray.is_initialized():
        ray.init(num_cpus=num_workers, ignore_reinit_error=True, log_to_driver=False)

    @ray.remote
    def _embed_batch(batch: list[Chunk]) -> list[EmbeddedChunk]:
        texts = [c.text for c in batch]
        vecs = embed_texts(texts, model_name, batch_size=len(texts))
        return [
            EmbeddedChunk(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                index=c.index,
                text=c.text,
                source_path=c.source_path,
                source_type=c.source_type,
                embedding=vec.tolist(),
                metadata=dict(c.metadata),
            )
            for c, vec in zip(batch, vecs, strict=False)
        ]

    futures = [_embed_batch.remote(b) for b in _batched(chunks, batch_size)]
    results: list[EmbeddedChunk] = []
    for r in ray.get(futures):
        results.extend(r)
    return results


def _embed_inline(chunks: list[Chunk], model_name: str, batch_size: int) -> list[EmbeddedChunk]:
    texts = [c.text for c in chunks]
    vecs = embed_texts(texts, model_name, batch_size=batch_size)
    return [
        EmbeddedChunk(
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            index=c.index,
            text=c.text,
            source_path=c.source_path,
            source_type=c.source_type,
            embedding=v.tolist(),
            metadata=dict(c.metadata),
        )
        for c, v in zip(chunks, vecs, strict=False)
    ]
