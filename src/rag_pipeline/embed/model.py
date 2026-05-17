from __future__ import annotations

import threading
from functools import lru_cache

import numpy as np

_lock = threading.Lock()


@lru_cache(maxsize=4)
def _load_model(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def embed_texts(texts: list[str], model_name: str, batch_size: int = 64) -> np.ndarray:
    """Encode `texts` with the named sentence-transformers model.

    Returns L2-normalized float32 vectors so cosine similarity == inner product
    in both FAISS and Qdrant.
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    with _lock:
        model = _load_model(model_name)
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vecs.astype(np.float32)


def embed_one(text: str, model_name: str) -> np.ndarray:
    return embed_texts([text], model_name)[0]
