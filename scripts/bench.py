"""Benchmark search latency for FAISS (Flat vs IVF-PQ) on synthetic vectors.

Backs up the resume claim about reducing embedding query latency through
vector compression.

Run: python scripts/bench.py
"""

from __future__ import annotations

import statistics
import time

import numpy as np


def _gen(n: int, d: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, d)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v


def bench(index, queries: np.ndarray, k: int = 10) -> tuple[float, float]:
    # warmup
    for q in queries[:5]:
        index.search(q.reshape(1, -1), k)
    samples = []
    for q in queries:
        t0 = time.perf_counter()
        index.search(q.reshape(1, -1), k)
        samples.append((time.perf_counter() - t0) * 1000)
    return statistics.median(samples), float(np.percentile(samples, 95))


def main() -> None:
    import faiss

    N, D = 100_000, 384
    vecs = _gen(N, D)
    queries = _gen(200, D, seed=42)

    print(f"corpus={N}, dim={D}, queries={len(queries)}")

    flat = faiss.IndexFlatIP(D)
    flat.add(vecs)
    p50, p95 = bench(flat, queries)
    flat_bytes = N * D * 4
    print(f"FlatIP        : p50={p50:6.2f}ms  p95={p95:6.2f}ms  size={flat_bytes / 1e6:6.1f} MB")

    nlist, m, nbits = 256, 16, 8
    quant = faiss.IndexFlatIP(D)
    ivfpq = faiss.IndexIVFPQ(quant, D, nlist, m, nbits)
    ivfpq.metric_type = faiss.METRIC_INNER_PRODUCT
    ivfpq.train(vecs)
    ivfpq.add(vecs)
    ivfpq.nprobe = 16
    p50, p95 = bench(ivfpq, queries)
    pq_bytes = N * m * (nbits / 8) + nlist * D * 4  # rough: codes + centroids
    print(
        f"IVFPQ(nlist={nlist},m={m},nbits={nbits}): "
        f"p50={p50:6.2f}ms  p95={p95:6.2f}ms  size≈{pq_bytes / 1e6:6.1f} MB "
        f"(compression {flat_bytes / pq_bytes:5.1f}x)"
    )


if __name__ == "__main__":
    main()
