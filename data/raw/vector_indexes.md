# Vector Index Structures

ANN indexes trade exact recall for sub-linear query latency.

## Flat (brute force)
Stores every vector uncompressed. Exact, but O(N) per query. Fine up to a few
hundred thousand vectors on a single machine.

## IVF (Inverted File)
Partition the vector space into `nlist` Voronoi cells using k-means. At query
time, probe only `nprobe` cells. Recall depends on `nprobe / nlist`.

## PQ (Product Quantization)
Split each vector into `m` sub-vectors and encode each with a small codebook
(`nbits` per sub-vector). Compresses a 384-dim float32 vector (1536 bytes)
down to `m * nbits / 8` bytes — typically a 30-60x reduction.

## IVF-PQ
Combine the two: cluster with IVF, then PQ-compress residuals within each
cell. The default FAISS production index for medium-to-large corpora.

## Scalar Quantization
Qdrant's default optimization. Each float32 dimension is mapped to int8,
yielding 4x compression with minimal recall loss. Faster than PQ to build
but less aggressive in compression.
