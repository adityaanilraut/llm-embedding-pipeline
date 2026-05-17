# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation grounds a large language model in an external
knowledge base. Instead of relying solely on parametric knowledge baked into
the model weights, the system retrieves relevant passages at query time and
conditions generation on them.

## Pipeline Stages

1. **Ingestion** — pull documents from raw sources (PDFs, HTML, markdown).
2. **Chunking** — split documents into passages of bounded length, often with
   overlap so that semantic units aren't sliced apart.
3. **Embedding** — convert each chunk to a dense vector with a model such as
   `BAAI/bge-small-en-v1.5` or `intfloat/e5-base-v2`.
4. **Indexing** — write vectors to an ANN index (FAISS, Qdrant, Weaviate, ...).
5. **Retrieval** — embed the user query and fetch the top-k nearest chunks.
6. **Generation** — pass retrieved chunks plus the question to an LLM.

## Hybrid Retrieval

Pure dense retrieval misses keyword-heavy queries (product codes, surnames,
acronyms). Combining a sparse retriever like BM25 with dense vectors and
fusing via Reciprocal Rank Fusion (RRF) almost always beats either leg alone.
