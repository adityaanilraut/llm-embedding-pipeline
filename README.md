# LLM Embedding & Vector Data Infrastructure Pipeline

A distributed document ingestion and embedding pipeline that demonstrates the
core data infrastructure underneath a production RAG system.

- **Ray** for distributed embedding compute
- **Airflow** for pipeline orchestration
- **FAISS (IVF-PQ)** + **Qdrant (scalar-quantized)** as pluggable vector stores
- **Parquet** as the columnar staging format
- **Hybrid retrieval** (dense vectors + BM25, fused via Reciprocal Rank Fusion)
- **Incremental embedding** — only re-embed files (and chunks within files) whose
  SHA-256 hash changed since the last run; deleted files are tombstoned

## Architecture

```
data/raw/ (PDF, MD, TXT, HTML)
   │
   ▼ [Airflow DAG: rag_pipeline]
┌──────────────┐   ┌──────────────┐   ┌──────────────────┐   ┌──────────────┐
│ 1. Discover  │──▶│ 2. Parse +   │──▶│ 3. Ray-embed     │──▶│ 4. Index     │
│    + hash    │   │    chunk     │   │    (distributed) │   │    FAISS +   │
│ (incremental)│   │    → Parquet │   │    → Parquet     │   │    Qdrant    │
└──────────────┘   └──────────────┘   └──────────────────┘   └──────────────┘
                                                                    │
                                                                    ▼
                                                          FastAPI /search  ◀── Streamlit UI
                                                          (hybrid: BM25 + dense + RRF)
```

## Project layout

```
.
├── docker-compose.yml           # full stack: airflow, qdrant, api, ui
├── Dockerfile, airflow.Dockerfile
├── pyproject.toml               # dependencies
├── airflow/dags/rag_pipeline.py # 3-stage DAG
├── data/raw/                    # drop your documents here
├── indices/                     # FAISS index + BM25 pickle persist here
├── scripts/
│   ├── run_pipeline.py          # CLI: run pipeline once without airflow
│   └── bench.py                 # FAISS Flat vs IVF-PQ latency / size comparison
├── src/rag_pipeline/
│   ├── config.py                # pydantic settings, reads .env
│   ├── schema.py                # Document, Chunk, EmbeddedChunk
│   ├── state.py                 # SQLite hash store for incremental updates
│   ├── io_parquet.py            # date-partitioned Parquet writers
│   ├── ingest/                  # parsers, chunker, discover
│   ├── embed/                   # SentenceTransformer + Ray batching
│   ├── store/                   # FaissStore, QdrantStore, factory
│   ├── retrieve/                # BM25, hybrid RRF
│   ├── api/                     # FastAPI + Streamlit
│   └── pipeline.py              # high-level orchestration + airflow stage entry points
└── tests/
```

## Quick start (Docker)

```bash
cp .env.example .env
docker compose build
docker compose up -d
```

Services:

| Component  | URL                       | Notes |
|------------|---------------------------|-------|
| Airflow    | http://localhost:8080     | admin / admin |
| Qdrant     | http://localhost:6333     | dashboard at `/dashboard` |
| FastAPI    | http://localhost:8000     | OpenAPI at `/docs` |
| Streamlit  | http://localhost:8501     | hybrid search UI |

Drop documents (PDF/MD/TXT/HTML) into `data/raw/`, then in the Airflow UI
unpause and trigger the `rag_pipeline` DAG (or hit `POST /ingest?backend=both`).

## Quick start (local Python, no docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# run the whole pipeline once into FAISS
python scripts/run_pipeline.py --backends faiss

# serve the API
uvicorn rag_pipeline.api.main:app --reload

# ...and the UI in another shell
streamlit run src/rag_pipeline/api/ui.py
```

To also push to a local Qdrant: `docker run -p 6333:6333 qdrant/qdrant:v1.9.2`
then `python scripts/run_pipeline.py --backends faiss,qdrant`.

## Search

```bash
curl 'http://localhost:8000/search?q=what+is+RRF&backend=faiss&k=5'
```

The response includes both fused RRF score and the per-leg ranks
(`dense_rank`, `sparse_rank`) so you can see how each query is answered.

## Incremental updates

The pipeline is driven by hashes, not timestamps:

- A `state.db` SQLite stores `sha256(file_content)` for each file and
  `sha256(chunk_text)` for each chunk.
- On each run, files whose hash matches the prior record are skipped entirely.
- For changed files, individual chunks are diffed by hash — unchanged chunks
  reuse their existing embedding in the vector store.
- Files that disappeared from `data/raw/` are tombstoned: the pipeline deletes
  their chunks from both FAISS and Qdrant by `doc_id`.

Smoke test:

```bash
python scripts/run_pipeline.py                 # full first run
python scripts/run_pipeline.py                 # second run: chunks_embedded=0
echo "more text" >> data/raw/intro_rag.md
python scripts/run_pipeline.py                 # only the changed chunks re-embed
```

## Benchmarks

```bash
python scripts/bench.py
```

Compares Flat (uncompressed, exact) against IVF-PQ (compressed, approximate)
on 100k synthetic 384-dim vectors. IVF-PQ typically gives ~30-60x storage
compression with sub-millisecond search at recall around 0.9.

## Tests

```bash
pytest -q
```

Covers chunking, the SQLite state store + chunk-level incremental diff,
FAISS upsert/search/delete + persistence, and an end-to-end hybrid retrieval
test with a tiny corpus.

## Configuration

All knobs live in `.env` (see `.env.example`):

| Variable | Purpose |
|---|---|
| `RAG_EMBED_MODEL` | sentence-transformers model id (default BGE-small) |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | chunking parameters |
| `RAG_RAY_NUM_WORKERS` | Ray task parallelism for embedding |
| `RAG_FAISS_NLIST` / `RAG_FAISS_M` / `RAG_FAISS_NBITS` | IVF-PQ compression knobs |
| `RAG_QDRANT_URL`, `RAG_QDRANT_COLLECTION` | Qdrant connection |
| `RAG_RRF_K` | denominator constant for Reciprocal Rank Fusion |
