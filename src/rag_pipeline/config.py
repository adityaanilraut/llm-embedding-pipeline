from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", extra="ignore")

    data_dir: Path = Path("./data")
    index_dir: Path = Path("./indices")
    state_db: Path = Path("./data/state.db")

    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384
    embed_batch_size: int = 64

    chunk_size: int = 512
    chunk_overlap: int = 64

    ray_num_workers: int = 2

    faiss_nlist: int = 32
    faiss_m: int = 16
    faiss_nbits: int = 8

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "rag_chunks"

    rrf_k: int = 60
    top_k: int = 10

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def chunks_dir(self) -> Path:
        return self.data_dir / "staging" / "chunks"

    @property
    def embeddings_dir(self) -> Path:
        return self.data_dir / "staging" / "embeddings"

    @property
    def faiss_dir(self) -> Path:
        return self.index_dir / "faiss"

    @property
    def bm25_dir(self) -> Path:
        return self.index_dir / "bm25"


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
