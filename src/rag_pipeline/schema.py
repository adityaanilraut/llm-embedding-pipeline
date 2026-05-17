from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    doc_id: str  # sha256 of file content
    source_path: str
    source_type: str  # "pdf" | "md" | "txt" | "html"
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str  # f"{doc_id}::{index}"
    doc_id: str
    index: int
    text: str
    content_hash: str  # sha256 of chunk text — enables chunk-level incremental updates
    source_path: str
    source_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddedChunk:
    chunk_id: str
    doc_id: str
    index: int
    text: str
    source_path: str
    source_type: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
