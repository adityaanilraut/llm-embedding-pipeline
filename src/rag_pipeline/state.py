from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    path        TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL,           -- sha256 of file content
    mtime       REAL NOT NULL,
    indexed_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      TEXT PRIMARY KEY,
    doc_id        TEXT NOT NULL,
    content_hash  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
"""


class StateStore:
    """SQLite-backed state for incremental embedding."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as cx:
            cx.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        cx = sqlite3.connect(self.db_path)
        try:
            yield cx
            cx.commit()
        finally:
            cx.close()

    # --- files ---

    def known_file(self, path: str) -> tuple[str, float] | None:
        """Return (doc_id, mtime) for `path` if known, else None."""
        with self._conn() as cx:
            row = cx.execute("SELECT doc_id, mtime FROM files WHERE path = ?", (path,)).fetchone()
        return (row[0], row[1]) if row else None

    def upsert_file(self, path: str, doc_id: str, mtime: float, indexed_at: float) -> None:
        with self._conn() as cx:
            cx.execute(
                "INSERT INTO files(path, doc_id, mtime, indexed_at) VALUES(?,?,?,?) "
                "ON CONFLICT(path) DO UPDATE SET doc_id=excluded.doc_id, "
                "mtime=excluded.mtime, indexed_at=excluded.indexed_at",
                (path, doc_id, mtime, indexed_at),
            )

    def all_paths(self) -> set[str]:
        with self._conn() as cx:
            return {row[0] for row in cx.execute("SELECT path FROM files")}

    def delete_file(self, path: str) -> str | None:
        """Remove file row; return the doc_id that was deleted (for tombstoning)."""
        with self._conn() as cx:
            row = cx.execute("SELECT doc_id FROM files WHERE path = ?", (path,)).fetchone()
            if not row:
                return None
            doc_id = row[0]
            cx.execute("DELETE FROM files WHERE path = ?", (path,))
            cx.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            return doc_id

    # --- chunks ---

    def known_chunks_for_doc(self, doc_id: str) -> dict[str, str]:
        """chunk_id -> content_hash for the given doc."""
        with self._conn() as cx:
            return {
                row[0]: row[1]
                for row in cx.execute(
                    "SELECT chunk_id, content_hash FROM chunks WHERE doc_id = ?", (doc_id,)
                )
            }

    def replace_chunks(self, doc_id: str, chunk_id_to_hash: dict[str, str]) -> None:
        with self._conn() as cx:
            cx.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            cx.executemany(
                "INSERT INTO chunks(chunk_id, doc_id, content_hash) VALUES(?,?,?)",
                [(cid, doc_id, h) for cid, h in chunk_id_to_hash.items()],
            )
