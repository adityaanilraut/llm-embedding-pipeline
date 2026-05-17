from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class BM25State:
    bm25: BM25Okapi
    chunk_ids: list[str]
    payloads: list[dict]


class BM25Index:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._state: BM25State | None = None

    def build(self, chunk_ids: list[str], texts: list[str], payloads: list[dict]) -> None:
        tokens = [tokenize(t) for t in texts]
        if not tokens:
            self._state = None
            return
        self._state = BM25State(
            bm25=BM25Okapi(tokens),
            chunk_ids=chunk_ids,
            payloads=payloads,
        )

    def search(
        self, query: str, k: int, filters: dict | None = None
    ) -> list[tuple[str, float, dict]]:
        if self._state is None:
            return []
        q_tok = tokenize(query)
        if not q_tok:
            return []
        scores = self._state.bm25.get_scores(q_tok)
        order = scores.argsort()[::-1]
        hits: list[tuple[str, float, dict]] = []
        for idx in order:
            pl = self._state.payloads[idx]
            if filters and not all(pl.get(k_) == v for k_, v in filters.items()):
                continue
            hits.append((self._state.chunk_ids[idx], float(scores[idx]), pl))
            if len(hits) >= k:
                break
        return hits

    def persist(self) -> None:
        with open(self.path, "wb") as f:
            pickle.dump(self._state, f)

    def load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, "rb") as f:
            self._state = pickle.load(f)
