"""One-shot CLI to run the ingestion pipeline end-to-end.

Useful for local testing without spinning up Airflow.

Example:
    python scripts/run_pipeline.py --backends faiss
    python scripts/run_pipeline.py --backends faiss,qdrant
"""

from __future__ import annotations

import argparse
import logging

from rag_pipeline.pipeline import run_pipeline


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backends", default="faiss", help="comma-separated subset of {faiss,qdrant}")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    backends = tuple(b.strip() for b in args.backends.split(",") if b.strip())
    report = run_pipeline(backends=backends)
    print(report)


if __name__ == "__main__":
    main()
