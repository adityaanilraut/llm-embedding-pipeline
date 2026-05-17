"""Airflow DAG: rag_pipeline.

Stages:
  1. discover_and_chunk  — diff raw_dir against state, chunk new/changed docs, write Parquet.
  2. embed               — Ray-distributed embedding of changed chunks; write Parquet.
  3. index               — push to FAISS + Qdrant; rebuild BM25.

State (file hashes + chunk hashes) is persisted in SQLite, so each run is
incremental: unchanged files are skipped, unchanged chunks reuse embeddings.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

from airflow.operators.python import PythonOperator

from airflow import DAG

# Make src/ importable when DAGs run from the airflow container
PROJECT_ROOT = os.environ.get("RAG_PROJECT_ROOT", "/opt/airflow/project")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))


def _discover(**ctx):
    from rag_pipeline.pipeline import stage_discover_and_chunk

    out = stage_discover_and_chunk()
    return out


def _embed(**ctx):
    from rag_pipeline.pipeline import stage_embed

    payload = ctx["ti"].xcom_pull(task_ids="discover_and_chunk")
    return stage_embed(payload)


def _index(**ctx):
    from rag_pipeline.pipeline import stage_index

    payload = ctx["ti"].xcom_pull(task_ids="embed")
    backends = tuple(os.environ.get("RAG_BACKENDS", "faiss").split(","))
    return stage_index(payload, backends=backends)


default_args = {
    "owner": "rag-pipeline",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="rag_pipeline",
    description="Incremental document → chunk → embed → index pipeline",
    start_date=datetime(2024, 1, 1),
    schedule="@hourly",
    catchup=False,
    default_args=default_args,
    tags=["rag", "embeddings", "ray", "faiss", "qdrant"],
) as dag:
    discover_and_chunk = PythonOperator(
        task_id="discover_and_chunk",
        python_callable=_discover,
    )
    embed = PythonOperator(
        task_id="embed",
        python_callable=_embed,
    )
    index = PythonOperator(
        task_id="index",
        python_callable=_index,
    )

    discover_and_chunk >> embed >> index
