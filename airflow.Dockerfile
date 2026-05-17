FROM apache/airflow:2.9.2-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 \
 && rm -rf /var/lib/apt/lists/*

USER airflow
COPY pyproject.toml /opt/airflow/project/
COPY src /opt/airflow/project/src

RUN pip install --no-cache-dir \
    "pydantic>=2.6" "pydantic-settings>=2.2" \
    "pyarrow>=15.0" "pandas>=2.2" "numpy>=1.26,<2.0" \
    "sentence-transformers>=2.7" "torch>=2.2" \
    "faiss-cpu>=1.7.4" "qdrant-client>=1.9" \
    "rank-bm25>=0.2.2" "pypdf>=4.2" "beautifulsoup4>=4.12" \
    "markdown-it-py>=3.0" "langchain-text-splitters>=0.0.2" \
    "ray[default]>=2.10"

ENV PYTHONPATH=/opt/airflow/project/src
ENV RAG_PROJECT_ROOT=/opt/airflow/project
