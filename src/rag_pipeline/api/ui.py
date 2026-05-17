"""Streamlit search UI. Run with: streamlit run src/rag_pipeline/api/ui.py"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")


def main() -> None:
    st.set_page_config(page_title="RAG Pipeline Search", layout="wide")
    st.title("Hybrid Semantic Search")
    st.caption("Dense (sentence-transformers) + Sparse (BM25), fused via RRF.")

    with st.sidebar:
        backend = st.radio("Vector backend", ["faiss", "qdrant"], index=0)
        k = st.slider("Top-K", 1, 25, 10)
        source_type = st.selectbox(
            "Filter by source type", [None, "pdf", "md", "txt", "html"], index=0
        )
        if st.button("Trigger re-ingest"):
            with st.spinner("Running pipeline..."):
                r = httpx.post(f"{API_URL}/ingest", params={"backend": backend}, timeout=600)
                r.raise_for_status()
                st.success(r.json())

    query = st.text_input("Query", placeholder="What is reciprocal rank fusion?")
    if not query:
        return

    params = {"q": query, "k": k, "backend": backend}
    if source_type:
        params["source_type"] = source_type
    try:
        resp = httpx.get(f"{API_URL}/search", params=params, timeout=60)
        resp.raise_for_status()
    except Exception as exc:
        st.error(f"Search failed: {exc}")
        return

    hits = resp.json()["hits"]
    st.write(f"**{len(hits)} hits**")
    for i, h in enumerate(hits, 1):
        with st.expander(
            f"{i}. {h['source_path'].split('/')[-1]} · RRF={h['score']:.4f} "
            f"(dense={h['dense_rank']}, sparse={h['sparse_rank']})"
        ):
            st.caption(f"`{h['chunk_id']}` · {h['source_type']}")
            st.write(h["text"])


if __name__ == "__main__":
    main()
