"""
RAG retrieval — BM25 keyword search over Alpie-core_core_indian_law.json.
No embeddings, no model downloads, no external API calls.
"""

import json
import os

import numpy as np
import streamlit as st

_BASE    = os.path.dirname(os.path.abspath(__file__))
_DATASET = os.path.join(_BASE, "Alpie-core_core_indian_law.json")


def index_ready() -> bool:
    return False  # BM25-only; vector index not used


@st.cache_resource(show_spinner="Loading legal knowledge base …")
def _load_bm25_index():
    from rank_bm25 import BM25Okapi
    with open(_DATASET, encoding="utf-8") as f:
        data = json.load(f)
    corpus = [
        (d["prompt"] + " " + d["response"][:300]).lower().split()
        for d in data
    ]
    meta = [{"prompt": d["prompt"], "response": d["response"][:600]} for d in data]
    return BM25Okapi(corpus), meta


def retrieve(query: str, api_key: str, k: int = 8) -> list[dict]:
    if not query.strip():
        return []
    try:
        bm25, meta = _load_bm25_index()
        tokens = query.lower().split()
        scores = bm25.get_scores(tokens)
        top_k  = min(k, len(meta))
        idx    = np.argpartition(scores, -top_k)[-top_k:]
        idx    = idx[np.argsort(scores[idx])[::-1]]
        return [meta[i] for i in idx if scores[i] > 0]
    except Exception:
        return []


def format_context(results: list[dict]) -> str:
    if not results:
        return ""
    return "\n\n---\n\n".join(
        f"Q: {r['prompt']}\nA: {r['response']}" for r in results
    )
