"""
RAG retrieval for Indian Law Chatbot.

Retrieval strategy (in priority order):
  1. Vector search  — if rag_embeddings.npy + rag_meta.json exist (pre-built
     by build_rag_index.py using all-MiniLM-L6-v2, 384-dim).
     Query embedded locally (no API calls), cosine search via numpy dot
     product on normalised float16 vectors. Fast even for 15K entries (~2ms).

  2. BM25 fallback  — rank_bm25 keyword index built in-memory on first call.
     Works without any pre-built files. Good for section/act name queries.

Public API
----------
  retrieve(query, api_key, k=8)  → list[dict]   (prompt, response keys)
  format_context(results)        → str           (ready for system prompt)
  index_ready()                  → bool          (True when vector index found)
"""

import json
import os

import numpy as np
import streamlit as st

_BASE      = os.path.dirname(os.path.abspath(__file__))
_EMB_PATH  = os.path.join(_BASE, "rag_embeddings.npy")
_META_PATH = os.path.join(_BASE, "rag_meta.json")
_DATASET   = os.path.join(_BASE, "Alpie-core_core_indian_law.json")
_MODEL     = "all-MiniLM-L6-v2"


def index_ready() -> bool:
    return os.path.exists(_EMB_PATH) and os.path.exists(_META_PATH)


# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading legal knowledge base …")
def _load_vector_index():
    embs = np.load(_EMB_PATH).astype(np.float32)
    with open(_META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    return embs, meta


@st.cache_resource(show_spinner="Loading embedding model …")
def _load_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(_MODEL)


@st.cache_resource(show_spinner="Building keyword index …")
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


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve(query: str, api_key: str, k: int = 8) -> list[dict]:
    """Return top-k most relevant Q&A entries for *query*."""
    if not query.strip():
        return []

    # ── Vector search ─────────────────────────────────────────────────────────
    if index_ready():
        try:
            embs, meta = _load_vector_index()
            encoder    = _load_encoder()
            q_vec = encoder.encode(
                [query], normalize_embeddings=True, convert_to_numpy=True
            )                                             # (1, 384)
            scores = np.dot(embs, q_vec.T).squeeze()     # (N,)
            top_k  = min(k, len(meta))
            idx    = np.argpartition(scores, -top_k)[-top_k:]
            idx    = idx[np.argsort(scores[idx])[::-1]]
            return [meta[i] for i in idx]
        except Exception:
            pass

    # ── BM25 fallback ─────────────────────────────────────────────────────────
    try:
        bm25, meta = _load_bm25_index()
        scores = bm25.get_scores(query.lower().split())
        top_k  = min(k, len(meta))
        idx    = np.argpartition(scores, -top_k)[-top_k:]
        idx    = idx[np.argsort(scores[idx])[::-1]]
        return [meta[i] for i in idx]
    except Exception:
        return []


def format_context(results: list[dict]) -> str:
    if not results:
        return ""
    return "\n\n---\n\n".join(
        f"Q: {r['prompt']}\nA: {r['response']}" for r in results
    )
