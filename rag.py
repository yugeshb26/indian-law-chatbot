"""
RAG retrieval for Indian Law Chatbot.

Retrieval strategy (in priority order):
  1. Vector search  — if rag_embeddings.npy + rag_meta.json exist (pre-built)
     Embeds user query with Gemini text-embedding-004, cosine similarity
     via numpy dot product on normalised float16 vectors.  Fast even for
     15 K entries (~2 ms CPU).

  2. BM25 fallback  — rank_bm25 keyword index built in-memory on first call.
     Good for legal queries that use specific section/act names.  No API
     calls, no pre-built files needed.

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

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE     = os.path.dirname(os.path.abspath(__file__))
_EMB_PATH = os.path.join(_BASE, "rag_embeddings.npy")
_META_PATH = os.path.join(_BASE, "rag_meta.json")
_DATASET  = os.path.join(_BASE, "Alpie-core_core_indian_law.json")


def index_ready() -> bool:
    """True when the pre-built vector index files are present."""
    return os.path.exists(_EMB_PATH) and os.path.exists(_META_PATH)


# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading legal knowledge base …")
def _load_vector_index():
    """Load embeddings matrix and metadata once per process."""
    embs = np.load(_EMB_PATH).astype(np.float32)   # (N, 768)
    with open(_META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    return embs, meta


@st.cache_resource(show_spinner="Building keyword index …")
def _load_bm25_index():
    """Build BM25 index from the full dataset (fallback path)."""
    from rank_bm25 import BM25Okapi

    with open(_DATASET, encoding="utf-8") as f:
        data = json.load(f)

    # Tokenise on whitespace (fast; legal terms are mostly single tokens)
    corpus = [
        (d["prompt"] + " " + d["response"][:300]).lower().split()
        for d in data
    ]
    bm25 = BM25Okapi(corpus)
    meta = [{"prompt": d["prompt"], "response": d["response"][:600]} for d in data]
    return bm25, meta


# ── Embedding helper (runtime, one call per user query) ───────────────────────

def _embed_query(text: str, api_key: str) -> np.ndarray:
    """Return a normalised float32 query vector, shape (1, 768)."""
    from google import genai

    client = genai.Client(api_key=api_key)
    resp = client.models.embed_content(
        model="models/text-embedding-004",
        contents=[text],
    )
    vec = np.array(resp.embeddings[0].values, dtype=np.float32)
    norm = np.linalg.norm(vec)
    vec = vec / max(norm, 1e-9)
    return vec.reshape(1, -1)


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve(query: str, api_key: str, k: int = 8) -> list[dict]:
    """Return the top-k most relevant Q&A entries for *query*.

    Falls back to BM25 if the vector index has not been built yet or if
    the Gemini embedding call fails.
    """
    if not query.strip():
        return []

    # ── Vector path ───────────────────────────────────────────────────────────
    if index_ready():
        try:
            embs, meta = _load_vector_index()
            q_vec = _embed_query(query, api_key)          # (1, 768)
            scores = np.dot(embs, q_vec.T).squeeze()      # (N,)
            top_k = int(np.minimum(k, len(meta)))
            idx = np.argpartition(scores, -top_k)[-top_k:]
            idx = idx[np.argsort(scores[idx])[::-1]]      # sort desc
            return [meta[i] for i in idx]
        except Exception:
            pass  # fall through to BM25

    # ── BM25 fallback ─────────────────────────────────────────────────────────
    try:
        bm25, meta = _load_bm25_index()
        tokens = query.lower().split()
        scores = bm25.get_scores(tokens)
        top_k = int(np.minimum(k, len(meta)))
        idx = np.argpartition(scores, -top_k)[-top_k:]
        idx = idx[np.argsort(scores[idx])[::-1]]
        return [meta[i] for i in idx]
    except Exception:
        return []


def format_context(results: list[dict]) -> str:
    """Format retrieved results as a context block for the system prompt."""
    if not results:
        return ""
    parts = [
        f"Q: {r['prompt']}\nA: {r['response']}"
        for r in results
    ]
    return "\n\n---\n\n".join(parts)
