"""
Semantic-search index for the legal Q&A dataset, built on Gemini's embedding
API (models/text-embedding-004) instead of the old in-prompt keyword dump.

Files on disk:
  rag_embeddings.npy — float16 array, shape (N, EMBED_DIM), L2-normalized
                        rows, in the same order as rag_meta.json.
  rag_meta.json       — list of {"prompt", "response", "hash"} dicts, one
                        per embedding row. `hash` is sha1(prompt+response),
                        used to detect which dataset entries already have
                        an embedding so rebuilds only embed what changed.

build_or_update_index() is incremental: it re-embeds only entries that are
new since the last run, and drops rows for entries no longer present (e.g.
evicted by daily_update_rss.py's MAX_TOTAL_ENTRIES cap) — a full 40k-entry
dataset should only cost a handful of new embedding calls per day.
"""

import hashlib
import json
import os
import time

import numpy as np
from google import genai
from google.genai.types import EmbedContentConfig

EMBED_MODEL = "models/text-embedding-004"
EMBED_DIM = 768
BATCH_SIZE = 100
MAX_RETRIES = 4
RESPONSE_CHARS_FOR_EMBEDDING = 2000  # keep embedding requests small/cheap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "rag_embeddings.npy")
META_PATH = os.path.join(BASE_DIR, "rag_meta.json")


def _entry_hash(item: dict) -> str:
    raw = (item.get("prompt", "") + "\x00" + item.get("response", "")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def _embed_batch(client: "genai.Client", texts: list[str], task_type: str) -> list[list[float]]:
    """Embed a batch of texts, retrying on transient API errors."""
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.models.embed_content(
                model=EMBED_MODEL,
                contents=texts,
                config=EmbedContentConfig(task_type=task_type, output_dimensionality=EMBED_DIM),
            )
            return [e.values for e in resp.embeddings]
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 * (attempt + 1))
    raise last_err


def embed_query(client: "genai.Client", text: str) -> np.ndarray:
    """Embed a single user question for retrieval (normalized, float32)."""
    vecs = _embed_batch(client, [text], task_type="RETRIEVAL_QUERY")
    return _normalize(np.array(vecs, dtype=np.float32))[0]


def load_index() -> tuple[np.ndarray | None, list[dict] | None]:
    """Load the on-disk index. Returns (embeddings, meta), or (None, None)
    if the files are missing or inconsistent."""
    if not (os.path.exists(EMBEDDINGS_PATH) and os.path.exists(META_PATH)):
        return None, None
    try:
        embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return None, None
    if embeddings.ndim != 2 or embeddings.shape[0] != len(meta) or embeddings.shape[1] != EMBED_DIM:
        return None, None
    return embeddings, meta


def semantic_search(
    client: "genai.Client",
    query: str,
    embeddings: np.ndarray,
    meta: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """Return the top_k most semantically similar dataset entries to `query`."""
    if embeddings is None or not meta:
        return []
    q = embed_query(client, query)
    sims = embeddings @ q  # cosine similarity: both sides are L2-normalized
    top_idx = np.argsort(-sims)[:top_k]
    return [meta[i] for i in top_idx]


def build_or_update_index(dataset_path: str, api_key: str, verbose: bool = True) -> None:
    """Incrementally (re)build rag_embeddings.npy / rag_meta.json from the
    current dataset at `dataset_path`."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    hashes = [_entry_hash(item) for item in dataset]
    current_hashes = set(hashes)

    old_embeddings, old_meta = load_index()
    old_by_hash: dict[str, tuple[np.ndarray, dict]] = {}
    if old_meta:
        for row, meta_item in zip(old_embeddings, old_meta):
            h = meta_item.get("hash")
            if h in current_hashes:
                old_by_hash[h] = (row, meta_item)

    new_items = [(item, h) for item, h in zip(dataset, hashes) if h not in old_by_hash]
    if verbose:
        print(
            f"Dataset entries: {len(dataset)} | already indexed: {len(old_by_hash)} "
            f"| to embed: {len(new_items)}"
        )

    new_by_hash: dict[str, tuple[list[float], dict]] = {}
    if new_items:
        client = genai.Client(api_key=api_key)
        for i in range(0, len(new_items), BATCH_SIZE):
            batch = new_items[i : i + BATCH_SIZE]
            texts = [
                f"{item['prompt']}\n{item.get('response', '')[:RESPONSE_CHARS_FOR_EMBEDDING]}"
                for item, _ in batch
            ]
            vecs = _embed_batch(client, texts, task_type="RETRIEVAL_DOCUMENT")
            for (item, h), vec in zip(batch, vecs):
                new_by_hash[h] = (
                    vec,
                    {"prompt": item.get("prompt", ""), "response": item.get("response", ""), "hash": h},
                )
            if verbose:
                print(f"  embedded {min(i + BATCH_SIZE, len(new_items))}/{len(new_items)}")

    all_rows = []
    all_meta = []
    for h in hashes:
        if h in old_by_hash:
            row, meta_item = old_by_hash[h]
        else:
            row, meta_item = new_by_hash[h]
        all_rows.append(row)
        all_meta.append(meta_item)

    embeddings = _normalize(np.array(all_rows, dtype=np.float32)).astype(np.float16)
    np.save(EMBEDDINGS_PATH, embeddings)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False)

    if verbose:
        print(f"Saved {embeddings.shape[0]} embeddings (dim={embeddings.shape[1]}) to {EMBEDDINGS_PATH}")
