"""
Build the RAG vector index for the Indian Law dataset.

Run once (locally, before deploying):
    python3 build_rag_index.py

Reads  : Alpie-core_core_indian_law.json (15 K entries)
Writes : rag_embeddings.npy   — float16 matrix, shape (N, 768)
         rag_meta.json        — list of {prompt, response} dicts (truncated)

Uses Gemini text-embedding-004 (768-dim).
Supports resume: re-run if interrupted and it picks up where it left off.
Total runtime: ~3-5 min for 15 K entries on free-tier API.
"""

import json
import os
import sys
import time

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATASET  = os.path.join(BASE, "Alpie-core_core_indian_law.json")
EMB_PATH = os.path.join(BASE, "rag_embeddings.npy")
META_PATH = os.path.join(BASE, "rag_meta.json")

BATCH_SIZE   = 100    # texts per API call (Gemini limit)
DIM          = 768    # text-embedding-004 output dim
RESP_MAXLEN  = 600    # truncate responses in metadata to keep file small


# ── API key ───────────────────────────────────────────────────────────────────
def _get_key() -> str:
    # 1. Command-line arg
    if len(sys.argv) > 1:
        return sys.argv[1]
    # 2. Streamlit secrets file
    secrets_path = os.path.join(BASE, ".streamlit", "secrets.toml")
    if os.path.exists(secrets_path):
        with open(secrets_path) as f:
            for line in f:
                if "GEMINI_API_KEYS" in line:
                    import re
                    keys = re.findall(r'"([^"]+)"', line)
                    if keys:
                        return keys[0]
    # 3. Prompt
    return input("Gemini API key: ").strip()


# ── Embed a batch of texts ────────────────────────────────────────────────────
def _embed_batch(client, texts: list[str]) -> np.ndarray:
    """Return normalised float32 embeddings, shape (len(texts), DIM)."""
    resp = client.models.embed_content(
        model="models/text-embedding-004",
        contents=texts,
    )
    vecs = np.array([e.values for e in resp.embeddings], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return (vecs / np.maximum(norms, 1e-9)).astype(np.float16)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    from google import genai

    api_key = _get_key()
    client  = genai.Client(api_key=api_key)

    print("Loading dataset …")
    with open(DATASET, encoding="utf-8") as f:
        data = json.load(f)
    n = len(data)
    print(f"  {n} entries")

    prompts = [d["prompt"] for d in data]

    # ── Resume support ────────────────────────────────────────────────────────
    if os.path.exists(EMB_PATH):
        existing = np.load(EMB_PATH)
        done = existing.shape[0]
        all_embs = list(existing)
        print(f"Resuming from {done}/{n}")
    else:
        done, all_embs = 0, []

    # ── Embed remaining ───────────────────────────────────────────────────────
    if done < n:
        print(f"Embedding {n - done} remaining entries …")
        for start in range(done, n, BATCH_SIZE):
            batch = prompts[start : start + BATCH_SIZE]
            while True:
                try:
                    vecs = _embed_batch(client, batch)
                    all_embs.extend(vecs)
                    break
                except Exception as e:
                    err = str(e).lower()
                    wait = 30 if ("429" in err or "quota" in err) else 5
                    print(f"\n  Error at {start}: {e!s:.80} — retrying in {wait}s …")
                    time.sleep(wait)

            end = min(start + BATCH_SIZE, n)
            print(f"  {end}/{n} ({end*100//n}%)", end="\r", flush=True)

            # Save progress every 1 000 entries
            if len(all_embs) % 1000 < BATCH_SIZE:
                np.save(EMB_PATH, np.array(all_embs, dtype=np.float16))

            time.sleep(0.15)   # gentle rate-limit buffer

    # ── Save final embeddings ─────────────────────────────────────────────────
    matrix = np.array(all_embs, dtype=np.float16)
    np.save(EMB_PATH, matrix)
    print(f"\nSaved {EMB_PATH}  shape={matrix.shape}")

    # ── Save metadata ─────────────────────────────────────────────────────────
    meta = [
        {
            "prompt":   d["prompt"],
            "response": d["response"][:RESP_MAXLEN],
        }
        for d in data
    ]
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, separators=(",", ":"))
    size_mb = os.path.getsize(META_PATH) / 1e6
    print(f"Saved {META_PATH}  ({size_mb:.1f} MB)")

    emb_mb = os.path.getsize(EMB_PATH) / 1e6
    print(f"\nDone! Index size: {emb_mb:.1f} MB (embeddings) + {size_mb:.1f} MB (meta)")
    print("Commit both files to git and deploy.")


if __name__ == "__main__":
    main()
