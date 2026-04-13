"""
Build the RAG vector index for the Indian Law dataset.

Uses a local sentence-transformers model — no API calls, no rate limits.
Model: all-MiniLM-L6-v2 (22 MB, 384-dim, ~3 min for 15K entries on CPU)

Run once locally:
    python3 build_rag_index.py

Writes:
    rag_embeddings.npy   — float16 matrix, shape (N, 384)
    rag_meta.json        — list of {prompt, response} dicts

Commit both files to git and deploy.
"""

import json
import os
import numpy as np

BASE      = os.path.dirname(os.path.abspath(__file__))
DATASET   = os.path.join(BASE, "Alpie-core_core_indian_law.json")
EMB_PATH  = os.path.join(BASE, "rag_embeddings.npy")
META_PATH = os.path.join(BASE, "rag_meta.json")

BATCH_SIZE   = 256
RESP_MAXLEN  = 600
MODEL_NAME   = "all-MiniLM-L6-v2"


def main():
    from sentence_transformers import SentenceTransformer

    print("Loading dataset …")
    with open(DATASET, encoding="utf-8") as f:
        data = json.load(f)
    n = len(data)
    print(f"  {n} entries")

    prompts = [d["prompt"] for d in data]

    # Resume support
    if os.path.exists(EMB_PATH):
        existing = np.load(EMB_PATH)
        done = existing.shape[0]
        all_embs = list(existing)
        print(f"Resuming from {done}/{n}")
    else:
        done, all_embs = 0, []

    if done < n:
        print(f"Loading model {MODEL_NAME} …")
        model = SentenceTransformer(MODEL_NAME)

        print(f"Embedding {n - done} remaining entries …")
        remaining = prompts[done:]
        vecs = model.encode(
            remaining,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,   # cosine via dot product
            convert_to_numpy=True,
        )
        all_embs.extend(vecs.astype(np.float16))

    matrix = np.array(all_embs, dtype=np.float16)
    np.save(EMB_PATH, matrix)
    emb_mb = os.path.getsize(EMB_PATH) / 1e6
    print(f"\nSaved {EMB_PATH}  shape={matrix.shape}  ({emb_mb:.1f} MB)")

    meta = [
        {"prompt": d["prompt"], "response": d["response"][:RESP_MAXLEN]}
        for d in data
    ]
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, separators=(",", ":"))
    meta_mb = os.path.getsize(META_PATH) / 1e6
    print(f"Saved {META_PATH}  ({meta_mb:.1f} MB)")
    print(f"\nDone! Commit rag_embeddings.npy + rag_meta.json and deploy.")


if __name__ == "__main__":
    main()
