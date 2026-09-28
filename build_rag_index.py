"""
CLI entry point: (re)build the semantic-search index for the legal dataset.

Usage:
    python build_rag_index.py

Reads GEMINI_API_KEYS the same way Chatbot.py does — a JSON list in the
environment variable first, falling back to .streamlit/secrets.toml for
local runs. Safe to run after every daily_update_rss.py run: it only
embeds dataset entries that aren't already indexed (see rag_index.py).
"""

import json
import os

from rag_index import build_or_update_index

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "Alpie-core_core_indian_law.json")


def _get_api_key() -> str:
    if os.environ.get("GEMINI_API_KEYS"):
        keys = json.loads(os.environ["GEMINI_API_KEYS"])
        if keys:
            return keys[0]

    secrets_path = os.path.join(BASE_DIR, ".streamlit", "secrets.toml")
    if os.path.exists(secrets_path):
        try:
            import tomllib as _toml  # Python 3.11+
        except ModuleNotFoundError:
            import tomli as _toml  # fallback for older Python
        with open(secrets_path, "rb") as f:
            secrets = _toml.load(f)
        keys = secrets.get("GEMINI_API_KEYS", [])
        if keys:
            return keys[0]

    raise RuntimeError(
        "No GEMINI_API_KEYS found. Set the GEMINI_API_KEYS env var (JSON list) "
        "or add it to .streamlit/secrets.toml."
    )


if __name__ == "__main__":
    build_or_update_index(DATASET_PATH, _get_api_key())
