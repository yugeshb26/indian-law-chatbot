"""
Gemini streaming engine with API key rotation, retry logic, and auto-continue.
"""

import time
import threading
from google import genai
from google.genai.types import GenerateContentConfig

# ── Constants ────────────────────────────────────────────────────────────────
MODEL = "gemini-3-flash-preview"
MAX_RETRIES = 3
RETRY_DELAY = 2
MAX_OUTPUT_TOKENS = 4096
CONTINUE_PROMPT = "Continue your response exactly where you left off. Do not repeat what you already said."


# ── API Key Rotator ──────────────────────────────────────────────────────────

class KeyRotator:
    """Round-robin API key rotation with automatic failover on rate limit."""

    def __init__(self, keys: list[str]):
        self._keys = [k for k in keys if k and len(k) > 10]
        if not self._keys:
            raise ValueError("No valid API keys provided")
        self._index = 0
        self._lock = threading.Lock()
        self._failed = {}  # key -> timestamp when it was rate-limited

    @property
    def count(self) -> int:
        return len(self._keys)

    def get_key(self) -> str:
        """Get the next available key, skipping recently rate-limited ones."""
        with self._lock:
            now = time.time()
            # Try each key once
            for _ in range(len(self._keys)):
                key = self._keys[self._index]
                self._index = (self._index + 1) % len(self._keys)

                # Skip if rate-limited in last 60 seconds
                if key in self._failed and now - self._failed[key] < 60:
                    continue
                return key

            # All keys rate-limited — return least recently failed
            oldest_key = min(self._failed, key=self._failed.get, default=self._keys[0])
            return oldest_key

    def mark_failed(self, key: str):
        """Mark a key as rate-limited."""
        with self._lock:
            self._failed[key] = time.time()

    def mark_success(self, key: str):
        """Clear rate-limit flag on success."""
        with self._lock:
            self._failed.pop(key, None)


# ── Global rotator (set by Chatbot.py) ───────────────────────────────────────
_rotator: KeyRotator | None = None


def init_rotator(keys: list[str]):
    """Initialize the key rotator. Called once from Chatbot.py."""
    global _rotator
    _rotator = KeyRotator(keys)
    return _rotator


def _get_rotator() -> KeyRotator:
    if _rotator is None:
        raise RuntimeError("Key rotator not initialized. Call init_rotator() first.")
    return _rotator


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_config():
    return GenerateContentConfig(
        temperature=0.5,
        top_p=0.85,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )


# ── Streaming with key rotation ──────────────────────────────────────────────

def stream_response(api_key_unused: str, system_prompt: str, messages: list[dict]):
    """
    Yield text chunks from Gemini streaming API.
    Rotates API keys on rate limit errors.
    """
    rotator = _get_rotator()
    config = _build_config()

    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    full_text = ""
    attempt = 0

    while attempt < MAX_RETRIES:
        key = rotator.get_key()
        client = genai.Client(api_key=key)

        try:
            stream = client.models.generate_content_stream(
                model=MODEL,
                contents=contents,
                config=config,
            )
            for chunk in stream:
                if chunk.text:
                    full_text += chunk.text
                    yield chunk.text

            rotator.mark_success(key)

            if _seems_complete(full_text):
                return
            else:
                contents.append({"role": "model", "parts": [{"text": full_text}]})
                contents.append({"role": "user", "parts": [{"text": CONTINUE_PROMPT}]})
                yield "\n"
                attempt += 1
                continue

        except Exception as e:
            err = str(e).lower()
            if "429" in err or "resource_exhausted" in err or "quota" in err:
                rotator.mark_failed(key)
                attempt += 1
                time.sleep(1)  # Quick switch to next key
                continue
            else:
                attempt += 1
                if attempt >= MAX_RETRIES:
                    if full_text:
                        return
                    raise RuntimeError(f"Gemini API failed: {str(e)[:200]}")
                time.sleep(RETRY_DELAY * attempt)
                continue


def _seems_complete(text: str) -> bool:
    text = text.rstrip()
    if not text:
        return False
    if text[-1] in ".!?:;)\"]}>*`~":
        return True
    if text.endswith("\n"):
        return True
    last_line = text.split("\n")[-1]
    if len(last_line) > 200 and text[-1].isalpha():
        return False
    return True


# ── Non-streaming (regenerate) ───────────────────────────────────────────────

def regenerate_response(api_key_unused: str, system_prompt: str, messages: list[dict]) -> str:
    rotator = _get_rotator()
    config = _build_config()

    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    for attempt in range(MAX_RETRIES):
        key = rotator.get_key()
        client = genai.Client(api_key=key)
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=config,
            )
            if resp and resp.text:
                rotator.mark_success(key)
                return resp.text
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                rotator.mark_failed(key)
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(RETRY_DELAY * (attempt + 1))

    raise RuntimeError("Failed to generate response")


# ── Title generator ──────────────────────────────────────────────────────────

def generate_title(api_key_unused: str, question: str) -> str:
    rotator = _get_rotator()
    key = rotator.get_key()
    client = genai.Client(api_key=key)
    try:
        resp = client.models.generate_content(
            model=MODEL,
            contents=[{
                "role": "user",
                "parts": [{"text": f"Generate a very short title (max 5 words, no quotes) for a chat that starts with this question: {question}"}],
            }],
            config=GenerateContentConfig(temperature=0.3, max_output_tokens=20),
        )
        if resp and resp.text:
            rotator.mark_success(key)
            return resp.text.strip().strip('"').strip("'")[:50]
    except Exception:
        rotator.mark_failed(key)
    return question[:40] + ("..." if len(question) > 40 else "")
