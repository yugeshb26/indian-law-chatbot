"""
Gemini streaming engine with API key rotation, retry logic, and auto-continue.
"""

import time
import threading
from google import genai
from google.genai.types import GenerateContentConfig

# ── Constants ────────────────────────────────────────────────────────────────
MODEL = "gemini-3.6-flash"
MAX_RETRIES = 6      # more retries across keys
MAX_CONTINUE = 3     # max auto-continue iterations
RETRY_DELAY = 1      # base delay
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

            # All keys rate-limited — return least recently failed and reset index past it
            oldest_key = min(self._failed, key=self._failed.get, default=self._keys[0])
            if oldest_key in self._keys:
                self._index = (self._keys.index(oldest_key) + 1) % len(self._keys)
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
    api_attempts = 0        # counts only actual API errors (rate-limit or other exceptions)
    continue_count = 0      # counts successful-but-incomplete auto-continuations
    last_error = None
    last_error_is_quota = False

    while api_attempts < MAX_RETRIES:
        key = rotator.get_key()
        client = genai.Client(api_key=key)
        
        print(f"[DEBUG] API Attempt {api_attempts+1} using key ending in ...{key[-4:]}")

        try:
            stream = client.models.generate_content_stream(
                model=MODEL,
                contents=contents,
                config=config,
            )
            
            yielded_in_this_pass = False
            for chunk in stream:
                if chunk and chunk.text:
                    full_text += chunk.text
                    yield chunk.text
                    yielded_in_this_pass = True

            rotator.mark_success(key)
            last_error = None
            last_error_is_quota = False

            if _seems_complete(full_text) or continue_count >= MAX_CONTINUE:
                return

            # Response is incomplete, try to continue
            print(f"[DEBUG] Response looks incomplete, auto-continuing (iteration {continue_count+1})")
            contents.append({"role": "model", "parts": [{"text": full_text}]})
            contents.append({"role": "user", "parts": [{"text": CONTINUE_PROMPT}]})
            yield "\n"
            continue_count += 1
            continue

        except Exception as e:
            err_msg = str(e).lower()
            last_error = e
            api_attempts += 1
            
            # Check for quota vs rate limit
            is_quota = "quota" in err_msg or "429" in err_msg or "resource_exhausted" in err_msg
            if is_quota:
                last_error_is_quota = True
                rotator.mark_failed(key)
                print(f"[WARN] Key ...{key[-4:]} hit rate limit: {err_msg[:100]}")
            else:
                last_error_is_quota = False
                print(f"[ERROR] Key ...{key[-4:]} encountered error: {err_msg[:100]}")

            if full_text:
                # We have partial response. Instead of restarting (which creates duplicates),
                # we treat this as a cut-off and try to "continue" if we have retries left.
                if continue_count < MAX_CONTINUE and api_attempts < MAX_RETRIES:
                    print(f"[DEBUG] Mid-stream failure. Contextualizing partial response and retrying...")
                    # Update contents to include what we got so far as if it were a completed turn
                    # This tells the model where to pick up from.
                    contents = [{"role": "user", "parts": [{"text": messages[0]["content"]}]}] # reset context
                    contents.append({"role": "model", "parts": [{"text": full_text}]})
                    contents.append({"role": "user", "parts": [{"text": CONTINUE_PROMPT}]})
                    # Use a fresh key for the continuation
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    # Out of continues or retries, return what we have
                    return

            # No text yielded yet, safe to retry with another key
            if api_attempts < MAX_RETRIES:
                wait = RETRY_DELAY * api_attempts
                print(f"[DEBUG] Retrying full request in {wait}s...")
                time.sleep(wait)
                continue
            break

    # All API error retries exhausted
    if full_text:
        return

    if last_error_is_quota:
        raise RuntimeError(
            "API_QUOTA_EXHAUSTED: All Gemini API keys have hit their rate limit or daily quota. "
            "This could be a temporary traffic spike (wait 60s) or daily exhaustion."
        )
    
    err_str = str(last_error) if last_error else "Unknown error"
    raise RuntimeError(f"API_FAILURE: Gemini API failed after {MAX_RETRIES} retries: {err_str[:200]}")


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
