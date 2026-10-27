"""
Groq fallback engine. Gemini stays primary for both chat generation and
the embedding-based semantic search (rag_index.py is built specifically on
Gemini's embedding API) — this module only ever runs when every Gemini key
is simultaneously rate-limited or quota-exhausted (see gemini_engine.py's
stream_response()). Purely a resilience layer against a Gemini-wide outage
or daily-quota exhaustion, not a replacement.
"""

import time

from groq import Groq

from key_rotator import KeyRotator
from gemini_engine import _seems_complete, CONTINUE_PROMPT, MAX_CONTINUE

# Verified against Groq's current production model list (console.groq.com/docs/models)
# as of this integration — re-check there if this ever needs to change.
MODEL = "openai/gpt-oss-120b"
MAX_RETRIES = 3
RETRY_DELAY = 1
MAX_OUTPUT_TOKENS = 4096

_rotator: KeyRotator | None = None


def init_groq_rotator(keys: list[str]):
    """Initialize the Groq key rotator. Safe to call with an empty/missing
    list — the fallback simply stays unconfigured (is_configured() -> False)
    rather than erroring, since Groq is optional."""
    global _rotator
    valid_keys = [k for k in keys if k and len(k) > 10]
    if not valid_keys:
        _rotator = None
        return None
    _rotator = KeyRotator(valid_keys)
    return _rotator


def is_configured() -> bool:
    return _rotator is not None


def _to_groq_messages(messages: list[dict]) -> list[dict]:
    """Convert the same message list Gemini received into OpenAI-style
    roles. The system prompt is already folded into the first user
    message's content by Chatbot.py (gemini_engine.stream_response ignores
    its own `system_prompt` parameter for the same reason) — so this adds
    no separate system message, keeping parity with what Gemini actually
    saw for this turn."""
    out = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "assistant"
        out.append({"role": role, "content": msg["content"]})
    return out


def stream_groq_response(messages: list[dict]):
    """Yield text chunks from Groq's streaming API, mirroring
    gemini_engine.stream_response's retry/auto-continue behavior."""
    if _rotator is None:
        raise RuntimeError("GROQ_NOT_CONFIGURED: No Groq API keys configured for fallback.")

    contents = _to_groq_messages(messages)
    full_text = ""
    api_attempts = 0
    continue_count = 0
    last_error = None

    while api_attempts < MAX_RETRIES:
        key = _rotator.get_key()
        client = Groq(api_key=key)
        print(f"[DEBUG] Groq fallback attempt {api_attempts + 1} using key ending in ...{key[-4:]}")

        try:
            stream = client.chat.completions.create(
                model=MODEL,
                messages=contents,
                temperature=0.5,
                top_p=0.85,
                max_completion_tokens=MAX_OUTPUT_TOKENS,
                stream=True,
            )

            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    full_text += delta
                    yield delta

            _rotator.mark_success(key)

            if _seems_complete(full_text) or continue_count >= MAX_CONTINUE:
                return

            contents.append({"role": "assistant", "content": full_text})
            contents.append({"role": "user", "content": CONTINUE_PROMPT})
            yield "\n"
            continue_count += 1
            continue

        except Exception as e:
            err_msg = str(e).lower()
            last_error = e
            api_attempts += 1
            is_quota = "429" in err_msg or "rate" in err_msg or "quota" in err_msg
            if is_quota:
                _rotator.mark_failed(key)
                print(f"[WARN] Groq key ...{key[-4:]} hit rate limit: {err_msg[:100]}")
            else:
                print(f"[ERROR] Groq key ...{key[-4:]} encountered error: {err_msg[:100]}")

            if full_text:
                # Keep whatever we got rather than restart and duplicate text.
                return

            if api_attempts < MAX_RETRIES:
                time.sleep(RETRY_DELAY * api_attempts)
                continue
            break

    if full_text:
        return
    raise RuntimeError(f"GROQ_FAILURE: Groq fallback failed after {MAX_RETRIES} retries: {str(last_error)[:200]}")
