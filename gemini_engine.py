"""
Gemini streaming engine with retry logic, auto-continue, and token management.
"""

import time
from google import genai
from google.genai.types import GenerateContentConfig

# ── Constants ────────────────────────────────────────────────────────────────
MODEL = "gemini-2.5-flash"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds, doubles each retry
MAX_OUTPUT_TOKENS = 4096
CONTINUE_PROMPT = "Continue your response exactly where you left off. Do not repeat what you already said."


def _build_client(api_key: str):
    return genai.Client(api_key=api_key)


def _build_config():
    return GenerateContentConfig(
        temperature=0.5,
        top_p=0.85,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )


# ── Streaming generation with retry ─────────────────────────────────────────

def stream_response(api_key: str, system_prompt: str, messages: list[dict]):
    """
    Yield text chunks from Gemini streaming API.
    - messages: list of {"role": "user"/"assistant", "content": "..."}
    - Retries on failure with exponential backoff
    - Auto-continues if response appears truncated
    """
    client = _build_client(api_key)
    config = _build_config()

    # Build Gemini contents from chat history
    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    full_text = ""
    attempt = 0

    while attempt < MAX_RETRIES:
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
            # Success — check if response seems complete
            if _seems_complete(full_text):
                return
            else:
                # Auto-continue: append current response and ask to continue
                contents.append({"role": "model", "parts": [{"text": full_text}]})
                contents.append({"role": "user", "parts": [{"text": CONTINUE_PROMPT}]})
                yield "\n"  # visual separator
                attempt += 1
                continue

        except Exception as e:
            attempt += 1
            if attempt >= MAX_RETRIES:
                if full_text:
                    return  # return what we have
                raise RuntimeError(f"Gemini API failed after {MAX_RETRIES} retries: {str(e)[:200]}")
            time.sleep(RETRY_DELAY * attempt)
            continue


def _seems_complete(text: str) -> bool:
    """Heuristic: check if response ended naturally (not mid-sentence)."""
    text = text.rstrip()
    if not text:
        return False
    # Ended with sentence-ending punctuation or list/code block
    if text[-1] in ".!?:;)\"]}>*`~":
        return True
    # Ended with a newline (lists, code blocks)
    if text.endswith("\n"):
        return True
    # Looks cut off mid-word (less than 100 chars in last chunk is suspicious)
    last_line = text.split("\n")[-1]
    if len(last_line) > 200 and text[-1].isalpha():
        return False  # likely truncated
    return True


# ── Non-streaming fallback (for regenerate) ──────────────────────────────────

def regenerate_response(api_key: str, system_prompt: str, messages: list[dict]) -> str:
    """Full (non-streaming) regeneration. Returns complete text."""
    client = _build_client(api_key)
    config = _build_config()

    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=config,
            )
            if resp and resp.text:
                return resp.text
        except Exception:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(RETRY_DELAY * (attempt + 1))

    raise RuntimeError("Failed to generate response")


# ── Title generator ──────────────────────────────────────────────────────────

def generate_title(api_key: str, question: str) -> str:
    """Generate a short chat title from the first user question."""
    client = _build_client(api_key)
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
            return resp.text.strip().strip('"').strip("'")[:50]
    except Exception:
        pass
    return question[:40] + ("..." if len(question) > 40 else "")
