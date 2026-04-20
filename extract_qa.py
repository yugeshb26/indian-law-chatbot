"""
Extract Q&A pairs from raw legal text files using Gemini API.
Chunks text → sends to Gemini → collects structured Q&A → merges into dataset.
"""

import json
import os
import time
import glob
import re
from google import genai
from google.genai.types import GenerateContentConfig

API_KEY = "AIzaSyCnsyhPuKbHLgifn3yVDBRJn9nQ3pTR5u0"
MODEL = "gemini-2.5-flash"
CHUNK_SIZE = 8000  # characters per chunk (fits in context window)
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "Alpie-core_core_indian_law.json")

# Map filenames to readable source names
SOURCE_MAP = {
    "raw_250883_english_01042024.txt": "Bharatiya Nyaya Sanhita 2023 (BNS)",
    "raw_250884_2_english_01042024.txt": "Bharatiya Nagarik Suraksha Sanhita 2023 (BNSS)",
    "raw_250882_english_01042024_0.txt": "Bharatiya Sakshya Adhiniyam 2023 (BSA)",
    "raw_20240716890312078.txt": "Constitution of India 2024",
}

QA_PROMPT = """You are a legal expert. Based on the following excerpt from {source}, generate exactly 10 high-quality question-answer pairs.

Rules:
- Questions should be practical and what a law student or citizen would actually ask
- Answers must be accurate, cite specific sections/articles, and be 2-4 sentences
- Cover different aspects: definitions, rights, procedures, penalties, scope
- Format as JSON array: [{{"prompt": "question", "response": "answer"}}]
- Return ONLY the JSON array, no other text

Text excerpt:
{text}
"""


def chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    """Split text into chunks, breaking at paragraph boundaries."""
    chunks = []
    while text:
        if len(text) <= size:
            chunks.append(text)
            break
        # Find last newline within chunk size
        cut = text[:size].rfind("\n\n")
        if cut < size // 2:
            cut = text[:size].rfind("\n")
        if cut < size // 2:
            cut = size
        chunks.append(text[:cut])
        text = text[cut:].lstrip()
    return chunks


def clean_text(text: str) -> str:
    """Remove Hindi/Devanagari text and gazette headers."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        # Skip lines that are mostly non-ASCII (Hindi text)
        ascii_chars = sum(1 for c in line if ord(c) < 128)
        if len(line) > 0 and ascii_chars / len(line) < 0.5:
            continue
        # Skip gazette boilerplate
        if any(skip in line for skip in ["xxxGID", "REGISTERED NO", "jftLV", "GAZETTE OF INDIA"]):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def extract_qa_from_chunk(client, chunk: str, source: str) -> list[dict]:
    """Send a chunk to Gemini and extract Q&A pairs."""
    prompt = QA_PROMPT.format(source=source, text=chunk[:CHUNK_SIZE])

    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                config=GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=4096,
                ),
            )
            if not resp or not resp.text:
                continue

            # Parse JSON from response
            text = resp.text.strip()
            # Remove markdown code fences if present
            text = re.sub(r'^```json\s*', '', text)
            text = re.sub(r'\s*```$', '', text)

            pairs = json.loads(text)
            if isinstance(pairs, list):
                return [p for p in pairs if "prompt" in p and "response" in p]

        except json.JSONDecodeError:
            print(f"  [!] JSON parse error on attempt {attempt + 1}")
        except Exception as e:
            print(f"  [!] API error on attempt {attempt + 1}: {str(e)[:100]}")
            time.sleep(3 * (attempt + 1))

    return []


def process_file(client, filepath: str, source: str) -> list[dict]:
    """Process one raw text file into Q&A pairs."""
    print(f"\n{'='*60}")
    print(f"Processing: {source}")
    print(f"File: {os.path.basename(filepath)}")

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    cleaned = clean_text(raw)
    chunks = chunk_text(cleaned)
    print(f"  Cleaned: {len(cleaned)} chars → {len(chunks)} chunks")

    all_qa = []
    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) < 200:
            continue  # skip tiny chunks
        print(f"  Chunk {i+1}/{len(chunks)}...", end=" ", flush=True)
        pairs = extract_qa_from_chunk(client, chunk, source)
        print(f"→ {len(pairs)} Q&A pairs")
        all_qa.extend(pairs)
        time.sleep(2)  # rate limit

    print(f"  Total from {source}: {len(all_qa)} Q&A pairs")
    return all_qa


def main():
    print("Indian Law PDF → Q&A Dataset Extractor")
    print("=" * 60)

    client = genai.Client(api_key=API_KEY)

    # Load existing dataset
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
        print(f"Existing dataset: {len(existing)} entries")
    else:
        existing = []

    # Process each raw text file
    raw_dir = os.path.dirname(__file__)
    all_new_qa = []

    for filename, source in SOURCE_MAP.items():
        filepath = os.path.join(raw_dir, filename)
        if not os.path.exists(filepath):
            print(f"  [SKIP] {filename} not found")
            continue
        qa_pairs = process_file(client, filepath, source)
        all_new_qa.extend(qa_pairs)

    # Merge with existing
    combined = existing + all_new_qa
    print(f"\n{'='*60}")
    print(f"Existing: {len(existing)}")
    print(f"New:      {len(all_new_qa)}")
    print(f"Total:    {len(combined)}")

    # Save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
