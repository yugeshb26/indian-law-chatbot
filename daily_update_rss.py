"""
Daily Indian legal data updater using Indian Kanoon RSS feeds.

Source: https://indiankanoon.org/feeds/
- No API key required
- No rate limits
- Official XML feeds, daily updated
- Covers Supreme Court, all 24 High Courts, 11 tribunals, district courts

Run daily/weekly via cron to keep dataset fresh.
"""

import json
import os
import re
import time
import html
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(BASE_DIR, "Alpie-core_core_indian_law.json")
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# ── RSS Feed Sources ─────────────────────────────────────────────────────────
FEED_BASE = "https://indiankanoon.org/feeds/latest"

FEEDS = {
    # Supreme Court
    "Supreme Court of India": f"{FEED_BASE}/supremecourt/",
    "Supreme Court Daily Orders": f"{FEED_BASE}/scorders/",

    # High Courts (24)
    "Allahabad High Court": f"{FEED_BASE}/allahabad/",
    "Andhra Pradesh High Court": f"{FEED_BASE}/andhra/",
    "Bombay High Court": f"{FEED_BASE}/bombay/",
    "Calcutta High Court": f"{FEED_BASE}/calcutta/",
    "Chhattisgarh High Court": f"{FEED_BASE}/chattisgarh/",
    "Delhi High Court": f"{FEED_BASE}/delhi/",
    "Gauhati High Court": f"{FEED_BASE}/gauhati/",
    "Gujarat High Court": f"{FEED_BASE}/gujarat/",
    "Himachal Pradesh High Court": f"{FEED_BASE}/himachal_pradesh/",
    "Jammu & Kashmir High Court": f"{FEED_BASE}/jammu/",
    "Jharkhand High Court": f"{FEED_BASE}/jharkhand/",
    "Karnataka High Court": f"{FEED_BASE}/karnataka/",
    "Kerala High Court": f"{FEED_BASE}/kerala/",
    "Madhya Pradesh High Court": f"{FEED_BASE}/madhyapradesh/",
    "Madras High Court": f"{FEED_BASE}/madras/",
    "Manipur High Court": f"{FEED_BASE}/manipur/",
    "Meghalaya High Court": f"{FEED_BASE}/meghalaya/",
    "Orissa High Court": f"{FEED_BASE}/orissa/",
    "Patna High Court": f"{FEED_BASE}/patna/",
    "Punjab and Haryana High Court": f"{FEED_BASE}/punjab/",
    "Rajasthan High Court": f"{FEED_BASE}/rajasthan/",
    "Sikkim High Court": f"{FEED_BASE}/sikkim/",
    "Telangana High Court": f"{FEED_BASE}/telangana/",
    "Tripura High Court": f"{FEED_BASE}/tripura/",
    "Uttarakhand High Court": f"{FEED_BASE}/uttarakhand/",

    # Tribunals
    "National Green Tribunal": f"{FEED_BASE}/ngt/",
    "National Consumer Disputes Redressal Commission": f"{FEED_BASE}/ncdrc/",
    "Income Tax Appellate Tribunal": f"{FEED_BASE}/itat/",
    "Competition Commission of India": f"{FEED_BASE}/cci/",
    "Central Information Commission": f"{FEED_BASE}/cic/",
    "Central Administrative Tribunal": f"{FEED_BASE}/cat/",
    "Customs Excise & Service Tax Tribunal": f"{FEED_BASE}/cestat/",
    "Securities Appellate Tribunal": f"{FEED_BASE}/sat/",
    "National Company Law Appellate Tribunal": f"{FEED_BASE}/nclat/",
    "Appellate Tribunal for Electricity": f"{FEED_BASE}/aptel/",
    "Telecom Disputes Settlement & Appellate Tribunal": f"{FEED_BASE}/tdsat/",

    # District Courts
    "Delhi District Court": f"{FEED_BASE}/delhidc/",
    "Bangalore District Court": f"{FEED_BASE}/bangaloredc/",

    # Other
    "Lok Sabha Debates": f"{FEED_BASE}/loksabha/",
}


# ── HTTP helper ──────────────────────────────────────────────────────────────

def fetch(url: str, retries: int = 3) -> str:
    """Fetch URL with browser UA and retry logic."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8", errors="ignore")
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                print(f"      [error] {str(e)[:80]}")
    return ""


def clean_text(text: str) -> str:
    """Decode HTML entities and clean whitespace."""
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)  # strip HTML tags
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ── RSS parsing ──────────────────────────────────────────────────────────────

def parse_rss(xml_text: str) -> list[dict]:
    """Parse RSS XML and extract items."""
    items = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        print(f"      [parse error] {e}")
        return items

    # RSS 2.0 structure: rss > channel > item
    # Atom structure: feed > entry
    channel = root.find("channel")
    if channel is not None:
        # RSS 2.0
        for item in channel.findall("item"):
            title = item.findtext("title", "").strip()
            link = item.findtext("link", "").strip()
            desc = item.findtext("description", "").strip()
            pub_date = item.findtext("pubDate", "").strip()
            if title:
                items.append({
                    "title": clean_text(title),
                    "link": link,
                    "description": clean_text(desc),
                    "date": pub_date,
                })
    else:
        # Atom
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("atom:entry", ns):
            title = entry.findtext("atom:title", "", ns).strip()
            link_el = entry.find("atom:link", ns)
            link = link_el.get("href", "") if link_el is not None else ""
            summary = entry.findtext("atom:summary", "", ns).strip()
            updated = entry.findtext("atom:updated", "", ns).strip()
            if title:
                items.append({
                    "title": clean_text(title),
                    "link": link,
                    "description": clean_text(summary),
                    "date": updated,
                })

    return items


# ── Q&A generation ───────────────────────────────────────────────────────────

def parse_case_title(full_title: str) -> tuple[str, str]:
    """Split 'X vs Y on date' into (case_name, date)."""
    date_match = re.search(r' on (\d{1,2} \w+,? \d{4})$', full_title)
    if date_match:
        date = date_match.group(1)
        case_name = full_title[: date_match.start()].strip()
        return case_name, date
    return full_title, ""


def items_to_qa(court_name: str, items: list[dict]) -> list[dict]:
    """Convert RSS items to Q&A pairs."""
    qa = []
    for item in items:
        case_name, date = parse_case_title(item["title"])

        # Q&A 1: Basic case info
        if date:
            response = f"{item['title']} is a recent judgment from the {court_name}, decided on {date}."
        else:
            response = f"{item['title']} is a recent judgment from the {court_name}."

        if item["description"]:
            response += f" Summary: {item['description'][:500]}"

        qa.append({
            "prompt": f"What is the case {case_name}?",
            "response": response,
        })

        # Q&A 2: Court-specific
        qa.append({
            "prompt": f"Tell me about the {court_name} judgment in {case_name}.",
            "response": response,
        })

        # Q&A 3: If description has substance, add a contextual one
        if item["description"] and len(item["description"]) > 100:
            qa.append({
                "prompt": f"Summarize the recent {court_name} judgment {case_name}.",
                "response": f"In {item['title']}, the {court_name} held: {item['description'][:600]}",
            })

    return qa


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Indian Kanoon RSS Daily Updater")
    print("=" * 60)
    print(f"Run time: {datetime.now().isoformat()}")
    print(f"Total feeds: {len(FEEDS)}")

    # Load existing dataset
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
        print(f"Existing dataset: {len(existing)} entries")
    else:
        existing = []

    all_qa = []
    feed_stats = {}

    for i, (court_name, feed_url) in enumerate(FEEDS.items(), 1):
        print(f"  [{i}/{len(FEEDS)}] {court_name}...", end=" ", flush=True)
        xml = fetch(feed_url)
        if not xml:
            print("failed")
            feed_stats[court_name] = 0
            continue

        items = parse_rss(xml)
        qa = items_to_qa(court_name, items)
        all_qa.extend(qa)
        feed_stats[court_name] = len(items)
        print(f"{len(items)} items → {len(qa)} Q&A")
        time.sleep(0.5)  # be polite to the server

    # Deduplicate
    existing_prompts = {item["prompt"].lower().strip() for item in existing}
    unique_new = [q for q in all_qa if q["prompt"].lower().strip() not in existing_prompts]

    combined = existing + unique_new

    print(f"\n{'=' * 60}")
    print("RESULTS:")
    print(f"  Existing entries:    {len(existing)}")
    print(f"  Total scraped:       {len(all_qa)}")
    print(f"  Unique new entries:  {len(unique_new)}")
    print(f"  Duplicates skipped:  {len(all_qa) - len(unique_new)}")
    print(f"  TOTAL DATASET:       {len(combined)}")

    print(f"\n  Top contributing feeds:")
    for court, count in sorted(feed_stats.items(), key=lambda x: -x[1])[:10]:
        print(f"    {count:>4}  {court}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"\n  Saved to {OUTPUT_PATH}")
    print(f"  File size: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
