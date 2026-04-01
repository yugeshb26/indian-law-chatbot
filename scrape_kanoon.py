"""
Scrape Indian Kanoon for judgments from all major courts.
Extracts case names, judges, dates, and first paragraphs to build Q&A pairs.
"""

import json
import os
import re
import time
import html
import urllib.request
import urllib.error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(BASE_DIR, "Alpie-core_core_indian_law.json")
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# ── Courts to scrape (key years only to stay within limits) ──────────────────
COURTS = {
    "supremecourt": "Supreme Court of India",
    "delhi": "Delhi High Court",
    "bombay": "Bombay High Court",
    "madras": "Madras High Court",
    "calcutta": "Calcutta High Court",
    "allahabad": "Allahabad High Court",
    "karnataka": "Karnataka High Court",
    "kerala": "Kerala High Court",
    "punjab": "Punjab and Haryana High Court",
    "gujarat": "Gujarat High Court",
    "rajasthan": "Rajasthan High Court",
    "telangana": "Telangana High Court",
    "patna": "Patna High Court",
    "jharkhand": "Jharkhand High Court",
    "madhya_pradesh": "Madhya Pradesh High Court",
    "chattisgarh": "Chhattisgarh High Court",
    "uttarakhand": "Uttarakhand High Court",
    "gauhati": "Gauhati High Court",
    "himachal_pradesh": "Himachal Pradesh High Court",
    "orissa": "Orissa High Court",
}

# Years to scrape per court — recent + landmark
YEARS_SUPREME = list(range(2020, 2026)) + [2017, 2014, 2010, 2005, 2000, 1993, 1978, 1973, 1950]
YEARS_HC = [2024, 2023, 2022, 2021, 2020]

# Months
MONTHS = [
    ("1-1", "31-1", "January"), ("1-2", "28-2", "February"), ("1-3", "31-3", "March"),
    ("1-4", "30-4", "April"), ("1-5", "31-5", "May"), ("1-6", "30-6", "June"),
    ("1-7", "31-7", "July"), ("1-8", "31-8", "August"), ("1-9", "30-9", "September"),
    ("1-10", "31-10", "October"), ("1-11", "30-11", "November"), ("1-12", "31-12", "December"),
]


def fetch(url: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8", errors="ignore")
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
    return ""


def decode(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ── Extract cases from search results page ───────────────────────────────────

def extract_cases_from_search(page_html: str) -> list[dict]:
    """Extract case title, date, doc_id, and judge from search results."""
    cases = []

    # Pattern: <a href="/docfragment/ID/?...">Title vs Party on Date</a>
    matches = re.findall(
        r'<a href="/docfragment/(\d+)/\?[^"]*">([^<]+)</a>',
        page_html
    )

    for doc_id, title in matches:
        title = decode(title)
        # Parse "Party A vs Party B on Date" pattern
        date_match = re.search(r' on (\d{1,2} \w+ \d{4})$', title)
        date = date_match.group(1) if date_match else ""
        case_name = title.replace(f" on {date}", "").strip() if date else title

        cases.append({
            "doc_id": doc_id,
            "title": case_name,
            "date": date,
            "full_title": title,
        })

    return cases


# ── Get case headnote/summary from doc page ──────────────────────────────────

def get_case_summary(doc_id: str) -> dict:
    """Fetch a judgment page and extract author, bench, and first paragraph."""
    url = f"https://indiankanoon.org/doc/{doc_id}/"
    page = fetch(url)
    if not page:
        return {}

    result = {}

    # Author
    author_match = re.search(r'Author:\s*([^\n<]+)', page)
    if author_match:
        result["author"] = decode(author_match.group(1))

    # Bench
    bench_match = re.search(r'Bench:\s*([^\n<]+)', page)
    if bench_match:
        result["bench"] = decode(bench_match.group(1))

    # Citation
    cite_match = re.search(r'(\d{4}\s+INSC\s+\d+)', page)
    if cite_match:
        result["citation"] = cite_match.group(1)

    # Extract first meaningful paragraph (the headnote/summary)
    # Remove all HTML tags and get text after the judgment header
    text = re.sub(r'<[^>]+>', '\n', page)
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    # Find the judgment text start (after "REPORTABLE" or "JUDGMENT" etc.)
    start_idx = 0
    for i, line in enumerate(lines):
        if any(kw in line.upper() for kw in ['JUDGMENT', 'ORDER', 'REPORTABLE', 'J U D G M E N T']):
            start_idx = i + 1
            break

    # Collect first 3-5 meaningful paragraphs (skip short lines)
    summary_lines = []
    char_count = 0
    for line in lines[start_idx:]:
        if len(line) > 30 and not line.startswith(('Cites', 'Cited', 'Author:', 'Bench:', 'Tools')):
            summary_lines.append(line)
            char_count += len(line)
            if char_count > 500:
                break

    result["summary"] = " ".join(summary_lines)[:600] if summary_lines else ""
    return result


# ── Scrape one court ─────────────────────────────────────────────────────────

def scrape_court(court_key: str, court_name: str, years: list[int], max_cases_per_month: int = 10) -> list[dict]:
    """Scrape judgments for a court across specified years."""
    qa = []
    total_cases = 0

    for year in years:
        print(f"    {year}...", end=" ", flush=True)
        year_cases = 0

        for from_d, to_d, month_name in MONTHS:
            url = (
                f"https://indiankanoon.org/search/?formInput=doctypes:{court_key}"
                f"+fromdate:{from_d}-{year}+todate:+{to_d}-{year}"
            )
            page = fetch(url)
            if not page:
                continue

            cases = extract_cases_from_search(page)[:max_cases_per_month]

            for case in cases:
                # Basic Q&A from search listing
                qa.append({
                    "prompt": f"What is the case {case['title']}?",
                    "response": f"{case['full_title']} is a judgment by the {court_name}. "
                                f"It was decided on {case['date']}." if case['date'] else
                                f"{case['title']} is a judgment by the {court_name}."
                })
                year_cases += 1

            time.sleep(1)  # Rate limit

        # For Supreme Court, also get summaries for top cases of the year
        if court_key == "supremecourt" and year >= 2020:
            url = f"https://indiankanoon.org/search/?formInput=doctypes:supremecourt+year:{year}"
            page = fetch(url)
            if page:
                top_cases = extract_cases_from_search(page)[:5]
                for case in top_cases:
                    summary = get_case_summary(case["doc_id"])
                    if summary.get("summary"):
                        judge = summary.get("author", "the bench")
                        citation = summary.get("citation", "")
                        qa.append({
                            "prompt": f"Summarize the Supreme Court judgment in {case['title']}.",
                            "response": f"{case['full_title']} ({citation}). "
                                        f"Authored by {judge}. "
                                        f"{summary['summary']}"
                        })
                    time.sleep(2)

        total_cases += year_cases
        print(f"{year_cases} cases")

    return qa


# ── Also scrape tribunals and commissions ────────────────────────────────────

def scrape_tribunals() -> list[dict]:
    """Scrape key tribunals for recent decisions."""
    qa = []
    tribunals = {
        "National Green Tribunal": "ngt",
        "National Consumer Disputes Redressal Commission": "ncdrc",
        "Income Tax Appellate Tribunal": "itat",
        "Competition Commission of India": "cci",
        "Securities Appellate Tribunal": "sat",
    }

    for name, key in tribunals.items():
        print(f"    {name}...", end=" ", flush=True)
        url = f"https://indiankanoon.org/search/?formInput=doctypes:{key}+year:2024"
        page = fetch(url)
        if not page:
            print("failed")
            continue

        cases = extract_cases_from_search(page)[:15]
        for case in cases:
            qa.append({
                "prompt": f"What is the {name} case {case['title']}?",
                "response": f"{case['full_title']} is a decision by the {name}."
                            f" Decided on {case['date']}." if case['date'] else
                            f"{case['title']} is a decision by the {name}."
            })
        print(f"{len(cases)} cases")
        time.sleep(2)

    return qa


# ── Scrape landmark judgments with detailed summaries ────────────────────────

def scrape_landmark_cases() -> list[dict]:
    """Search for famous landmark judgments and get their summaries."""
    qa = []
    landmarks = [
        "Kesavananda Bharati",
        "Maneka Gandhi vs Union of India",
        "Vishaka vs State of Rajasthan",
        "Navtej Singh Johar",
        "K.S. Puttaswamy privacy",
        "Shreya Singhal vs Union of India",
        "Indian Young Lawyers Association Sabarimala",
        "Mohd. Ahmed Khan vs Shah Bano Begum",
        "Indra Sawhney vs Union of India",
        "S.R. Bommai vs Union of India",
        "Minerva Mills vs Union of India",
        "Golaknath vs State of Punjab",
        "A.K. Gopalan vs State of Madras",
        "State of West Bengal vs Committee for Protection",
        "TMA Pai Foundation",
        "Olga Tellis vs Bombay Municipal Corporation",
        "Lily Thomas vs Union of India",
        "Ashoka Kumar Thakur OBC reservation",
        "MC Mehta environmental",
        "DK Basu vs State of West Bengal arrest guidelines",
    ]

    print("\n  Landmark judgments:")
    for query in landmarks:
        print(f"    {query[:40]}...", end=" ", flush=True)
        url = f"https://indiankanoon.org/search/?formInput={urllib.request.quote(query)}+doctypes:supremecourt"
        page = fetch(url)
        if not page:
            print("failed")
            continue

        cases = extract_cases_from_search(page)[:1]
        if cases:
            case = cases[0]
            summary = get_case_summary(case["doc_id"])
            if summary.get("summary"):
                judge = summary.get("author", "the bench")
                citation = summary.get("citation", "")
                qa.append({
                    "prompt": f"What is the landmark judgment in {case['title']}?",
                    "response": f"{case['full_title']} ({citation}) is a landmark Supreme Court judgment. "
                                f"Authored by {judge}. {summary['summary']}"
                })
                print("OK")
            else:
                qa.append({
                    "prompt": f"Tell me about the case {case['title']}.",
                    "response": f"{case['full_title']} is a landmark Supreme Court judgment."
                })
                print("basic")
        else:
            print("not found")
        time.sleep(2)

    return qa


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Indian Kanoon Judgment Scraper")
    print("=" * 60)

    # Load existing
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
        print(f"Existing dataset: {len(existing)} entries")
    else:
        existing = []

    all_qa = []

    # 1. Supreme Court (detailed, more years)
    print(f"\n  Supreme Court of India ({len(YEARS_SUPREME)} years):")
    qa = scrape_court("supremecourt", "Supreme Court of India", YEARS_SUPREME, max_cases_per_month=8)
    all_qa.extend(qa)
    print(f"  → {len(qa)} Q&A pairs")

    # 2. High Courts (recent years)
    for court_key, court_name in list(COURTS.items())[1:]:  # Skip SC (already done)
        print(f"\n  {court_name} ({len(YEARS_HC)} years):")
        qa = scrape_court(court_key, court_name, YEARS_HC, max_cases_per_month=5)
        all_qa.extend(qa)
        print(f"  → {len(qa)} Q&A pairs total")

    # 3. Tribunals
    print("\n  Tribunals:")
    qa_tribunals = scrape_tribunals()
    all_qa.extend(qa_tribunals)

    # 4. Landmark judgments with summaries
    qa_landmarks = scrape_landmark_cases()
    all_qa.extend(qa_landmarks)

    # Deduplicate
    existing_prompts = {item["prompt"].lower().strip() for item in existing}
    unique_new = [q for q in all_qa if q["prompt"].lower().strip() not in existing_prompts]

    combined = existing + unique_new

    print(f"\n{'=' * 60}")
    print(f"RESULTS:")
    print(f"  Existing:        {len(existing)}")
    print(f"  New scraped:     {len(all_qa)}")
    print(f"  After dedup:     {len(unique_new)}")
    print(f"  TOTAL DATASET:   {len(combined)}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"\n  Saved to {OUTPUT_PATH}")
    print(f"  File size: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
