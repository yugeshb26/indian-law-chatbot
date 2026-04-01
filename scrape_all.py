"""
Scrape Indian law data from multiple government sources:
1. indiacode.nic.in — All Central Acts with sections
2. indiankanoon.org — Legal encyclopaedia, case law summaries
3. legislative.gov.in — Acts list

Converts everything into Q&A pairs and merges into the dataset.
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

# ── HTTP helper ──────────────────────────────────────────────────────────────

def fetch(url: str, retries: int = 3) -> str:
    """Fetch URL with browser user-agent and retry logic."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8", errors="ignore")
        except Exception as e:
            print(f"    [retry {attempt+1}] {str(e)[:80]}")
            time.sleep(2 * (attempt + 1))
    return ""


def decode_html(text: str) -> str:
    """Decode HTML entities and clean whitespace."""
    text = html.unescape(text)
    text = re.sub(r'&#x20;', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ── 1. India Code — Acts list by year ────────────────────────────────────────

def get_all_years() -> list[str]:
    """Get all available act years from India Code."""
    page = fetch("https://www.indiacode.nic.in/handle/123456789/1362/browse?type=actyear&order=DESC")
    years = sorted(set(re.findall(r'value=(\d{4})', page)), reverse=True)
    print(f"  Found {len(years)} years ({years[0]} to {years[-1]})")
    return years


def get_acts_for_year(year: str) -> list[dict]:
    """Get all acts for a given year from India Code."""
    url = f"https://www.indiacode.nic.in/handle/123456789/1362/browse?type=actyear&order=DESC&rpp=100&value={year}"
    page = fetch(url)
    if not page:
        return []

    acts = []
    # Parse table rows: date, act number, short title, view link
    rows = re.findall(
        r'<tr><td[^>]*>([^<]*)</td><td[^>]*>\s*<em>(\d+)</em>\s*</td><td[^>]*>([^<]*)</td><td[^>]*><a href="([^"]*)"',
        page
    )
    for date, act_num, title, link in rows:
        acts.append({
            "date": decode_html(date.strip()),
            "act_number": act_num,
            "title": decode_html(title),
            "url": f"https://www.indiacode.nic.in{link}" if link.startswith("/") else link,
            "year": year,
        })
    return acts


def get_act_sections(act_url: str) -> list[dict]:
    """Get sections of a specific act from India Code."""
    page = fetch(act_url)
    if not page:
        return []

    sections = []
    # Try to find section links/text
    sec_matches = re.findall(
        r'Section\s+(\d+[A-Z]?)\s*[.:\-–]\s*([^<\n]{10,200})',
        page, re.IGNORECASE
    )
    for sec_num, sec_title in sec_matches:
        sections.append({
            "section": sec_num,
            "title": decode_html(sec_title).strip().rstrip(".")
        })
    return sections


def scrape_indiacode() -> list[dict]:
    """Scrape India Code for all Central Acts and generate Q&A pairs."""
    print("\n" + "=" * 60)
    print("1. SCRAPING INDIA CODE (indiacode.nic.in)")
    print("=" * 60)

    qa = []
    years = get_all_years()

    # Focus on important years (recent + landmark)
    key_years = [str(y) for y in range(2000, 2026)] + ["1950", "1860", "1872", "1973", "1908", "1882", "1881"]
    target_years = [y for y in years if y in key_years]
    print(f"  Targeting {len(target_years)} key years")

    all_acts = []
    for i, year in enumerate(target_years):
        print(f"  [{i+1}/{len(target_years)}] Year {year}...", end=" ", flush=True)
        acts = get_acts_for_year(year)
        print(f"{len(acts)} acts")
        all_acts.extend(acts)
        time.sleep(1)

    print(f"\n  Total acts found: {len(all_acts)}")

    # Generate Q&A for each act
    for act in all_acts:
        title = act["title"]
        year = act["year"]
        act_num = act["act_number"]

        qa.append({
            "prompt": f"What is {title}?",
            "response": f"{title} is Act No. {act_num} of {year}, a Central Act enacted by the Parliament of India. It was enacted on {act['date']}."
        })
        qa.append({
            "prompt": f"When was {title} enacted?",
            "response": f"{title} (Act No. {act_num} of {year}) was enacted on {act['date']} by the Parliament of India."
        })

    # Fetch sections for top important acts
    print("\n  Fetching sections for major acts...")
    important_acts = [a for a in all_acts if any(kw in a["title"].lower() for kw in [
        "constitution", "nyaya", "suraksha", "sakshya", "information technology",
        "right to information", "consumer protection", "companies", "labour",
        "hindu marriage", "muslim", "criminal", "civil", "evidence",
        "arbitration", "negotiable", "contract", "property", "motor vehicle",
        "banking", "income tax", "goods and services", "insolvency",
        "environment", "forest", "wildlife", "food safety", "narcotic",
        "prevention of corruption", "domestic violence", "juvenile",
        "scheduled castes", "scheduled tribes", "reservation",
    ])][:40]  # Cap at 40 to avoid rate limiting

    for i, act in enumerate(important_acts):
        print(f"    [{i+1}/{len(important_acts)}] {act['title'][:50]}...", end=" ", flush=True)
        sections = get_act_sections(act["url"])
        print(f"{len(sections)} sections")

        for sec in sections:
            qa.append({
                "prompt": f"What is Section {sec['section']} of {act['title']}?",
                "response": f"Section {sec['section']} of {act['title']} (Act No. {act['act_number']} of {act['year']}) deals with: {sec['title']}."
            })
        time.sleep(1)

    print(f"  India Code Q&A pairs: {len(qa)}")
    return qa


# ── 2. Indian Kanoon — Legal encyclopaedia ───────────────────────────────────

def scrape_indiankanoon() -> list[dict]:
    """Scrape Indian Kanoon for popular legal topics and case summaries."""
    print("\n" + "=" * 60)
    print("2. SCRAPING INDIAN KANOON (indiankanoon.org)")
    print("=" * 60)

    qa = []

    # Search for key legal topics
    topics = [
        "fundamental rights india",
        "right to equality article 14",
        "freedom of speech article 19",
        "right to life article 21",
        "bail conditions india",
        "FIR procedure india",
        "habeas corpus india",
        "writ petition types",
        "anticipatory bail",
        "section 498A dowry",
        "section 302 murder",
        "section 420 cheating",
        "divorce grounds india",
        "maintenance wife india",
        "custody children india",
        "RTI application process",
        "consumer complaint india",
        "cyber crime laws india",
        "defamation law india",
        "sedition law india",
        "land acquisition india",
        "tenant rights india",
        "labour laws minimum wages",
        "sexual harassment workplace",
        "juvenile justice india",
        "environmental law india",
        "contempt of court india",
        "PIL public interest litigation",
        "arbitration clause india",
        "cheque bounce section 138",
    ]

    for i, topic in enumerate(topics):
        print(f"  [{i+1}/{len(topics)}] {topic}...", end=" ", flush=True)
        url = f"https://indiankanoon.org/search/?formInput={urllib.request.quote(topic)}"
        page = fetch(url)
        if not page:
            print("failed")
            continue

        # Extract case/document titles and snippets
        results = re.findall(
            r'<a[^>]*class="result_title"[^>]*>([^<]+)</a>.*?<span[^>]*>([^<]{50,500})</span>',
            page, re.DOTALL
        )
        if not results:
            # Try alternate pattern
            results = re.findall(
                r'class="result_title"[^>]*>([^<]+)<.*?<div[^>]*class="result_text"[^>]*>([^<]{30,400})',
                page, re.DOTALL
            )

        count = 0
        for title, snippet in results[:3]:
            title = decode_html(title)
            snippet = decode_html(snippet)
            if len(snippet) > 30:
                qa.append({
                    "prompt": f"What is the legal position on '{topic}' in India?",
                    "response": f"Regarding {topic}: {snippet[:400]}. Reference: {title}."
                })
                count += 1
        print(f"{count} results")
        time.sleep(2)  # Be polite

    print(f"  Indian Kanoon Q&A pairs: {len(qa)}")
    return qa


# ── 3. legislative.gov.in — Acts list ────────────────────────────────────────

def scrape_legislative() -> list[dict]:
    """Scrape legislative.gov.in for additional acts data."""
    print("\n" + "=" * 60)
    print("3. SCRAPING LEGISLATIVE.GOV.IN")
    print("=" * 60)

    qa = []
    url = "https://legislative.gov.in/document-category/list-of-central-acts/"
    page = fetch(url)
    if not page:
        print("  Could not access legislative.gov.in")
        return qa

    # Extract act entries
    acts = re.findall(
        r'<a[^>]*href="([^"]*)"[^>]*>([^<]*(?:Act|Sanhita|Adhiniyam|Code|Ordinance)[^<]*)</a>',
        page, re.IGNORECASE
    )

    print(f"  Found {len(acts)} act references")

    for link, title in acts:
        title = decode_html(title)
        if len(title) > 10:
            qa.append({
                "prompt": f"Tell me about {title}.",
                "response": f"{title} is a Central legislation enacted by the Parliament of India. It is part of the corpus of Indian law available at legislative.gov.in."
            })

    # Add well-known acts with detailed Q&A
    landmark_acts = [
        ("Indian Contract Act, 1872", "The Indian Contract Act 1872 governs contracts in India. It defines what constitutes a valid contract, the essentials of a contract (offer, acceptance, consideration, free consent, capacity), and provides remedies for breach of contract. Sections 1-75 deal with general principles, while sections 124-238 cover special contracts like indemnity, guarantee, bailment, and agency."),
        ("Transfer of Property Act, 1882", "The Transfer of Property Act 1882 governs transfer of immovable property in India. It covers sale, mortgage, lease, exchange, and gift of immovable property. Section 54 defines sale, Section 58 defines mortgage types, and Section 105 defines lease."),
        ("Negotiable Instruments Act, 1881", "The Negotiable Instruments Act 1881 deals with promissory notes, bills of exchange, and cheques. Section 138 makes dishonour of cheque for insufficiency of funds a criminal offence punishable with imprisonment up to 2 years or fine up to twice the cheque amount, or both."),
        ("Information Technology Act, 2000", "The Information Technology Act 2000 provides legal recognition for electronic commerce and digital signatures. It defines cyber crimes including hacking (Section 66), identity theft (Section 66C), cyber terrorism (Section 66F), and publishing obscene content (Section 67). The IT Act also establishes the Cyber Appellate Tribunal."),
        ("Right to Information Act, 2005", "The Right to Information Act 2005 empowers citizens to access information from public authorities. Any citizen can file an RTI application with a fee of Rs. 10. The Public Information Officer must respond within 30 days. It covers all government bodies and substantially funded NGOs. Penalties for non-compliance can be up to Rs. 25,000."),
        ("Consumer Protection Act, 2019", "The Consumer Protection Act 2019 replaced the 1986 Act. It introduces provisions for e-commerce, product liability, and unfair trade practices. It establishes Central Consumer Protection Authority (CCPA), Consumer Disputes Redressal Commissions at District, State, and National levels. Complaints can be filed online."),
        ("Companies Act, 2013", "The Companies Act 2013 governs incorporation, regulation, and winding up of companies in India. It replaced the Companies Act 1956. Key features include one person company (Section 3), corporate social responsibility (Section 135), class action suits (Section 245), and National Company Law Tribunal."),
        ("Arbitration and Conciliation Act, 1996", "The Arbitration and Conciliation Act 1996 governs domestic and international commercial arbitration. It provides for appointment of arbitrators, conduct of proceedings, and enforcement of awards. Section 11 deals with appointment, Section 34 allows setting aside awards, and Section 36 deals with enforcement."),
        ("Motor Vehicles Act, 2019", "The Motor Vehicles Act 2019 (amended) significantly increased penalties for traffic violations. Drunk driving penalty increased to Rs. 10,000, driving without license to Rs. 5,000, and death due to rash driving can attract imprisonment up to 7 years. It also introduced electronic monitoring and cashless treatment for accident victims."),
        ("Protection of Women from Domestic Violence Act, 2005", "The Protection of Women from Domestic Violence Act 2005 protects women from domestic violence including physical, sexual, verbal, emotional, and economic abuse. It provides for protection orders, residence orders, monetary relief, and custody orders. The aggrieved woman can file a complaint before a Magistrate."),
        ("Prevention of Corruption Act, 1988", "The Prevention of Corruption Act 1988 punishes public servants for corruption. Section 7 deals with taking gratification (bribery), Section 13 covers criminal misconduct. The 2018 amendment also made giving bribe a specific offence under Section 8."),
        ("Goods and Services Tax Acts, 2017", "The GST Acts 2017 (CGST, SGST, IGST, UTGST) replaced multiple indirect taxes with a unified tax system. GST has four rate slabs: 5%, 12%, 18%, and 28%. Input tax credit allows businesses to claim credit for taxes paid on inputs. GST Council determines rates and policies."),
        ("Insolvency and Bankruptcy Code, 2016", "The Insolvency and Bankruptcy Code (IBC) 2016 provides a time-bound process for resolving insolvency. Corporate insolvency must be resolved within 330 days. It established the National Company Law Tribunal (NCLT) as the adjudicating authority and the Insolvency and Bankruptcy Board of India (IBBI) as the regulator."),
        ("POCSO Act, 2012", "The Protection of Children from Sexual Offences (POCSO) Act 2012 protects children below 18 from sexual assault, harassment, and pornography. It provides for Special Courts and child-friendly procedures. Punishments range from 3 years to life imprisonment depending on the offence. It mandates reporting of offences."),
        ("Hindu Marriage Act, 1955", "The Hindu Marriage Act 1955 governs marriage and divorce among Hindus, Buddhists, Jains, and Sikhs. Section 5 specifies conditions for valid marriage, Section 13 lists grounds for divorce including cruelty, desertion, conversion, mental disorder, and mutual consent (Section 13B). Minimum marriage age is 18 for women and 21 for men."),
        ("Muslim Personal Law (Shariat) Application Act, 1937", "The Muslim Personal Law (Shariat) Application Act 1937 applies Shariat law to Muslims in matters of marriage, divorce, maintenance, inheritance, and succession. Muslim marriage is a civil contract (Nikah). Divorce can be through Talaq, Khula, or judicial decree. The Triple Talaq was criminalized by the Muslim Women (Protection of Rights on Marriage) Act 2019."),
        ("Scheduled Castes and Scheduled Tribes (Prevention of Atrocities) Act, 1989", "The SC/ST Prevention of Atrocities Act 1989 prevents atrocities against members of Scheduled Castes and Scheduled Tribes. It defines specific offences like forced consumption of noxious substances, dumping excreta, land dispossession, forced labour, and sexual exploitation. It provides for Special Courts, rehabilitation of victims, and enhanced penalties."),
        ("Environmental Protection Act, 1986", "The Environment Protection Act 1986 empowers the Central Government to take measures for environmental protection. It provides for setting environmental standards, restricting industrial operations in certain areas, and penalizing polluters. Section 15 provides for imprisonment up to 5 years and fine up to Rs. 1 lakh for violations."),
        ("Food Safety and Standards Act, 2006", "The Food Safety and Standards Act 2006 consolidates laws relating to food. It established FSSAI (Food Safety and Standards Authority of India) as the apex body. It covers food standards, licensing and registration of food businesses, food recalls, and penalties for food adulteration and misleading advertisements."),
        ("NDPS Act, 1985", "The Narcotic Drugs and Psychotropic Substances (NDPS) Act 1985 prohibits production, manufacture, possession, sale, transport, and consumption of narcotic drugs and psychotropic substances. Punishments are based on quantity — small quantity (up to 1 year), commercial quantity (10-20 years rigorous imprisonment). Section 37 restricts bail for commercial quantities."),
    ]

    for act_name, description in landmark_acts:
        qa.append({
            "prompt": f"What is the {act_name}? Explain its key provisions.",
            "response": description
        })

    print(f"  Legislative Q&A pairs: {len(qa)}")
    return qa


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Indian Law Multi-Source Data Scraper")
    print("=" * 60)

    # Load existing dataset
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
        print(f"Existing dataset: {len(existing)} entries")
    else:
        existing = []

    # Scrape all sources
    qa_indiacode = scrape_indiacode()
    qa_kanoon = scrape_indiankanoon()
    qa_legislative = scrape_legislative()

    total_new = qa_indiacode + qa_kanoon + qa_legislative

    # Deduplicate by prompt
    existing_prompts = {item["prompt"].lower().strip() for item in existing}
    unique_new = [q for q in total_new if q["prompt"].lower().strip() not in existing_prompts]

    combined = existing + unique_new

    print(f"\n{'=' * 60}")
    print(f"RESULTS:")
    print(f"  Existing entries:     {len(existing)}")
    print(f"  India Code new:       {len(qa_indiacode)}")
    print(f"  Indian Kanoon new:    {len(qa_kanoon)}")
    print(f"  Legislative.gov new:  {len(qa_legislative)}")
    print(f"  Duplicates removed:   {len(total_new) - len(unique_new)}")
    print(f"  TOTAL DATASET:        {len(combined)}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"\n  Saved to {OUTPUT_PATH}")
    print(f"  File size: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
