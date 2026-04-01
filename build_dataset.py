"""
Build Q&A dataset from raw legal text files by parsing sections directly.
No API calls needed — fast, complete, and offline.
"""

import json
import os
import re

RAW_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(RAW_DIR, "Alpie-core_core_indian_law.json")

# ── Text cleaning ────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove Hindi, gazette headers, and extra whitespace."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip mostly non-ASCII (Hindi)
        ascii_chars = sum(1 for c in line if ord(c) < 128)
        if len(line) > 3 and ascii_chars / len(line) < 0.5:
            continue
        # Skip gazette boilerplate
        if any(s in line for s in [
            "xxxGID", "REGISTERED NO", "jftLV", "GAZETTE OF INDIA",
            "___________", "EXTRAORDINARY", "PUBLISHED BY AUTHORITY",
            "Part II", "Sec. 1]", "Sec.", "[Part II",
        ]):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


# ── Section parsers for each Act ─────────────────────────────────────────────

def parse_bns(text: str) -> list[dict]:
    """Parse Bharatiya Nyaya Sanhita 2023 sections."""
    qa = []
    # Match section patterns like "123. (1) ..." or "Section 123"
    sections = re.split(r'\n(\d{1,3})\.\s+\(1\)\s+', text)

    # Also extract chapter-level content
    chapters = re.findall(r'CHAPTER\s+[IVXLCDM]+\s*\n\s*(.+?)(?=\nCHAPTER|\Z)', text, re.DOTALL)

    # Parse definitions (Section 2)
    defs = re.findall(r'\((\d+)\)\s+"([^"]+)"[^;]*?(?:means|includes)\s+(.+?)(?=\(\d+\)|$)', text, re.DOTALL)
    for num, term, definition in defs:
        definition = " ".join(definition.split())[:500]
        qa.append({
            "prompt": f"What is the definition of '{term}' under the Bharatiya Nyaya Sanhita 2023?",
            "response": f"Under Section 2({num}) of the Bharatiya Nyaya Sanhita (BNS) 2023, '{term}' {definition.strip()}"
        })

    # Parse offence sections with punishments
    punishments = re.findall(
        r'(\d{1,3})\.\s*(.*?)(?:shall be punished|punishable)\s+with\s+(.*?)(?:\.|$)',
        text, re.DOTALL
    )
    for sec_num, offence_text, punishment in punishments:
        offence = " ".join(offence_text.split())[:300]
        punishment = " ".join(punishment.split())[:300]
        if len(offence) > 20:
            qa.append({
                "prompt": f"What is the offence and punishment under Section {sec_num} of the Bharatiya Nyaya Sanhita 2023?",
                "response": f"Section {sec_num} of BNS 2023 deals with: {offence.strip()}. The punishment is {punishment.strip()}."
            })

    # General BNS questions
    qa.extend([
        {"prompt": "What is the Bharatiya Nyaya Sanhita 2023?",
         "response": "The Bharatiya Nyaya Sanhita (BNS) 2023 is Act No. 45 of 2023, enacted on December 25, 2023. It replaces the Indian Penal Code (IPC) 1860 as the primary criminal law of India. It consolidates and amends provisions relating to offences and connected matters."},
        {"prompt": "When did the Bharatiya Nyaya Sanhita 2023 come into effect?",
         "response": "The Bharatiya Nyaya Sanhita 2023 received the President's assent on December 25, 2023. It came into force on July 1, 2024, replacing the Indian Penal Code 1860."},
        {"prompt": "What is the scope of application of BNS 2023?",
         "response": "Under Section 1 of BNS 2023, every person is liable for acts or omissions within India. It also applies to Indian citizens abroad, persons on Indian-registered ships/aircraft, and anyone committing offences targeting computer resources in India."},
        {"prompt": "What does Section 1(5) of BNS say about extraterritorial jurisdiction?",
         "response": "Section 1(5) of BNS 2023 extends jurisdiction to: (a) any citizen of India committing offences abroad, (b) any person on Indian-registered ships or aircraft, and (c) any person committing offences targeting computer resources located in India."},
        {"prompt": "How does BNS 2023 define 'child'?",
         "response": "Under Section 2(3) of the Bharatiya Nyaya Sanhita 2023, 'child' means any person below the age of eighteen years."},
        {"prompt": "How does BNS 2023 define 'document'?",
         "response": "Under Section 2(8) of BNS 2023, 'document' means any matter expressed upon any substance by means of letters, figures or marks, and includes electronic and digital records, intended to be used as evidence."},
    ])
    return qa


def parse_bnss(text: str) -> list[dict]:
    """Parse Bharatiya Nagarik Suraksha Sanhita 2023 (replaces CrPC)."""
    qa = []

    # Extract definitions
    defs = re.findall(r'\(([a-z])\)\s+"([^"]+)"\s+(?:means|includes)\s+(.+?)(?=\([a-z]\)|$)', text, re.DOTALL)
    for letter, term, definition in defs:
        definition = " ".join(definition.split())[:500]
        qa.append({
            "prompt": f"What is the definition of '{term}' under the Bharatiya Nagarik Suraksha Sanhita 2023?",
            "response": f"Under Section 2(1)({letter}) of BNSS 2023, '{term}' {definition.strip()}"
        })

    # Bail provisions
    bail_sections = re.findall(r'(\d{1,3})\.\s*(.*?bail.*?)(?=\n\d{1,3}\.|$)', text, re.DOTALL | re.IGNORECASE)
    for sec_num, content in bail_sections[:10]:
        content = " ".join(content.split())[:400]
        qa.append({
            "prompt": f"What does Section {sec_num} of BNSS 2023 say about bail?",
            "response": f"Section {sec_num} of BNSS 2023 provides: {content}"
        })

    # FIR and investigation
    fir_sections = re.findall(r'(\d{1,3})\.\s*(.*?(?:information|first information|F\.I\.R|complaint).*?)(?=\n\d{1,3}\.|$)',
                               text, re.DOTALL | re.IGNORECASE)
    for sec_num, content in fir_sections[:10]:
        content = " ".join(content.split())[:400]
        qa.append({
            "prompt": f"What does Section {sec_num} of BNSS 2023 say about FIR/complaints?",
            "response": f"Section {sec_num} of BNSS 2023 states: {content}"
        })

    # General BNSS questions
    qa.extend([
        {"prompt": "What is the Bharatiya Nagarik Suraksha Sanhita 2023?",
         "response": "The Bharatiya Nagarik Suraksha Sanhita (BNSS) 2023 is Act No. 46 of 2023, replacing the Code of Criminal Procedure (CrPC) 1973. It consolidates and amends the law relating to criminal procedure in India."},
        {"prompt": "What is 'audio-video electronic means' under BNSS 2023?",
         "response": "Under Section 2(1)(a) of BNSS 2023, 'audio-video electronic means' includes use of any communication device for video conferencing, recording of identification processes, search and seizure, evidence, and electronic communication."},
        {"prompt": "What is the definition of 'bail' under BNSS 2023?",
         "response": "Under Section 2(1)(b) of BNSS 2023, 'bail' means release of a person accused or suspected of an offence from custody upon conditions imposed by an officer or Court on execution of a bond or bail bond."},
        {"prompt": "What is a cognizable offence under BNSS 2023?",
         "response": "Under Section 2(1)(g) of BNSS 2023, a 'cognizable offence' is one for which a police officer may arrest without warrant, as listed in the First Schedule or any other law in force."},
        {"prompt": "What is a 'victim' under BNSS 2023?",
         "response": "Under Section 2(1)(y) of BNSS 2023, 'victim' means a person who has suffered any loss or injury caused by the act or omission of the accused, and includes the guardian or legal heir of such victim."},
        {"prompt": "What is a warrant-case under BNSS 2023?",
         "response": "Under Section 2(1)(z) of BNSS 2023, a 'warrant-case' means a case relating to an offence punishable with death, imprisonment for life, or imprisonment for a term exceeding two years."},
        {"prompt": "Does BNSS 2023 apply to all of India?",
         "response": "BNSS 2023 applies to all of India except certain provisions do not apply to Nagaland and tribal areas. However, State Governments may extend these provisions by notification."},
    ])
    return qa


def parse_bsa(text: str) -> list[dict]:
    """Parse Bharatiya Sakshya Adhiniyam 2023 (replaces Evidence Act)."""
    qa = []

    # Extract definitions
    defs = re.findall(r'\(([a-z])\)\s+"([^"]+)"[^;]*?(?:means|includes)\s+(.+?)(?=\([a-z]\)|$)', text, re.DOTALL)
    for letter, term, definition in defs:
        definition = " ".join(definition.split())[:500]
        qa.append({
            "prompt": f"What is the definition of '{term}' under the Bharatiya Sakshya Adhiniyam 2023?",
            "response": f"Under Section 2(1)({letter}) of BSA 2023, '{term}' {definition.strip()}"
        })

    # Evidence rules
    evidence_sections = re.findall(r'(\d{1,3})\.\s*(.*?(?:evidence|proved|witness|document|fact).*?)(?=\n\d{1,3}\.|$)',
                                    text, re.DOTALL | re.IGNORECASE)
    for sec_num, content in evidence_sections[:15]:
        content = " ".join(content.split())[:400]
        qa.append({
            "prompt": f"What does Section {sec_num} of BSA 2023 say about evidence?",
            "response": f"Section {sec_num} of BSA 2023 provides: {content}"
        })

    qa.extend([
        {"prompt": "What is the Bharatiya Sakshya Adhiniyam 2023?",
         "response": "The Bharatiya Sakshya Adhiniyam (BSA) 2023 is Act No. 47 of 2023, replacing the Indian Evidence Act 1872. It provides general rules and principles of evidence for fair trial in India."},
        {"prompt": "What is 'document' under BSA 2023?",
         "response": "Under Section 2(1)(d) of BSA 2023, 'document' means any matter expressed upon any substance by means of letters, figures or marks, including electronic and digital records. This covers emails, server logs, computer documents, smartphone messages, websites, and voice mail."},
        {"prompt": "What constitutes 'evidence' under BSA 2023?",
         "response": "Under Section 2(1)(e) of BSA 2023, 'evidence' includes: (i) oral evidence — all statements including electronic statements that the Court permits witnesses to make, and (ii) documentary evidence — all documents including electronic or digital records produced for Court inspection."},
        {"prompt": "When is a fact considered 'proved' under BSA 2023?",
         "response": "Under Section 2(1)(j) of BSA 2023, a fact is 'proved' when, after considering the matters before it, the Court either believes it to exist, or considers its existence so probable that a prudent man ought to act upon the supposition that it exists."},
        {"prompt": "What is 'conclusive proof' under BSA 2023?",
         "response": "Under Section 2(1)(b) of BSA 2023, when one fact is declared as conclusive proof of another, the Court shall regard the other as proved on proof of the first fact, and shall not allow evidence to disprove it."},
        {"prompt": "What is Section 3 of BSA 2023 about?",
         "response": "Section 3 of BSA 2023 states that evidence may be given in any suit or proceeding of the existence or non-existence of every fact in issue and of other facts declared to be relevant, and of no others."},
    ])
    return qa


def parse_constitution(text: str) -> list[dict]:
    """Parse Constitution of India."""
    qa = []

    # Extract Articles
    articles = re.findall(
        r'(?:^|\n)(\d{1,3}[A-Z]?)\.\s+(.+?)(?=\n\d{1,3}[A-Z]?\.\s|\nPART|\Z)',
        text, re.DOTALL
    )
    for art_num, content in articles:
        content_clean = " ".join(content.split())[:500]
        if len(content_clean) > 30:
            # Get first sentence as title
            title = content_clean.split(".")[0] if "." in content_clean else content_clean[:80]
            qa.append({
                "prompt": f"What does Article {art_num} of the Indian Constitution say?",
                "response": f"Article {art_num} of the Indian Constitution: {content_clean}"
            })

    # Fundamental Rights - detailed Q&A
    qa.extend([
        {"prompt": "What are the Fundamental Rights in the Indian Constitution?",
         "response": "Fundamental Rights are enshrined in Part III (Articles 12-35) of the Indian Constitution. They include: Right to Equality (Arts 14-18), Right to Freedom (Arts 19-22), Right against Exploitation (Arts 23-24), Right to Freedom of Religion (Arts 25-28), Cultural and Educational Rights (Arts 29-30), and Right to Constitutional Remedies (Art 32)."},
        {"prompt": "What is Article 14 of the Indian Constitution?",
         "response": "Article 14 guarantees equality before law. It states that the State shall not deny to any person equality before the law or the equal protection of the laws within the territory of India."},
        {"prompt": "What is Article 15 of the Indian Constitution?",
         "response": "Article 15 prohibits discrimination on grounds of religion, race, caste, sex or place of birth. The State shall not discriminate against any citizen on these grounds. It also allows special provisions for women, children, socially and educationally backward classes, SCs and STs."},
        {"prompt": "What is Article 19 of the Indian Constitution?",
         "response": "Article 19 guarantees six freedoms to all citizens: (a) freedom of speech and expression, (b) freedom to assemble peaceably, (c) freedom to form associations/unions, (d) freedom to move freely throughout India, (e) freedom to reside and settle anywhere in India, and (g) freedom to practise any profession or carry on any trade/business."},
        {"prompt": "What is Article 21 of the Indian Constitution?",
         "response": "Article 21 states: 'No person shall be deprived of his life or personal liberty except according to procedure established by law.' This is the most fundamental right and has been expansively interpreted to include right to livelihood, dignity, privacy, clean environment, health, education and more."},
        {"prompt": "What is Article 21A of the Indian Constitution?",
         "response": "Article 21A provides the Right to Education. It states that the State shall provide free and compulsory education to all children of the age of six to fourteen years in such manner as the State may, by law, determine."},
        {"prompt": "What is Article 32 of the Indian Constitution?",
         "response": "Article 32 provides the Right to Constitutional Remedies. It empowers citizens to move the Supreme Court for enforcement of Fundamental Rights. The Supreme Court can issue writs including habeas corpus, mandamus, prohibition, quo warranto and certiorari. Dr. B.R. Ambedkar called it the 'heart and soul' of the Constitution."},
        {"prompt": "What are Directive Principles of State Policy?",
         "response": "Directive Principles of State Policy (DPSP) are contained in Part IV (Articles 36-51) of the Constitution. They are guidelines for the State to follow while making laws and policies. Unlike Fundamental Rights, DPSPs are not enforceable in court but are fundamental in governance."},
        {"prompt": "What is Article 370 of the Indian Constitution?",
         "response": "Article 370 originally gave special autonomous status to Jammu and Kashmir. The Constitution (Application to Jammu and Kashmir) Order 2019, issued under the President's authority, effectively abrogated Article 370, extending all provisions of the Indian Constitution to J&K. This was upheld by the Supreme Court in December 2023."},
        {"prompt": "How many amendments have been made to the Indian Constitution?",
         "response": "As of May 2024, 106 amendments have been made to the Indian Constitution. The most recent is the Constitution (One Hundred and Sixth Amendment) Act, 2023 which relates to reservation for women in Lok Sabha and State Legislative Assemblies."},
        {"prompt": "What is the Preamble of the Indian Constitution?",
         "response": "The Preamble declares India as a Sovereign, Socialist, Secular, Democratic Republic and resolves to secure for all citizens: Justice (social, economic, political), Liberty (of thought, expression, belief, faith, worship), Equality (of status and opportunity), and Fraternity (assuring dignity of the individual and unity of the nation)."},
        {"prompt": "What are Fundamental Duties in the Indian Constitution?",
         "response": "Fundamental Duties are listed in Article 51A (Part IVA), added by the 42nd Amendment 1976. There are 11 duties including: respecting the Constitution and national symbols, upholding sovereignty, promoting harmony, preserving cultural heritage, protecting the environment, developing scientific temper, safeguarding public property, and striving towards excellence."},
        {"prompt": "What is the amendment process of the Indian Constitution?",
         "response": "Article 368 provides the amendment process. Bills can be introduced in either House. Some provisions need simple majority, some need special majority (2/3 of members present and voting + majority of total membership), and some also need ratification by at least half the State Legislatures."},
        {"prompt": "What is Article 356 of the Indian Constitution?",
         "response": "Article 356 provides for President's Rule in States. If the President is satisfied that the government of a State cannot be carried on according to the Constitution, they can issue a proclamation assuming all State government powers. It must be approved by Parliament within 2 months and can last up to 3 years."},
        {"prompt": "What is the 106th Amendment to the Indian Constitution?",
         "response": "The Constitution (One Hundred and Sixth Amendment) Act, 2023 provides for reservation of seats for women in the Lok Sabha and State Legislative Assemblies. It introduces new Articles 330A and 332A, reserving one-third of seats for women, to take effect after a delimitation exercise."},
        {"prompt": "What is Article 17 of the Indian Constitution?",
         "response": "Article 17 abolishes 'untouchability' and forbids its practice in any form. The enforcement of any disability arising out of 'untouchability' is an offence punishable in accordance with law. This is one of the absolute Fundamental Rights with no exceptions."},
        {"prompt": "What are the emergency provisions in the Indian Constitution?",
         "response": "The Constitution provides three types of emergencies: (1) National Emergency under Article 352 (war, external aggression, armed rebellion), (2) State Emergency/President's Rule under Article 356 (failure of constitutional machinery in States), and (3) Financial Emergency under Article 360 (threat to financial stability of India)."},
        {"prompt": "What is the structure of Indian judiciary under the Constitution?",
         "response": "The Constitution establishes a three-tier judiciary: Supreme Court (Articles 124-147) as the apex court, High Courts (Articles 214-231) for each State or group of States, and subordinate courts (Articles 233-237). The judiciary is independent of the executive and legislature."},
    ])
    return qa


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Building Indian Law Q&A Dataset from PDFs")
    print("=" * 60)

    # Load existing dataset
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
        print(f"Existing dataset: {len(existing)} entries")
    else:
        existing = []

    files_and_parsers = [
        ("raw_250883_english_01042024.txt", "BNS 2023", parse_bns),
        ("raw_250884_2_english_01042024.txt", "BNSS 2023", parse_bnss),
        ("raw_250882_english_01042024_0.txt", "BSA 2023", parse_bsa),
        ("raw_20240716890312078.txt", "Constitution", parse_constitution),
    ]

    all_new = []
    for filename, label, parser in files_and_parsers:
        filepath = os.path.join(RAW_DIR, filename)
        if not os.path.exists(filepath):
            print(f"  [SKIP] {filename}")
            continue

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        cleaned = clean_text(raw)
        print(f"\n{label}: {len(raw)} chars → {len(cleaned)} chars cleaned")

        qa = parser(cleaned)
        print(f"  Generated {len(qa)} Q&A pairs")
        all_new.extend(qa)

    # Merge
    combined = existing + all_new
    print(f"\n{'='*60}")
    print(f"Existing:  {len(existing)}")
    print(f"New:       {len(all_new)}")
    print(f"Total:     {len(combined)}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"Saved to {OUTPUT_PATH}")
    print(f"File size: {os.path.getsize(OUTPUT_PATH) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
