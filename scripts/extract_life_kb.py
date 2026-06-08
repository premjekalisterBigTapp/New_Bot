#!/usr/bin/env python3
"""
One-time LOCAL extraction of the life-insurance brochure into a brand-neutral
knowledge-base text file used by the `life_insurance` intent.

The server never runs this script and never needs the PDF or PyMuPDF; it only
reads the committed output file `agentic/configs/life_insurance_kb.txt`.

Usage (from repo root, with the PDF available locally):
    python agentic/scripts/extract_life_kb.py \
        --pdf "Kotak/Kotak-e-Term-Plan-Brochure.pdf" \
        --out "agentic/configs/life_insurance_kb.txt"

The extracted text is scrubbed/rebranded so it never reveals the original
insurer: the product is presented as "Life Protect360" by "BigTapp", and
insurer-specific identifiers (UIN/CIN/IRDAI/registration/URLs/phones/addresses)
are removed.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PRODUCT_NAME = "Life Protect360"
BRAND_NAME = "BigTapp"

# Ordered brand-phrase replacements (longest / most specific first). Flexible
# whitespace (\s+) handles PDF line breaks that split a phrase across lines.
_PHRASE_REPLACEMENTS = [
    (r"Kotak\s+Critical\s+Illness\s+Plus\s+Benefit\s+Rider", "Critical Illness Plus Benefit Rider"),
    (r"Kotak\s+Permanent\s+Disability\s+Benefit\s+Rider", "Permanent Disability Benefit Rider"),
    (r"Kotak\s+e[-\s]*Term", PRODUCT_NAME),
    (r"Kotak\s+Mahindra\s+Life\s+Insurance\s+Company(?:'s|’s)", f"{BRAND_NAME}'s"),
    (r"Kotak\s+Mahindra\s+Life\s+Insurance\s+Company\s+Ltd\.?", BRAND_NAME),
    (r"Kotak\s+Mahindra\s+Bank\s+Limited", BRAND_NAME),
    (r"Kotak\s+Mahindra\s+Group", BRAND_NAME),
    (r"Kotak\s+Life\s+Insurance", BRAND_NAME),
    (r"Kotak\s+Group", BRAND_NAME),
    (r"Kotak", BRAND_NAME),  # catch-all, runs last
    (r"IRDAI", "the regulator"),  # neutralize regulator references that survive line-drop
]

# Identifier / boilerplate patterns to strip entirely.
_REMOVE_PATTERNS = [
    r"\(\s*UIN[^)]*\)",                       # (UIN: 107B002V03 ...)
    r"UIN\s*:?\s*[0-9A-Z]+V?\d*\.?,?",        # UIN: 107N129V03,
    r"https?://\S+",                           # URLs
    r"www\.[^\s)]+",                           # bare www. URLs
    r"CIN\s*:?\s*\S+",                         # CIN: U66030MH2000PLC128503
    r"Regn\.?\s*No\.?\s*:?\s*[0-9]+",          # Regn. No.:107
    r"Ref\.?\s*No\.?\s*:?\s*\S+",              # Ref. No.: KLI/...
    r"KLI/\S+",                                 # internal ref codes
    r"TOLL\s*FREE\s*[0-9 ]+",
    r"Toll\s*Free\s*No\.?\s*:?\s*[0-9 ]+",
    r"WhatsApp\s*:?\s*[0-9]+",
    r"\b1800[0-9 ]{3,}\b",
    r"\b18002098800\b",
    r"\b9321003007\b",
]

# Drop any line containing one of these (insurer legal/footer boilerplate).
_DROP_LINE_MARKERS = (
    "regd. office",
    "registered office",
    "trade logo",
    "irdai or its officials",
    "beware of spurious",
    "100% owned",
    "subsidiary of",
    "leading banking and financial",
    "group is one of",
    "growing insurance companies",
    "===== page",
)


def extract_text(pdf_path: Path) -> str:
    import fitz  # PyMuPDF (local-only dependency)

    doc = fitz.open(str(pdf_path))
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return "\n".join(pages)


def scrub(text: str) -> str:
    # 1) Rebrand phrases (flexible whitespace, case-sensitive on "Kotak").
    for pattern, repl in _PHRASE_REPLACEMENTS:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    # 2) Remove identifier patterns.
    for pattern in _REMOVE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

    # 3) Drop boilerplate lines.
    kept = []
    for line in text.splitlines():
        low = line.strip().lower()
        if any(marker in low for marker in _DROP_LINE_MARKERS):
            continue
        kept.append(line.rstrip())
    text = "\n".join(kept)

    # 4) Normalize whitespace: collapse 3+ blank lines to 1 blank line.
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract + scrub the life-insurance brochure.")
    parser.add_argument("--pdf", default="Kotak/Kotak-e-Term-Plan-Brochure.pdf")
    parser.add_argument("--out", default="agentic/configs/life_insurance_kb.txt")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    out_path = Path(args.out)
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}", file=sys.stderr)
        return 1

    raw = extract_text(pdf_path)
    cleaned = scrub(raw)

    header = (
        f"{PRODUCT_NAME} - Term Life Insurance (knowledge base)\n"
        f"Provider: {BRAND_NAME}\n"
        "Source of truth for life-insurance questions. All figures and terms below\n"
        "are authoritative; do not invent details beyond this document.\n"
        + ("=" * 70) + "\n\n"
    )
    final = header + cleaned + "\n"

    # Safety check: no original-brand leakage should remain.
    leaks = []
    for token in ("kotak", "uin", "irdai", "mahindra", "kotaklife"):
        if token in final.lower():
            leaks.append(token)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(final, encoding="utf-8")

    print(f"Wrote {out_path} ({len(final)} chars, ~{len(final)//4} tokens)")
    if leaks:
        print(f"WARNING: possible brand leakage tokens still present: {leaks}", file=sys.stderr)
        return 2
    print("OK: no brand-leak tokens detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
