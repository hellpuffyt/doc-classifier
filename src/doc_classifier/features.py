"""Text and layout feature extraction.

Tokenisation is deliberately crude (lower-case word runs, stopwords kept) —
a business-document classifier's signal is dominated by domain vocabulary
("invoice", "remit to", "purchase order"), and stopword lists tuned for
general prose tend to strip exactly the boilerplate phrases that identify a
document type.
"""

from __future__ import annotations

import re
from collections import Counter

from .types import Document, Features

_WORD = re.compile(r"[a-z][a-z0-9']*")
_TABLE_ROW = re.compile(r"\S(?:\s{2,}|\t)\S")
_CURRENCY = re.compile(r"[$€£¥]\s?\d|\bUSD\b|\bEUR\b|\d+\.\d{2}\b")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def extract(document: Document) -> Features:
    text = document.text
    lines = text.splitlines() or [text]
    tokens = tokenize(text)

    table_lines = sum(1 for line in lines if _TABLE_ROW.search(line))
    header_lines = sum(
        1 for line in lines
        if line.strip() and line.strip() == line.strip().upper()
        and any(ch.isalpha() for ch in line)
    )
    digits = sum(1 for ch in text if ch.isdigit())

    return Features(
        tokens=dict(Counter(tokens)),
        digit_ratio=digits / len(text) if text else 0.0,
        table_line_ratio=table_lines / len(lines),
        header_line_ratio=header_lines / len(lines),
        avg_line_length=len(tokens) / len(lines),
        has_currency=bool(_CURRENCY.search(text)),
    )
