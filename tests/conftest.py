"""Shared fixtures: small, deterministic synthetic corpora."""

from __future__ import annotations

from doc_classifier.types import Document


def invoice(i: int) -> Document:
    return Document(
        text=(
            f"INVOICE #{1000 + i}\nBill To: Acme Corp\n"
            f"Amount Due: ${100 + i}.00\nremit to accounts payable department"
        ),
        label="invoice",
    )


def receipt(i: int) -> Document:
    return Document(
        text=(
            f"RECEIPT\nThank you for your purchase\n"
            f"Total Paid: ${20 + i}.00\ncashier {i} change given"
        ),
        label="receipt",
    )


def contract(i: int) -> Document:
    return Document(
        text=(
            f"CONTRACT AGREEMENT\nThis agreement dated section {i}\n"
            "witnesseth whereas the parties agree to the following terms"
        ),
        label="contract",
    )


def corpus(n_per_class: int = 20) -> list[Document]:
    docs: list[Document] = []
    for i in range(n_per_class):
        docs.extend([invoice(i), receipt(i), contract(i)])
    return docs
