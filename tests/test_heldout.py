"""Generalisation on documents the model was not trained on.

`tests/fixtures/heldout_variants.jsonl` holds 16 hand-written documents (4 per
class) that reuse each class's vocabulary but change its layout: single-line
text with double-space separators, different capitalisation, no line breaks.
None of them appears in `examples/documents.jsonl`.
"""

from __future__ import annotations

from pathlib import Path

from doc_classifier import io, train
from doc_classifier.features import extract

ROOT = Path(__file__).resolve().parent.parent


def _model_and_heldout():  # type: ignore[no-untyped-def]
    model = train.fit_and_calibrate(io.load(ROOT / "examples" / "documents.jsonl"))
    return model, io.load(ROOT / "tests" / "fixtures" / "heldout_variants.jsonl")


def test_layout_does_not_swamp_vocabulary_on_held_out_documents() -> None:
    # With a 0.05 std floor on the line ratios, one double-spaced line read
    # as a "table row" outweighed every word: 10/16 correct, and the README's
    # own invoice example was classified purchase_order at 100%.
    model, heldout = _model_and_heldout()
    correct = sum(model.predict(extract(d)).label == d.label for d in heldout)
    assert correct >= 14, correct


def test_readme_invoice_example_is_an_invoice() -> None:
    model, heldout = _model_and_heldout()
    assert heldout[0].text.startswith("INVOICE #4471")
    assert model.predict(extract(heldout[0])).label == "invoice"
