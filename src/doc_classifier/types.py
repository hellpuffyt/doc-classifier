"""Core model: a labelled document, extracted features, and a prediction."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    """One document: its text and, optionally, the type it belongs to."""

    text: str
    label: str | None = None
    #: Where it came from, for error messages — never required.
    source: str = "<memory>"


@dataclass(frozen=True)
class Features:
    """Bag-of-words token counts plus a handful of layout signals.

    Layout features are kept separate from the word vocabulary rather than
    encoded as fake tokens, because they are measured on a different scale
    (a ratio in [0, 1], not a count) and the model weighs them accordingly.
    """

    tokens: dict[str, int] = field(default_factory=dict)
    #: Fraction of characters that are digits.
    digit_ratio: float = 0.0
    #: Fraction of lines that look like a table row (2+ runs of whitespace).
    table_line_ratio: float = 0.0
    #: Fraction of lines that are ALL CAPS headers.
    header_line_ratio: float = 0.0
    #: Average tokens per line — dense prose vs. sparse forms.
    avg_line_length: float = 0.0
    #: Has a currency symbol or amount-like pattern anywhere.
    has_currency: bool = False

    def token_total(self) -> int:
        return sum(self.tokens.values())


@dataclass(frozen=True)
class Prediction:
    """A classification with a full, checkable probability distribution."""

    label: str
    #: Calibrated probability of every label the model has seen, summing to 1.
    probabilities: dict[str, float]

    @property
    def confidence(self) -> float:
        return self.probabilities.get(self.label, 0.0)

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "probabilities": dict(
                sorted(self.probabilities.items(), key=lambda kv: -kv[1])
            ),
        }
