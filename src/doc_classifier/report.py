"""Render classification and evaluation results."""

from __future__ import annotations

import json
from dataclasses import dataclass

from . import calibration
from .features import extract
from .model import NaiveBayesModel
from .types import Document, Prediction


def classify(model: NaiveBayesModel, document: Document) -> Prediction:
    return model.predict(extract(document))


@dataclass(frozen=True)
class EvalResult:
    total: int
    correct: int
    ece: float
    #: (true_label, predicted_label) counts — a confusion matrix, sparse.
    confusion: dict[tuple[str, str], int]

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "expected_calibration_error": self.ece,
            "confusion": [
                {"true": t, "predicted": p, "count": c}
                for (t, p), c in sorted(self.confusion.items())
            ],
        }


def evaluate(model: NaiveBayesModel, documents: list[Document]) -> EvalResult:
    labelled = [doc for doc in documents if doc.label is not None]
    predictions: list[tuple[float, bool]] = []
    confusion: dict[tuple[str, str], int] = {}
    correct = 0
    for doc in labelled:
        prediction = classify(model, doc)
        is_correct = prediction.label == doc.label
        correct += int(is_correct)
        predictions.append((prediction.confidence, is_correct))
        # doc.label is guaranteed non-None here: `labelled` above filters
        # to exactly the documents that carry one.
        assert doc.label is not None
        key = (doc.label, prediction.label)
        confusion[key] = confusion.get(key, 0) + 1
    return EvalResult(
        total=len(labelled),
        correct=correct,
        ece=calibration.expected_calibration_error(predictions),
        confusion=confusion,
    )


def render_prediction_text(prediction: Prediction) -> str:
    lines = [f"{prediction.label}  (confidence {prediction.confidence:.1%})"]
    for label, probability in sorted(
        prediction.probabilities.items(), key=lambda kv: -kv[1]
    ):
        lines.append(f"  {probability:6.1%}  {label}")
    return "\n".join(lines)


def render_prediction_json(prediction: Prediction) -> str:
    return json.dumps(prediction.to_dict(), indent=2)


def render_eval_text(result: EvalResult) -> str:
    lines = [
        f"{result.correct}/{result.total} correct "
        f"({result.accuracy:.1%} accuracy, ECE {result.ece:.3f})",
    ]
    misclassified = {
        key: count for key, count in result.confusion.items() if key[0] != key[1]
    }
    if misclassified:
        lines.append("")
        lines.append("confused as:")
        for (true, predicted), count in sorted(
            misclassified.items(), key=lambda kv: -kv[1]
        ):
            lines.append(f"  {count:4d}  {true} -> {predicted}")
    return "\n".join(lines)


def render_eval_json(result: EvalResult) -> str:
    return json.dumps(result.to_dict(), indent=2)
