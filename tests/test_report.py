from __future__ import annotations

import json

from doc_classifier import model, report
from doc_classifier.types import Document

from .conftest import corpus


class TestClassify:
    def test_returns_a_valid_prediction(self) -> None:
        fitted = model.fit(corpus(10))
        prediction = report.classify(
            fitted, Document(text="INVOICE amount due remit to accounts")
        )
        assert prediction.label in fitted.classes


class TestEvaluate:
    def test_perfect_model_on_its_own_training_data(self) -> None:
        fitted = model.fit(corpus(15))
        result = report.evaluate(fitted, corpus(15))
        assert result.total == 45
        assert result.accuracy == 1.0

    def test_unlabelled_documents_are_excluded_from_evaluation(self) -> None:
        fitted = model.fit(corpus(10))
        docs = corpus(10) + [Document(text="no label")]
        result = report.evaluate(fitted, docs)
        assert result.total == 30

    def test_confusion_matrix_only_counts_what_happened(self) -> None:
        fitted = model.fit(corpus(10))
        result = report.evaluate(fitted, corpus(10))
        assert all(count > 0 for count in result.confusion.values())
        assert sum(result.confusion.values()) == result.total

    def test_ece_is_a_valid_probability_scale_value(self) -> None:
        fitted = model.fit(corpus(10))
        result = report.evaluate(fitted, corpus(10))
        assert 0.0 <= result.ece <= 1.0


class TestRenderPrediction:
    def test_text_names_the_label_and_confidence(self) -> None:
        fitted = model.fit(corpus(10))
        prediction = report.classify(fitted, Document(text="INVOICE amount due"))
        text = report.render_prediction_text(prediction)
        assert prediction.label in text
        assert "%" in text

    def test_json_round_trips_through_to_dict(self) -> None:
        fitted = model.fit(corpus(10))
        prediction = report.classify(fitted, Document(text="INVOICE amount due"))
        payload = json.loads(report.render_prediction_json(prediction))
        assert payload["label"] == prediction.label
        assert payload["probabilities"] == prediction.to_dict()["probabilities"]


class TestRenderEval:
    def test_text_reports_accuracy_and_ece(self) -> None:
        fitted = model.fit(corpus(10))
        result = report.evaluate(fitted, corpus(10))
        text = report.render_eval_text(result)
        assert "correct" in text
        assert "ECE" in text

    def test_json_carries_the_confusion_matrix(self) -> None:
        fitted = model.fit(corpus(10))
        result = report.evaluate(fitted, corpus(10))
        payload = json.loads(report.render_eval_json(result))
        assert "confusion" in payload
        assert payload["total"] == result.total
