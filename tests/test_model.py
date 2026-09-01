from __future__ import annotations

import math

import pytest

from doc_classifier import model
from doc_classifier.features import extract
from doc_classifier.types import Document

from .conftest import corpus


class TestSoftmax:
    def test_sums_to_one(self) -> None:
        probs = model.softmax({"a": 2.0, "b": 1.0, "c": 0.5})
        assert sum(probs.values()) == pytest.approx(1.0)

    def test_equal_scores_are_uniform(self) -> None:
        probs = model.softmax({"a": 1.0, "b": 1.0})
        assert probs["a"] == pytest.approx(0.5)
        assert probs["b"] == pytest.approx(0.5)

    def test_numerically_stable_for_large_scores(self) -> None:
        probs = model.softmax({"a": 10000.0, "b": 9999.0})
        assert math.isfinite(probs["a"])
        assert sum(probs.values()) == pytest.approx(1.0)

    def test_highest_score_gets_highest_probability(self) -> None:
        probs = model.softmax({"a": 5.0, "b": -5.0})
        assert probs["a"] > probs["b"]


class TestFit:
    def test_learns_every_class(self) -> None:
        fitted = model.fit(corpus(10))
        assert fitted.classes == ["contract", "invoice", "receipt"]

    def test_priors_reflect_class_balance(self) -> None:
        docs = corpus(10) + [
            Document(text="invoice invoice invoice extra", label="invoice")
        ]
        fitted = model.fit(docs)
        assert fitted.class_log_prior["invoice"] > fitted.class_log_prior["receipt"]

    def test_unlabelled_documents_are_ignored(self) -> None:
        docs = corpus(5) + [Document(text="mystery document")]
        fitted = model.fit(docs)
        assert "unlabelled" not in fitted.classes
        assert set(fitted.classes) == {"contract", "invoice", "receipt"}

    def test_a_clearly_typical_document_predicts_its_class(self) -> None:
        fitted = model.fit(corpus(15))
        features = extract(
            Document(text="INVOICE #500\nBill To: X\nAmount Due: $50.00\nremit to accounts")
        )
        assert fitted.predict(features).label == "invoice"

    def test_layout_stats_have_a_std_floor(self) -> None:
        # Every training document in a class has the exact same digit_ratio
        # -> raw variance is 0, and the floor must keep it usable.
        docs = [
            Document(text="AAAA no digits here at all", label="text")
            for _ in range(5)
        ] + [
            Document(text="BBBB also nothing numeric", label="other")
            for _ in range(5)
        ]
        fitted = model.fit(docs)
        floor = model.LAYOUT_STD_FLOOR["digit_ratio"]
        assert fitted.layout_stats["text"]["digit_ratio"].std == floor


class TestPredict:
    def test_probabilities_sum_to_one(self) -> None:
        fitted = model.fit(corpus(10))
        features = extract(Document(text="some ambiguous short text"))
        prediction = fitted.predict(features)
        assert sum(prediction.probabilities.values()) == pytest.approx(1.0)

    def test_label_is_the_argmax_probability(self) -> None:
        fitted = model.fit(corpus(10))
        features = extract(
            Document(text="RECEIPT thank you total paid change given cashier")
        )
        prediction = fitted.predict(features)
        assert prediction.label == max(
            prediction.probabilities, key=lambda label: prediction.probabilities[label]
        )

    def test_higher_temperature_flattens_confidence(self) -> None:
        fitted = model.fit(corpus(10))
        text = "mostly ambiguous words that hint faintly at invoice"
        features = extract(Document(text=text))
        fitted.temperature = 1.0
        sharp = fitted.predict(features).confidence
        fitted.temperature = 50.0
        flat = fitted.predict(features).confidence
        assert flat < sharp

    def test_unseen_tokens_fall_back_to_the_smoothed_floor(self) -> None:
        fitted = model.fit(corpus(10))
        features = extract(Document(text="zzzqwerty nonsense token stream"))
        prediction = fitted.predict(features)
        assert prediction.label in fitted.classes  # does not crash, picks something


class TestGaussianStats:
    def test_log_density_peaks_at_the_mean(self) -> None:
        stats = model.GaussianStats(mean=0.5, std=0.1)
        assert stats.log_density(0.5) > stats.log_density(0.8)
        assert stats.log_density(0.5) > stats.log_density(0.2)
