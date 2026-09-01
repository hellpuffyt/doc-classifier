from __future__ import annotations

import pytest

from doc_classifier import train
from doc_classifier.errors import DataFormatError
from doc_classifier.types import Document

from .conftest import corpus


class TestSplitHoldout:
    def test_every_class_appears_in_both_halves(self) -> None:
        train_docs, holdout_docs = train.split_holdout(corpus(10), 0.2)
        train_labels = {d.label for d in train_docs}
        holdout_labels = {d.label for d in holdout_docs}
        assert train_labels == {"contract", "invoice", "receipt"}
        assert holdout_labels == {"contract", "invoice", "receipt"}

    def test_split_is_a_partition_no_document_lost_or_duplicated(self) -> None:
        docs = corpus(10)
        train_docs, holdout_docs = train.split_holdout(docs, 0.2)
        assert len(train_docs) + len(holdout_docs) == len(docs)
        assert set(id(d) for d in train_docs) & set(id(d) for d in holdout_docs) == set()

    def test_roughly_matches_the_requested_fraction(self) -> None:
        train_docs, holdout_docs = train.split_holdout(corpus(50), 0.2)
        ratio = len(holdout_docs) / (len(train_docs) + len(holdout_docs))
        assert 0.1 < ratio < 0.3

    def test_a_tiny_class_still_keeps_at_least_one_training_example(self) -> None:
        docs = [
            Document(text="a", label="rare"),
            Document(text="b", label="rare"),
        ]
        train_docs, _ = train.split_holdout(docs, 0.99)
        assert any(d.label == "rare" for d in train_docs)

    def test_zero_fraction_puts_everything_in_training(self) -> None:
        train_docs, holdout_docs = train.split_holdout(corpus(5), 0.0)
        assert holdout_docs == []
        assert len(train_docs) == len(corpus(5))

    def test_out_of_range_fraction_is_refused(self) -> None:
        with pytest.raises(DataFormatError, match="holdout fraction"):
            train.split_holdout(corpus(5), 1.0)
        with pytest.raises(DataFormatError, match="holdout fraction"):
            train.split_holdout(corpus(5), -0.1)


class TestFitAndCalibrate:
    def test_produces_a_usable_model(self) -> None:
        fitted = train.fit_and_calibrate(corpus(15))
        assert fitted.classes == ["contract", "invoice", "receipt"]
        assert fitted.temperature > 0

    def test_no_labelled_examples_is_refused(self) -> None:
        with pytest.raises(DataFormatError, match="no labelled examples"):
            train.fit_and_calibrate([Document(text="no label here")])

    def test_a_thin_class_is_named_in_the_error(self) -> None:
        docs = corpus(5) + [Document(text="only one", label="lonely")]
        with pytest.raises(DataFormatError, match="lonely"):
            train.fit_and_calibrate(docs)
