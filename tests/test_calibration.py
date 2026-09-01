from __future__ import annotations

import pytest

from doc_classifier import calibration, model
from doc_classifier.features import extract

from .conftest import corpus


class TestExpectedCalibrationError:
    def test_perfectly_calibrated_predictions_score_zero(self) -> None:
        # Every bucket's average confidence exactly matches its accuracy.
        predictions = [(0.9, True)] * 90 + [(0.9, False)] * 10
        assert calibration.expected_calibration_error(predictions) == pytest.approx(
            0.0, abs=1e-9
        )

    def test_overconfident_predictions_score_positive(self) -> None:
        # Confidence 0.99 but only right half the time.
        predictions = [(0.99, True)] * 50 + [(0.99, False)] * 50
        assert calibration.expected_calibration_error(predictions) == pytest.approx(
            0.49, abs=1e-2
        )

    def test_empty_predictions_score_zero(self) -> None:
        assert calibration.expected_calibration_error([]) == 0.0

    def test_weighting_by_bucket_size(self) -> None:
        # One badly-miscalibrated example in an otherwise-perfect set should
        # move the score only a little.
        predictions = [(0.9, True)] * 99 + [(0.9, False)]
        small = calibration.expected_calibration_error(predictions)
        predictions_worse = [(0.9, True)] * 50 + [(0.9, False)] * 50
        large = calibration.expected_calibration_error(predictions_worse)
        assert small < large


class TestFitTemperature:
    def test_no_examples_returns_identity_temperature(self) -> None:
        fitted = model.fit(corpus(10))
        assert calibration.fit_temperature(fitted, []) == 1.0

    def test_temperature_is_positive_and_finite(self) -> None:
        fitted = model.fit(corpus(15))
        examples = [
            (extract(doc), doc.label)
            for doc in corpus(15)
            if doc.label is not None
        ]
        t = calibration.fit_temperature(fitted, examples)
        assert 0.0 < t < float("inf")

    def test_calibrating_never_makes_held_out_nll_worse_than_identity(self) -> None:
        fitted = model.fit(corpus(20))
        examples = [
            (extract(doc), doc.label)
            for doc in corpus(20)
            if doc.label is not None
        ]
        identity_nll = calibration.negative_log_likelihood(fitted, examples, 1.0)
        t = calibration.fit_temperature(fitted, examples)
        calibrated_nll = calibration.negative_log_likelihood(fitted, examples, t)
        assert calibrated_nll <= identity_nll + 1e-9


class TestNegativeLogLikelihood:
    def test_confident_correct_predictions_have_low_nll(self) -> None:
        fitted = model.fit(corpus(15))
        confident_examples = [
            (extract(doc), doc.label) for doc in corpus(15) if doc.label is not None
        ]
        low = calibration.negative_log_likelihood(fitted, confident_examples, 1.0)
        # A far higher temperature flattens toward uniform, raising NLL for
        # examples the model would otherwise get right confidently.
        high = calibration.negative_log_likelihood(fitted, confident_examples, 50.0)
        assert low < high
