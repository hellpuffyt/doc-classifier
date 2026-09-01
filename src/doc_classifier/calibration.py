"""Temperature scaling: one scalar that turns overconfident Naive Bayes
scores into probabilities that mean what they say.

Fitting minimises negative log-likelihood on held-out documents over the
temperature `T`, dividing every raw log-score by `T` before the softmax. In
raw-score space this is convex in `log(T)` for the same reason ordinary
temperature scaling is (Guo et al., 2017) — so a bounded golden-section
search over `log(T)` finds the minimum without gradients, without
overshoot, and deterministically.
"""

from __future__ import annotations

import math

from .model import NaiveBayesModel, softmax
from .types import Document, Features

#: Search bounds for T. Below 0.05 the softmax is effectively one-hot;
#: above 20 it is effectively uniform. Real fits land well inside this.
_LOG_T_MIN = math.log(0.05)
_LOG_T_MAX = math.log(20.0)
_GOLDEN = (math.sqrt(5) - 1) / 2
_ITERATIONS = 40


def negative_log_likelihood(
    model: NaiveBayesModel, examples: list[tuple[Features, str]], temperature: float
) -> float:
    total = 0.0
    for features, true_label in examples:
        scores = model.raw_scores(features)
        scaled = {label: score / temperature for label, score in scores.items()}
        probabilities = softmax(scaled)
        # Held-out examples always carry a real label from training data;
        # a label the model never saw would be a data problem, not a
        # calibration one, so it is a bug to hide rather than clamp away.
        total -= math.log(max(probabilities[true_label], 1e-12))
    return total / len(examples)


def fit_temperature(
    model: NaiveBayesModel, examples: list[tuple[Features, str]]
) -> float:
    """The temperature minimising NLL on `examples`, found by golden-section
    search over log(T). Returns 1.0 (no rescaling) if there is nothing to
    calibrate against."""
    if not examples:
        return 1.0

    def nll(log_t: float) -> float:
        return negative_log_likelihood(model, examples, math.exp(log_t))

    lo, hi = _LOG_T_MIN, _LOG_T_MAX
    c = hi - _GOLDEN * (hi - lo)
    d = lo + _GOLDEN * (hi - lo)
    fc, fd = nll(c), nll(d)
    for _ in range(_ITERATIONS):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - _GOLDEN * (hi - lo)
            fc = nll(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + _GOLDEN * (hi - lo)
            fd = nll(d)
    return math.exp((lo + hi) / 2)


def expected_calibration_error(
    predictions: list[tuple[float, bool]], buckets: int = 10
) -> float:
    """ECE: |confidence - accuracy| averaged over confidence buckets,
    weighted by bucket size — the standard reliability-diagram summary.

    ``predictions`` is (confidence, was_correct) pairs. Empty buckets
    contribute nothing, which is the honest treatment: there is no evidence
    to be wrong about in a bucket nothing landed in.
    """
    if not predictions:
        return 0.0
    bucket_conf = [0.0] * buckets
    bucket_correct = [0] * buckets
    bucket_count = [0] * buckets
    for confidence, correct in predictions:
        index = min(int(confidence * buckets), buckets - 1)
        bucket_conf[index] += confidence
        bucket_correct[index] += int(correct)
        bucket_count[index] += 1

    total = len(predictions)
    error = 0.0
    triples = zip(bucket_count, bucket_conf, bucket_correct, strict=True)
    for count, conf_sum, correct_count in triples:
        if count == 0:
            continue
        avg_conf = conf_sum / count
        accuracy = correct_count / count
        error += (count / total) * abs(avg_conf - accuracy)
    return error


def calibration_examples(
    model: NaiveBayesModel, documents: list[Document]
) -> list[tuple[Features, str]]:
    from .features import extract

    return [
        (extract(doc), doc.label)
        for doc in documents
        if doc.label is not None
    ]
