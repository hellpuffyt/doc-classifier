"""A Naive Bayes classifier over bag-of-words tokens and Gaussian layout
features, with temperature scaling for calibration.

Naive Bayes is chosen deliberately over anything fancier: it is a closed-form
fit (no gradient descent, no random initialisation, no hyperparameter search
beyond the smoothing constant), so training is deterministic and fast enough
to run in CI without a GPU or a dependency on one. What it gets wrong on its
own — Naive Bayes is a famously *overconfident* classifier, because its
independence assumption double-counts correlated evidence — is exactly what
temperature scaling (`calibration.py`) exists to fix after the fact.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from .features import extract
from .types import Document, Features, Prediction

#: Laplace smoothing constant for token counts. 1.0 is the classic choice:
#: enough to give every unseen token *some* probability without swamping
#: the signal from tokens that actually appear.
TOKEN_SMOOTHING = 1.0

#: Floor on a layout feature's estimated standard deviation, per feature —
#: a feature that is perfectly constant within a class (std = 0) would
#: otherwise assign near-zero probability to any real-world document that
#: differs by a hair. One constant does not fit every feature here: the
#: three ratio features live in [0, 1] and a small floor is proportionate,
#: but avg_line_length is an unbounded token count, where the same floor
#: would make a one-token difference look like a 15-standard-deviation
#: outlier and let layout swamp the token evidence entirely.
LAYOUT_STD_FLOOR: dict[str, float] = {
    "digit_ratio": 0.05,
    "table_line_ratio": 0.05,
    "header_line_ratio": 0.05,
    "avg_line_length": 1.0,
}

LAYOUT_FEATURES: tuple[str, ...] = tuple(LAYOUT_STD_FLOOR)


@dataclass
class GaussianStats:
    mean: float
    std: float

    def log_density(self, value: float) -> float:
        variance = self.std * self.std
        return -0.5 * math.log(2 * math.pi * variance) - (value - self.mean) ** 2 / (
            2 * variance
        )


@dataclass
class NaiveBayesModel:
    """A fitted classifier. Every field is plain data — no closures, no
    references to the training set — so it serialises to JSON exactly."""

    classes: list[str]
    class_log_prior: dict[str, float]
    #: log P(token | class) for tokens seen in that class.
    token_log_likelihood: dict[str, dict[str, float]]
    #: log P(unseen token | class) — the smoothed floor.
    unseen_token_log_likelihood: dict[str, float]
    #: currency-present rate per class, as a Bernoulli log-likelihood pair
    #: (log P(has_currency=True), log P(has_currency=False)).
    currency_log_likelihood: dict[str, tuple[float, float]]
    layout_stats: dict[str, dict[str, GaussianStats]]
    #: Softmax temperature; 1.0 until `calibration.fit_temperature` runs.
    temperature: float = 1.0

    def raw_scores(self, features: Features) -> dict[str, float]:
        """Unnormalised log-scores per class — the Naive Bayes posterior
        numerator, before softmax. Public so calibration can fit against
        it without re-running feature extraction."""
        scores: dict[str, float] = {}
        for label in self.classes:
            score = self.class_log_prior[label]
            likelihoods = self.token_log_likelihood[label]
            floor = self.unseen_token_log_likelihood[label]
            for token, count in features.tokens.items():
                score += count * likelihoods.get(token, floor)

            stats = self.layout_stats[label]
            for name in LAYOUT_FEATURES:
                score += stats[name].log_density(getattr(features, name))

            true_ll, false_ll = self.currency_log_likelihood[label]
            score += true_ll if features.has_currency else false_ll

            scores[label] = score
        return scores

    def predict(self, features: Features) -> Prediction:
        scores = self.raw_scores(features)
        scaled = {label: score / self.temperature for label, score in scores.items()}
        probabilities = softmax(scaled)
        best = max(probabilities, key=lambda label: probabilities[label])
        return Prediction(label=best, probabilities=probabilities)


def softmax(scores: dict[str, float]) -> dict[str, float]:
    """Numerically stable softmax over a label -> log-score mapping."""
    top = max(scores.values())
    exp = {label: math.exp(score - top) for label, score in scores.items()}
    total = sum(exp.values())
    return {label: value / total for label, value in exp.items()}


def fit(documents: list[Document]) -> NaiveBayesModel:
    """Fit token, layout and currency statistics from labelled documents."""
    by_class: dict[str, list[Document]] = defaultdict(list)
    for document in documents:
        if document.label is None:
            continue
        by_class[document.label].append(document)

    classes = sorted(by_class)
    total = sum(len(docs) for docs in by_class.values())

    class_log_prior = {
        label: math.log(len(docs) / total) for label, docs in by_class.items()
    }

    token_log_likelihood: dict[str, dict[str, float]] = {}
    unseen_token_log_likelihood: dict[str, float] = {}
    currency_log_likelihood: dict[str, tuple[float, float]] = {}
    layout_stats: dict[str, dict[str, GaussianStats]] = {}

    vocabulary: set[str] = set()
    per_class_features: dict[str, list[Features]] = {}
    for label, docs in by_class.items():
        feats = [extract(doc) for doc in docs]
        per_class_features[label] = feats
        for f in feats:
            vocabulary.update(f.tokens)
    vocab_size = max(len(vocabulary), 1)

    for label, feats in per_class_features.items():
        counts: Counter[str] = Counter()
        for f in feats:
            counts.update(f.tokens)
        class_token_total = sum(counts.values())
        denom = class_token_total + TOKEN_SMOOTHING * (vocab_size + 1)
        token_log_likelihood[label] = {
            token: math.log((count + TOKEN_SMOOTHING) / denom)
            for token, count in counts.items()
        }
        unseen_token_log_likelihood[label] = math.log(TOKEN_SMOOTHING / denom)

        currency_true = sum(1 for f in feats if f.has_currency)
        n = len(feats)
        # Laplace-smoothed Bernoulli, same rationale as the token smoothing.
        p_true = (currency_true + 1) / (n + 2)
        currency_log_likelihood[label] = (math.log(p_true), math.log(1 - p_true))

        layout_stats[label] = {
            name: _gaussian(name, feats) for name in LAYOUT_FEATURES
        }

    return NaiveBayesModel(
        classes=classes,
        class_log_prior=class_log_prior,
        token_log_likelihood=token_log_likelihood,
        unseen_token_log_likelihood=unseen_token_log_likelihood,
        currency_log_likelihood=currency_log_likelihood,
        layout_stats=layout_stats,
    )


def _gaussian(name: str, feats: list[Features]) -> GaussianStats:
    values = [getattr(f, name) for f in feats]
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    floor = LAYOUT_STD_FLOOR[name]
    return GaussianStats(mean=mean, std=max(math.sqrt(variance), floor))
