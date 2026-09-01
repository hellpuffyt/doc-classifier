"""Train a model and calibrate it, from one labelled set.

The calibration set is a per-class stratified hold-out from the training
data itself, not a second file the user must prepare: every class needs at
least one held-out example for `fit_temperature` to have something to
optimise against, and taking every Nth example within each class (in input
order) keeps every class represented in both halves without needing a
random seed to be reproducible.
"""

from __future__ import annotations

from collections import defaultdict

from . import calibration, model
from .errors import DataFormatError
from .types import Document

#: Default: roughly one in five labelled examples held out for calibration.
DEFAULT_HOLDOUT_FRACTION = 0.2

#: A class needs enough training examples for its statistics to mean
#: anything; fewer than this is a data problem worth naming, not a crash.
MIN_EXAMPLES_PER_CLASS = 2


def split_holdout(
    documents: list[Document], holdout_fraction: float
) -> tuple[list[Document], list[Document]]:
    """Per-class stratified split: every Nth example (by input order within
    its class) goes to the holdout, so every class with enough examples is
    represented in both halves."""
    if not 0.0 <= holdout_fraction < 1.0:
        raise DataFormatError(
            f"holdout fraction must be in [0, 1); got {holdout_fraction}"
        )
    by_class: dict[str, list[Document]] = defaultdict(list)
    for doc in documents:
        if doc.label is not None:
            by_class[doc.label].append(doc)

    # A class needs a positive stride of at least 2 or every example in it
    # goes to the holdout (stride 1 means "every example"). Guard this per
    # class rather than globally, since class sizes vary.
    stride = max(round(1 / holdout_fraction), 2) if holdout_fraction > 0 else 0
    train: list[Document] = []
    holdout: list[Document] = []
    for docs in by_class.values():
        if stride == 0 or len(docs) < 2:
            train.extend(docs)
            continue
        class_train: list[Document] = []
        class_holdout: list[Document] = []
        for index, doc in enumerate(docs):
            (class_holdout if (index + 1) % stride == 0 else class_train).append(doc)
        # A class small enough that the stride never lands keeps its
        # examples in training rather than contributing nothing to either
        # half's statistics.
        if not class_train:
            class_train, class_holdout = class_holdout, []
        train.extend(class_train)
        holdout.extend(class_holdout)
    return train, holdout


def fit_and_calibrate(
    documents: list[Document],
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
) -> model.NaiveBayesModel:
    labelled = [doc for doc in documents if doc.label is not None]
    if not labelled:
        raise DataFormatError("no labelled examples to train on")

    counts: dict[str, int] = defaultdict(int)
    for doc in labelled:
        counts[doc.label] += 1  # type: ignore[index]
    thin = sorted(label for label, n in counts.items() if n < MIN_EXAMPLES_PER_CLASS)
    if thin:
        raise DataFormatError(
            f"class(es) with fewer than {MIN_EXAMPLES_PER_CLASS} examples: "
            f"{', '.join(thin)} — add more labelled examples"
        )

    train_docs, holdout_docs = split_holdout(labelled, holdout_fraction)
    fitted = model.fit(train_docs)
    examples = calibration.calibration_examples(fitted, holdout_docs)
    fitted.temperature = calibration.fit_temperature(fitted, examples)
    return fitted
