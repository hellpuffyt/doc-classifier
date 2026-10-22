"""Console entry point: train, classify, eval.

Exit codes:

* ``train``   — 0 trained and saved, 1 usage/input/data error
* ``classify``— 0 always (a prediction is not a pass/fail judgement), 1 error
* ``eval``    — 0 accuracy at or above ``--min-accuracy`` (default: 0, i.e.
  informational only), 1 error, 3 below the threshold
"""

from __future__ import annotations

import argparse
import sys

from . import __version__, io, persist, report, train
from .errors import DocClassifierError
from .types import Document

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_BELOW_THRESHOLD = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="doc-classifier",
        description=(
            "Classify business documents by type from text and layout "
            "features, with a training CLI and calibrated confidence output."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="fit and calibrate a model")
    p_train.add_argument("data", help="labelled training data (.csv, .jsonl)")
    p_train.add_argument("--out", required=True, help="path to write the model JSON to")
    p_train.add_argument(
        "--holdout-fraction", type=float, default=train.DEFAULT_HOLDOUT_FRACTION,
        help="fraction of each class held out for calibration (default: 0.2)",
    )

    p_classify = sub.add_parser("classify", help="classify one document")
    p_classify.add_argument("--model", required=True, help="path to a trained model JSON")
    p_classify.add_argument(
        "text", nargs="?", help="document text; reads stdin if omitted",
    )
    p_classify.add_argument("--format", choices=("text", "json"), default="text")

    p_eval = sub.add_parser("eval", help="evaluate a model against labelled data")
    p_eval.add_argument("--model", required=True, help="path to a trained model JSON")
    p_eval.add_argument("data", help="labelled evaluation data (.csv, .jsonl)")
    p_eval.add_argument("--format", choices=("text", "json"), default="text")
    p_eval.add_argument(
        "--min-accuracy", type=float, default=0.0,
        help="exit 3 if accuracy falls below this (default: 0, informational)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train":
            return _train(args)
        if args.command == "classify":
            return _classify(args)
        if args.command == "eval":
            return _eval(args)
        raise AssertionError(f"unknown command {args.command!r}")
    except DocClassifierError as err:
        print(f"error: {err}", file=sys.stderr)
        return EXIT_USAGE


#: Below this many held-out documents a fitted temperature mostly reflects
#: which few examples landed in the holdout, not the model's calibration.
MIN_RELIABLE_HOLDOUT = 30


def _train(args: argparse.Namespace) -> int:
    documents = io.load(args.data)
    model = train.fit_and_calibrate(documents, args.holdout_fraction)
    persist.save(model, args.out)
    print(
        f"trained on {sum(1 for d in documents if d.label)} example(s), "
        f"{len(model.classes)} class(es): {', '.join(model.classes)}"
    )
    labelled = [d for d in documents if d.label]
    _, holdout = train.split_holdout(labelled, args.holdout_fraction)
    print(
        f"calibration temperature: {model.temperature:.3f} "
        f"(fitted on {len(holdout)} held-out example(s))"
    )
    if len(holdout) < MIN_RELIABLE_HOLDOUT:
        print(
            f"warning: fewer than {MIN_RELIABLE_HOLDOUT} held-out examples; the "
            "temperature (and so the confidence scores) is not reliable",
            file=sys.stderr,
        )
    print(f"model written to {args.out}")
    return EXIT_OK


def _classify(args: argparse.Namespace) -> int:
    text = args.text if args.text is not None else sys.stdin.read()
    if not text.strip():
        raise DocClassifierError("no text to classify (empty argument and empty stdin)")
    model = persist.load(args.model)
    prediction = report.classify(model, Document(text=text))
    print(
        report.render_prediction_json(prediction)
        if args.format == "json"
        else report.render_prediction_text(prediction)
    )
    return EXIT_OK


def _eval(args: argparse.Namespace) -> int:
    model = persist.load(args.model)
    documents = io.load(args.data)
    result = report.evaluate(model, documents)
    print(
        report.render_eval_json(result)
        if args.format == "json"
        else report.render_eval_text(result)
    )
    return EXIT_BELOW_THRESHOLD if result.accuracy < args.min_accuracy else EXIT_OK
