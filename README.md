# doc-classifier

Classify business documents by type from text and layout features, with a
training CLI and **calibrated** confidence output — a Naive Bayes classifier
whose confidence scores are corrected by temperature scaling so "90%"
actually means right about 90% of the time. Zero runtime dependencies —
Python 3.10+ and nothing else.

```console
$ doc-classifier train examples/documents.jsonl --out model.json
warning: fewer than 30 held-out examples; the temperature (and so the confidence scores) is not reliable
trained on 24 example(s), 4 class(es): contract, invoice, purchase_order, receipt
calibration temperature: 0.460 (fitted on 4 held-out example(s))
model written to model.json
$ doc-classifier classify --model model.json "INVOICE #4471  Bill to: Acme Corp  Amount Due: $1,204.00"
invoice  (confidence 100.0%)
  100.0%  invoice
    0.0%  purchase_order
    0.0%  contract
    0.0%  receipt
$ doc-classifier eval --model model.json tests/fixtures/heldout_variants.jsonl
15/16 correct (93.8% accuracy, ECE 0.062)

confused as:
     1  purchase_order -> contract
```

Real output. The bundled example set is tiny (24 documents), so the
calibration step sees only 4 held-out documents and *sharpens* the scores
(temperature < 1) rather than softening them — the tool says so. Calibrated
confidence needs a realistic training set (the warning disappears at 30+
held-out documents); the numbers above demonstrate the mechanics, not a
calibration result.

## Why calibration is the point, not a footnote

Naive Bayes is a famously *overconfident* classifier — its independence
assumption double-counts correlated evidence, so raw scores routinely say
"99.9%" about predictions that are wrong one time in five. `train` fits the
classifier, then holds out a stratified per-class slice and finds the
softmax temperature that minimises negative log-likelihood on it — one
scalar, found by golden-section search, that pulls the probabilities back
toward what actually happens. `eval` reports **Expected Calibration Error**
(ECE) alongside accuracy, so a claim of "well-calibrated" is checkable
rather than asserted.

## Features

- **Bag-of-words** tokens (crude, deliberately — domain vocabulary like
  "remit to" or "purchase order" is exactly what generic stopword lists
  strip).
- **Layout signals**: digit density, table-like line ratio, ALL-CAPS header
  line ratio, average line length, and whether a currency amount appears —
  each modelled as its own Gaussian or Bernoulli per class, not folded into
  the word vocabulary.

## Install

Not published on PyPI (the name `doc-classifier` there is an unrelated
project). Install from a clone:

```console
git clone https://github.com/hellpuffyt/doc-classifier && cd doc-classifier
python -m venv .venv && . .venv/bin/activate
pip install .
```

## Usage

```console
doc-classifier train data.jsonl --out model.json [--holdout-fraction 0.2]
doc-classifier classify --model model.json "some document text"
doc-classifier classify --model model.json < document.txt   # stdin
doc-classifier eval --model model.json held_out.csv [--min-accuracy 0.8]
```

Input format for `train`/`eval`: CSV (with header) or JSON Lines, columns
`text` (required) and `label` (required for train/eval, omitted when
`classify` reads a file directly isn't supported — `classify` takes text,
not a file).

Exit codes: `train`/`classify` return `0` on success, `1` on a usage or data
error. `eval` additionally returns `3` when accuracy falls below
`--min-accuracy` (default `0`, i.e. informational only) — the CI gate for
"did this model regress".

## What it deliberately does not do

- **No deep learning, no embeddings, no external model download.** A
  from-scratch, deterministic fit is the point: training is a closed-form
  computation, not a training run with a random seed to pin down.
- **No cross-validation search.** One smoothing constant, one calibration
  temperature, both named and defended in the code — not a grid search
  that would make two runs on the same data disagree.

## Development

```console
pip install -e ".[dev]"
pytest
ruff check .
mypy
```

Security: [SECURITY.md](SECURITY.md). Audit evidence:
[docs/PRODUCTION_READINESS.md](docs/PRODUCTION_READINESS.md).

MIT licensed.
