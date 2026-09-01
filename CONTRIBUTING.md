# Contributing

Thanks for looking into doc-classifier.

## Setup

```console
python -m venv .venv && . .venv/bin/activate   # Scripts\activate on Windows
pip install -e ".[dev]"
```

## Before opening a pull request

```console
pytest            # the whole suite; new behaviour needs new tests
ruff check .      # lint
mypy              # strict — the src/ tree carries full annotations
```

## Ground rules

- **No runtime dependencies.** `dependencies = []` is a feature, not an
  oversight.
- **Calibration is not optional.** `train` always fits a temperature; a
  change that removes or bypasses it needs a very good reason and a test
  showing ECE stays checkable.
- **Model files are plain JSON, never pickle.** A classifier is the kind of
  artefact that gets committed and shared; loading one must never execute
  code.
- **Every constant is named and defended** (`TOKEN_SMOOTHING`,
  `LAYOUT_STD_FLOOR`, `MIN_EXAMPLES_PER_CLASS`, ...) with a comment
  explaining the number, especially per-feature floors — a single global
  floor silently favours whichever feature has the smallest natural scale.
