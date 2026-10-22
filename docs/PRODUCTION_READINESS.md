# Production readiness — doc-classifier

Audit date: 2026-10-22. Tested commit: `39817a5` (branch `audit/production-readiness`, based on `main` @ `0a02fcd`).

## Scope

Stdlib-only multinomial/Gaussian/Bernoulli Naive Bayes over bag-of-words + layout features, with
temperature-scaling calibration on a stratified holdout, ECE reporting, JSON model files, and a CLI
(`train`, `classify`, `eval` with an accuracy gate, exit 3).

## Platforms

CPython 3.10+, no runtime dependencies. Verified on macOS 27 (Apple M5), CPython 3.12.14.

## Clean-clone verification (commit `39817a5`)

```bash
git clone -b audit/production-readiness <local clone> doc-classifier && cd doc-classifier
uv venv --python 3.12 .venv && . .venv/bin/activate && uv pip install .
doc-classifier train examples/documents.jsonl --out model.json        # T=0.460 on 4 held-out docs + warning
doc-classifier classify --model model.json "INVOICE #4471  Bill to: …" # invoice
doc-classifier eval --model model.json tests/fixtures/heldout_variants.jsonl --min-accuracy 0.99   # 15/16, exit 3 (gate works)
uv pip install -e ".[dev]" && pytest && ruff check . && mypy          # 94 passed / clean / clean
```

## Commands and results

| Command | Baseline (`0a02fcd`) | After (`39817a5`) |
| --- | --- | --- |
| `pytest` | exit 0 — 91 passed, 0 failed, 0 skipped | exit 0 — 94 passed, 0 failed, 0 skipped |
| `ruff check .` / `mypy` | exit 0 / exit 0 | exit 0 / exit 0 |
| `uvx pip-audit` | — | no known vulnerabilities |
| held-out variants (16 docs, 4/class) | 10/16, ECE 0.375 | 15/16, ECE 0.062 |

CI status — default branch: success (`0a02fcd`, 2026-09-01); audit branch: pending until pushed.

## Findings

- **README example was not real.** It claimed 40 training examples, temperature 2.14 and "invoice 87.3%";
  the bundled file has 24 examples, the real temperature is 0.460, and before the fix the same invoice
  text was classified **purchase_order at 100%**.
- **Root cause:** layout Gaussians for `table_line_ratio`/`header_line_ratio` had a std floor of 0.05; the
  single-line README text contains double spaces, so it is "100% table rows", which cost the invoice
  class ~200 nats against ~7 words of evidence. Floor raised to 0.25 — the resolution of a line ratio in
  a four-line document (justification independent of the evaluation set). Sweep on the held-out set:
  0.05 → 10/16, 0.10/0.15 → 12/16, 0.25 → 15/16, 0.35 → 14/16.
- **Calibration on the bundled data is not meaningful:** 4 held-out documents, all correct → T < 1
  (sharpening). `train` now prints the holdout size and warns below 30. ECE, temperature fitting and NLL
  formulas were reviewed and match Guo et al. (2017).

| Severity | Bug | Regression test |
| --- | --- | --- |
| High (core: confidently wrong) | layout std floor let one line override all vocabulary | `tests/test_heldout.py` (2 tests) |
| Medium (honesty) | README output fabricated; no signal that calibration used 4 docs | README real output; `tests/test_cli.py::test_train_reports_holdout_size_and_warns_when_small` |

Compatibility note: model JSON files store fitted std values, so models trained before this change keep
the old behaviour until retrained.

## Security review

Models are JSON (no pickle); no network. History secret scan: no hits. `pip-audit`: clean.

## Known limitations

- The bundled training set (24 docs) is a demo; e.g. a lowercase one-line purchase order is still
  classified `contract` (95.5%). Real accuracy/calibration depend on real training data, which was not available.
- The held-out fixture was written by the auditor; it measures robustness to layout changes, not real-world accuracy.

## Acceptance checklist

| # | Item | Expected | Evidence | Result |
| --- | --- | --- | --- | --- |
| 1 | train/classify/eval run | outputs | clean clone | PASS |
| 2 | eval gate | exit 3 below `--min-accuracy` | clean clone | PASS |
| 3 | vocabulary not swamped by layout | ≥ 14/16 held-out | test_heldout | PASS (15/16) |
| 4 | calibration reports its evidence | holdout size + warning | CLI test | PASS |
| 5 | calibrated confidence on real data | ECE small | no real dataset | NOT VERIFIED |

## Classification proposal

**Partially Verified** — mechanics verified; calibration quality cannot be shown with the bundled data.
