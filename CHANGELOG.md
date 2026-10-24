# Changelog

## Unreleased

- Fixed: the standard-deviation floor on the two line-ratio layout features
  (0.05) was far below their resolution (one line of a four-line document is
  0.25), so a single double-spaced line outweighed every word. The README's own
  invoice example was classified `purchase_order` at 100%; 10/16 held-out
  variants were correct. Floors are now 0.25: 15/16, ECE 0.375 -> 0.062.
  Models trained with the old floors keep their stored statistics; retrain.
- Added: `train` reports how many held-out documents the temperature was fitted
  on and warns below 30.
- Docs: README example replaced with real output and an honest note on
  calibration with tiny datasets; install from source. Added held-out variants
  fixture and tests, `SECURITY.md`, `.editorconfig`,
  `docs/PRODUCTION_READINESS.md`.

## 0.1.0 — 2026-09-01

Initial release.

- Naive Bayes classifier over bag-of-words tokens plus five layout features
  (digit density, table-line ratio, header-line ratio, average line length,
  currency presence), each modelled as its own Gaussian or Bernoulli
  distribution per class rather than folded into the word vocabulary.
- Temperature scaling for calibration: `train` fits on a per-class
  stratified hold-out and finds the softmax temperature minimising
  negative log-likelihood via golden-section search — one scalar, found
  deterministically, no gradient descent.
- `eval` reports accuracy alongside **Expected Calibration Error**, so a
  calibration claim is checkable rather than asserted.
- `doc-classifier train|classify|eval` CLI over CSV / JSON Lines, with
  `classify` accepting text as an argument or via stdin. Model files persist
  as plain JSON — never pickle — so loading one never executes code.
- Zero runtime dependencies; Python 3.10–3.13.
