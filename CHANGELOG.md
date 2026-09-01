# Changelog

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
