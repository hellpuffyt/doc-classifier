# Security policy

## Supported versions

Only the latest commit on `main` receives fixes. There are no maintained release branches.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's advisory form:
https://github.com/hellpuffyt/doc-classifier/security/advisories/new

Do not open a public issue for a suspected vulnerability. Include the version or commit,
a minimal reproduction, and the impact you observed. You should receive an acknowledgement
within a few days; fixes are released as soon as they are verified.

## Scope

doc-classifier reads local CSV/JSONL training data and JSON model files and prints predictions; no network access, no pickle (models are JSON). Relevant reports include: a crafted model or data file that executes code, writes outside `--out`, or exhausts resources.
