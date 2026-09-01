"""Serialise a fitted model to and from plain JSON.

JSON, not pickle: a classifier trained by one version of this tool must
never execute arbitrary code when loaded by another — model files are the
kind of artefact that gets committed, shared and downloaded from strangers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import ModelError
from .model import GaussianStats, NaiveBayesModel

#: Bumped only if the JSON shape changes incompatibly.
FORMAT_VERSION = 1


def to_dict(model: NaiveBayesModel) -> dict[str, object]:
    return {
        "format_version": FORMAT_VERSION,
        "classes": model.classes,
        "class_log_prior": model.class_log_prior,
        "token_log_likelihood": model.token_log_likelihood,
        "unseen_token_log_likelihood": model.unseen_token_log_likelihood,
        "currency_log_likelihood": {
            label: list(pair) for label, pair in model.currency_log_likelihood.items()
        },
        "layout_stats": {
            label: {
                name: {"mean": stats.mean, "std": stats.std}
                for name, stats in per_feature.items()
            }
            for label, per_feature in model.layout_stats.items()
        },
        "temperature": model.temperature,
    }


def from_dict(payload: dict[str, Any]) -> NaiveBayesModel:
    try:
        version = payload["format_version"]
        if version != FORMAT_VERSION:
            raise ModelError(
                f"model format version {version} is not supported "
                f"(expected {FORMAT_VERSION})"
            )
        currency = {
            label: (pair[0], pair[1])
            for label, pair in payload["currency_log_likelihood"].items()
        }
        layout = {
            label: {
                name: GaussianStats(mean=stats["mean"], std=stats["std"])
                for name, stats in per_feature.items()
            }
            for label, per_feature in payload["layout_stats"].items()
        }
        return NaiveBayesModel(
            classes=list(payload["classes"]),
            class_log_prior=dict(payload["class_log_prior"]),
            token_log_likelihood=dict(payload["token_log_likelihood"]),
            unseen_token_log_likelihood=dict(payload["unseen_token_log_likelihood"]),
            currency_log_likelihood=currency,
            layout_stats=layout,
            temperature=float(payload["temperature"]),
        )
    except (KeyError, TypeError, ValueError) as err:
        raise ModelError(f"model file is malformed: {err}") from err


def save(model: NaiveBayesModel, path: str | Path) -> None:
    Path(path).write_text(json.dumps(to_dict(model), indent=2), encoding="utf-8")


def load(path: str | Path) -> NaiveBayesModel:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as err:
        raise ModelError(f"cannot read {path}: {err.strerror or err}") from err
    except json.JSONDecodeError as err:
        raise ModelError(f"{path}: not valid JSON: {err.msg}") from err
    if not isinstance(raw, dict):
        raise ModelError(f"{path}: not a model file")
    return from_dict(raw)
