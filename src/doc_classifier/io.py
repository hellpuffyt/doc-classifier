"""Load labelled (or unlabelled) documents from CSV or JSON Lines.

Required: ``text``. ``label`` is optional — its absence is what
distinguishes a classification request from a training/evaluation example.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .errors import DataFormatError
from .types import Document


def load(path: str | Path) -> list[Document]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_csv(path)
    if suffix in (".jsonl", ".ndjson"):
        return _load_jsonl(path)
    raise DataFormatError(
        f"unsupported input {path.name!r}: expected .csv, .jsonl or .ndjson"
    )


def _load_csv(path: Path) -> list[Document]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise DataFormatError(f"{path.name}: file is empty")
            if "text" not in reader.fieldnames:
                raise DataFormatError(f"{path.name}: header lacks required column 'text'")
            documents = []
            for number, row in enumerate(reader, start=2):
                text = (row.get("text") or "").strip()
                if not text:
                    raise DataFormatError(f"{path.name}:{number}: empty 'text'")
                label = (row.get("label") or "").strip() or None
                documents.append(
                    Document(text=text, label=label, source=f"{path.name}:{number}")
                )
    except OSError as err:
        raise DataFormatError(f"cannot read {path}: {err.strerror or err}") from err
    if not documents:
        raise DataFormatError(f"{path.name}: no data rows")
    return documents


def _load_jsonl(path: Path) -> list[Document]:
    documents = []
    try:
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                where = f"{path.name}:{number}"
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as err:
                    raise DataFormatError(f"{where}: invalid JSON: {err.msg}") from err
                if not isinstance(obj, dict) or not isinstance(obj.get("text"), str):
                    raise DataFormatError(f"{where}: expected an object with a string 'text'")
                text = obj["text"].strip()
                if not text:
                    raise DataFormatError(f"{where}: empty 'text'")
                label_raw = obj.get("label")
                label = str(label_raw).strip() if label_raw not in (None, "") else None
                documents.append(Document(text=text, label=label, source=where))
    except OSError as err:
        raise DataFormatError(f"cannot read {path}: {err.strerror or err}") from err
    if not documents:
        raise DataFormatError(f"{path.name}: no data rows")
    return documents
