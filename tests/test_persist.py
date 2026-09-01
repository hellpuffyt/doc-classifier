from __future__ import annotations

from pathlib import Path

import pytest

from doc_classifier import model, persist
from doc_classifier.errors import ModelError
from doc_classifier.features import extract
from doc_classifier.types import Document

from .conftest import corpus


class TestRoundTrip:
    def test_save_then_load_predicts_identically(self, tmp_path: Path) -> None:
        fitted = model.fit(corpus(10))
        fitted.temperature = 1.7
        path = tmp_path / "model.json"
        persist.save(fitted, path)
        loaded = persist.load(path)

        features = extract(Document(text="INVOICE amount due remit to accounts"))
        assert fitted.predict(features).to_dict() == loaded.predict(features).to_dict()

    def test_round_trip_preserves_every_class(self, tmp_path: Path) -> None:
        fitted = model.fit(corpus(10))
        path = tmp_path / "model.json"
        persist.save(fitted, path)
        assert persist.load(path).classes == fitted.classes

    def test_saved_file_is_plain_json(self, tmp_path: Path) -> None:
        fitted = model.fit(corpus(5))
        path = tmp_path / "model.json"
        persist.save(fitted, path)
        import json

        json.loads(path.read_text(encoding="utf-8"))  # must not raise


class TestErrors:
    def test_missing_file_reports_the_path(self, tmp_path: Path) -> None:
        with pytest.raises(ModelError, match="cannot read"):
            persist.load(tmp_path / "absent.json")

    def test_not_json_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "m.json"
        path.write_text("{oops", encoding="utf-8")
        with pytest.raises(ModelError, match="not valid JSON"):
            persist.load(path)

    def test_non_object_json_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "m.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ModelError, match="not a model file"):
            persist.load(path)

    def test_wrong_format_version_is_refused(self, tmp_path: Path) -> None:
        fitted = model.fit(corpus(5))
        path = tmp_path / "m.json"
        persist.save(fitted, path)
        text = path.read_text(encoding="utf-8")
        payload = text.replace('"format_version": 1', '"format_version": 99')
        path.write_text(payload, encoding="utf-8")
        with pytest.raises(ModelError, match="format version 99"):
            persist.load(path)

    def test_missing_field_is_a_named_error(self, tmp_path: Path) -> None:
        path = tmp_path / "m.json"
        path.write_text('{"format_version": 1}', encoding="utf-8")
        with pytest.raises(ModelError, match="malformed"):
            persist.load(path)
