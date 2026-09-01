from __future__ import annotations

import json
from pathlib import Path

import pytest

from doc_classifier import io
from doc_classifier.errors import DataFormatError


def write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


class TestCsv:
    def test_loads_text_and_label(self, tmp_path: Path) -> None:
        docs = io.load(write(tmp_path / "d.csv", "text,label\nhello world,greeting\n"))
        assert docs[0].text == "hello world"
        assert docs[0].label == "greeting"

    def test_label_column_is_optional(self, tmp_path: Path) -> None:
        docs = io.load(write(tmp_path / "d.csv", "text\nhello world\n"))
        assert docs[0].label is None

    def test_blank_label_is_none_not_empty_string(self, tmp_path: Path) -> None:
        docs = io.load(write(tmp_path / "d.csv", "text,label\nhi,\n"))
        assert docs[0].label is None

    def test_missing_text_column_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(DataFormatError, match="'text'"):
            io.load(write(tmp_path / "d.csv", "label\nfoo\n"))

    def test_empty_text_names_the_row(self, tmp_path: Path) -> None:
        with pytest.raises(DataFormatError, match=r"d\.csv:2"):
            io.load(write(tmp_path / "d.csv", "text,label\n,x\n"))

    def test_empty_file_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(DataFormatError, match="empty"):
            io.load(write(tmp_path / "d.csv", ""))

    def test_header_only_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(DataFormatError, match="no data rows"):
            io.load(write(tmp_path / "d.csv", "text\n"))

    def test_source_records_file_and_row(self, tmp_path: Path) -> None:
        docs = io.load(write(tmp_path / "d.csv", "text\nhello\n"))
        assert docs[0].source == "d.csv:2"


class TestJsonl:
    def test_loads_scalars(self, tmp_path: Path) -> None:
        line = json.dumps({"text": "hi", "label": "x"})
        docs = io.load(write(tmp_path / "d.jsonl", line + "\n"))
        assert docs[0].text == "hi"
        assert docs[0].label == "x"

    def test_missing_label_is_none(self, tmp_path: Path) -> None:
        line = json.dumps({"text": "hi"})
        docs = io.load(write(tmp_path / "d.jsonl", line + "\n"))
        assert docs[0].label is None

    def test_non_string_text_is_refused(self, tmp_path: Path) -> None:
        line = json.dumps({"text": 123})
        with pytest.raises(DataFormatError, match="string 'text'"):
            io.load(write(tmp_path / "d.jsonl", line + "\n"))

    def test_invalid_json_names_the_line(self, tmp_path: Path) -> None:
        with pytest.raises(DataFormatError, match=r"d\.jsonl:2"):
            io.load(write(tmp_path / "d.jsonl", '{"text": "a"}\n{oops\n'))

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        line = json.dumps({"text": "hi"})
        docs = io.load(write(tmp_path / "d.jsonl", f"{line}\n\n{line}\n"))
        assert len(docs) == 2


class TestDispatch:
    def test_unknown_extension_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(DataFormatError, match="unsupported input"):
            io.load(write(tmp_path / "d.txt", "x"))

    def test_missing_file_reports_the_path(self, tmp_path: Path) -> None:
        with pytest.raises(DataFormatError, match="cannot read"):
            io.load(tmp_path / "absent.csv")
