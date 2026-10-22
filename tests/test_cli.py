from __future__ import annotations

import json
from pathlib import Path

import pytest

from doc_classifier import __version__, cli

from .conftest import corpus


def write_jsonl(path: Path, docs: list) -> str:
    lines = [
        json.dumps({"text": doc.text, "label": doc.label}) for doc in docs
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


class TestTrain:
    def test_trains_and_writes_a_model(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        data = write_jsonl(tmp_path / "d.jsonl", corpus(15))
        model_path = tmp_path / "model.json"
        code = cli.main(["train", data, "--out", str(model_path)])
        assert code == cli.EXIT_OK
        assert model_path.exists()
        out = capsys.readouterr().out
        assert "calibration temperature" in out

    def test_bad_input_exits_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = cli.main(
            ["train", str(tmp_path / "absent.jsonl"), "--out", str(tmp_path / "m.json")]
        )
        assert code == cli.EXIT_USAGE
        assert capsys.readouterr().err.startswith("error:")


class TestClassify:
    def _model(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> str:
        data = write_jsonl(tmp_path / "d.jsonl", corpus(15))
        model_path = tmp_path / "model.json"
        cli.main(["train", data, "--out", str(model_path)])
        capsys.readouterr()  # discard train's own stdout before the test asserts
        return str(model_path)

    def test_classifies_argument_text(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        model_path = self._model(tmp_path, capsys)
        code = cli.main(
            ["classify", "--model", model_path, "INVOICE amount due remit to accounts"]
        )
        assert code == cli.EXIT_OK
        assert "invoice" in capsys.readouterr().out

    def test_classifies_stdin_when_no_argument(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model_path = self._model(tmp_path, capsys)
        monkeypatch.setattr(
            "sys.stdin", __import__("io").StringIO("RECEIPT thank you total paid cashier")
        )
        code = cli.main(["classify", "--model", model_path])
        assert code == cli.EXIT_OK
        assert "receipt" in capsys.readouterr().out

    def test_empty_text_is_refused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        model_path = self._model(tmp_path, capsys)
        code = cli.main(["classify", "--model", model_path, "   "])
        assert code == cli.EXIT_USAGE

    def test_json_format(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        model_path = self._model(tmp_path, capsys)
        cli.main(["classify", "--model", model_path, "INVOICE amount due", "--format", "json"])
        payload = json.loads(capsys.readouterr().out)
        assert "probabilities" in payload

    def test_missing_model_file_exits_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = cli.main(["classify", "--model", str(tmp_path / "absent.json"), "text"])
        assert code == cli.EXIT_USAGE


class TestEval:
    def test_reports_accuracy_and_exits_zero_by_default(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        data = write_jsonl(tmp_path / "d.jsonl", corpus(15))
        model_path = tmp_path / "model.json"
        cli.main(["train", data, "--out", str(model_path)])
        code = cli.main(["eval", "--model", str(model_path), data])
        assert code == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "accuracy" in out.lower() or "correct" in out

    def test_min_accuracy_threshold_trips(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        data = write_jsonl(tmp_path / "d.jsonl", corpus(15))
        model_path = tmp_path / "model.json"
        cli.main(["train", data, "--out", str(model_path)])
        code = cli.main(
            ["eval", "--model", str(model_path), data, "--min-accuracy", "1.1"]
        )
        assert code == cli.EXIT_BELOW_THRESHOLD


class TestMisc:
    def test_version_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as excinfo:
            cli.main(["--version"])
        assert excinfo.value.code == 0
        assert __version__ in capsys.readouterr().out

    def test_no_command_is_a_usage_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit):
            cli.main([])


class TestModuleEntryPoint:
    def test_python_dash_m_invocation(self, tmp_path: Path) -> None:
        import subprocess
        import sys

        data = write_jsonl(tmp_path / "d.jsonl", corpus(15))
        model_path = tmp_path / "model.json"
        done = subprocess.run(
            [sys.executable, "-m", "doc_classifier", "train", data, "--out", str(model_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert done.returncode == 0
        assert model_path.exists()


def test_train_reports_holdout_size_and_warns_when_small(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    data = write_jsonl(tmp_path / "d.jsonl", corpus())
    assert cli.main(["train", data, "--out", str(tmp_path / "m.json")]) == 0
    captured = capsys.readouterr()
    assert "held-out example(s)" in captured.out
    assert "not reliable" in captured.err
