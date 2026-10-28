"""Strict JSONL persistence tests."""

from pathlib import Path

import pytest

from reliable_rag.contracts.records import ExperimentRecord
from reliable_rag.io.jsonl import JSONLValidationError, read_jsonl, write_jsonl

FIXTURE = Path(__file__).parents[1] / "fixtures" / "example_experiment.jsonl"


def test_jsonl_round_trip(tmp_path: Path) -> None:
    records = read_jsonl(FIXTURE, ExperimentRecord)
    output = tmp_path / "nested" / "roundtrip.jsonl"
    write_jsonl(output, records)
    reread = read_jsonl(output, ExperimentRecord)
    assert reread == records
    assert output.read_text(encoding="utf-8").endswith("\n")


def test_malformed_jsonl_reports_path_line_and_reason(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    valid_row = FIXTURE.read_text(encoding="utf-8").strip()
    bad.write_text(f"{valid_row}\n{{not-json}}\n", encoding="utf-8")
    with pytest.raises(JSONLValidationError) as error:
        read_jsonl(bad, ExperimentRecord)
    assert f"{bad}:2:" in str(error.value)
    assert "invalid JSON" in str(error.value)


def test_jsonl_rejects_blank_lines(tmp_path: Path) -> None:
    blank = tmp_path / "blank.jsonl"
    blank.write_text("\n", encoding="utf-8")
    with pytest.raises(JSONLValidationError, match="blank lines are not allowed"):
        read_jsonl(blank, ExperimentRecord)
