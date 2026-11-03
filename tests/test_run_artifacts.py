"""Run persistence protects inputs, preserves progress and refuses silent overwrites."""

import json
from pathlib import Path

import pytest

from reliable_rag.config import load_settings
from reliable_rag.run_artifacts import RunWriter

ROOT = Path(__file__).parents[1]


def test_streamed_failure_and_progress_survive_close(tmp_path):
    settings = load_settings(ROOT / "configs/base.yaml", environ={})
    settings.output_root = tmp_path
    writer = RunWriter(settings, "run1", inputs=[])
    error = {"question_id": "q1", "category": "model_contract_failure", "record_written": False}
    writer.append("errors", error)
    writer.metadata["summary"]["attempted"] = 1
    writer.metadata["summary"]["unrecorded_assessment_failures"] = 1
    writer.save()
    assert json.loads((tmp_path / "run1/errors.jsonl").read_text()) == error
    writer.close()
    metadata = json.loads((tmp_path / "run1/run_metadata.json").read_text())
    assert metadata["summary"]["attempted"] == 1
    assert (tmp_path / "run1/results.jsonl").read_text() == ""
    with pytest.raises(FileExistsError):
        RunWriter(settings, "run1", inputs=[])
    replacement = RunWriter(settings, "run1", inputs=[], force=True)
    replacement.close()


def test_artifacts_refuse_to_overwrite_inputs_or_escape_root(tmp_path):
    settings = load_settings(ROOT / "configs/base.yaml", environ={})
    settings.output_root = tmp_path
    output = tmp_path / "run1/input.jsonl"
    output.parent.mkdir()
    output.write_text("original")
    with pytest.raises(ValueError, match="overwrite"):
        RunWriter(settings, "run1", inputs=[output], force=True)
    assert output.read_text() == "original"
    with pytest.raises(ValueError, match="run_id"):
        RunWriter(settings, "../unsafe", inputs=[])
