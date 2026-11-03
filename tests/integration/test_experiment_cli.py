"""The normal runner persists contract failures and runtime interruptions separately."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import evaluate_results, run_experiment
from tests.model.conftest import FakeSLMClient, load_case

from reliable_rag.config import load_settings
from reliable_rag.contracts.records import RetrievalResult
from reliable_rag.io.jsonl import write_jsonl
from reliable_rag.model.answer import SLMAnswerGenerator
from reliable_rag.model.sufficiency import SLMSufficiencyAssessor

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize("runtime_failure", [False, True])
def test_runner_persists_distinct_failures_and_refuses_overwrite(
    monkeypatch, tmp_path, runtime_failure
):
    case = json.loads((ROOT / "tests/fixtures/sufficiency_cases.json").read_text())["cases"][0]
    question, evidence = load_case(case)
    settings = load_settings(ROOT / "configs/base.yaml", environ={})
    settings.output_root = tmp_path / "outputs"
    config = tmp_path / "config.yaml"
    config.write_text(json.dumps(settings.model_dump(mode="json")))
    questions, retrieval = tmp_path / "questions.jsonl", tmp_path / "retrieval.jsonl"
    write_jsonl(questions, [question])
    write_jsonl(
        retrieval,
        [
            RetrievalResult(
                question_id=question.question_id,
                query=question.question,
                stage="initial",
                top_k=3,
                evidence=evidence.evidence[:3],
            )
        ],
    )

    def components(*args):
        if runtime_failure:
            raise RuntimeError("load failure")
        client = FakeSLMClient("not json", "still not json")
        return SimpleNamespace(
            client=SimpleNamespace(model_path=tmp_path),
            assessor=SLMSufficiencyAssessor(client, settings),
            answer_generator=SLMAnswerGenerator(client, settings),
        )

    monkeypatch.setattr(run_experiment, "create_model_components", components)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment.py",
            "--config",
            str(config),
            "--run-id",
            "test",
            "--questions",
            str(questions),
            "--retrieval",
            str(retrieval),
        ],
    )
    if runtime_failure:
        with pytest.raises(RuntimeError, match="load failure"):
            run_experiment.main()
    else:
        assert run_experiment.main() == 1
    directory = settings.output_root / "test"
    metadata = json.loads((directory / "run_metadata.json").read_text())
    errors = [json.loads(line) for line in (directory / "errors.jsonl").read_text().splitlines()]
    assert (directory / "results.jsonl").read_text() == ""
    assert errors[0]["category"] == (
        "runtime_error" if runtime_failure else "model_contract_failure"
    )
    assert metadata["status"] == ("failed" if runtime_failure else "completed_with_model_errors")
    assert metadata["summary"]["attempted"] == (0 if runtime_failure else 1)
    if not runtime_failure:
        monkeypatch.setattr(
            sys, "argv", ["evaluate_results.py", "--config", str(config), "--run-id", "test"]
        )
        assert evaluate_results.main() == 0
        report = json.loads((directory / "evaluation.json").read_text())
        assert report["sufficiency"]["accuracy"]["denominator"] == 0
        assert report["answers"]["coverage_attempted"]["denominator"] == 1
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment.py",
            "--config",
            str(config),
            "--run-id",
            "test",
            "--questions",
            str(questions),
            "--retrieval",
            str(retrieval),
        ],
    )
    with pytest.raises(FileExistsError):
        run_experiment.main()
