"""Adaptive runner writes evaluable runs and can reuse baseline round-0 decisions."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from scripts import evaluate_results, run_adaptive
from tests.adaptive.test_controller import (
    LOST_GRAVITY,
    MACK_RIDES,
    QUESTION,
    FakeAnswerGenerator,
    FakeRetriever,
    GoldAwareAssessor,
)

from reliable_rag.config import load_settings
from reliable_rag.contracts.records import RetrievalResult
from reliable_rag.io.jsonl import write_jsonl
from reliable_rag.model.abstention import RAGAbstentionResponder
from reliable_rag.pipeline import run_initial_rag_abstention_case

ROOT = Path(__file__).parents[2]


def prepare_run(monkeypatch, tmp_path, assessor):
    """Write tiny inputs and swap in fakes for the retriever and Qwen components."""
    settings = load_settings(ROOT / "configs/base.yaml", environ={})
    settings.output_root = tmp_path / "outputs"
    config = tmp_path / "config.yaml"
    config.write_text(json.dumps(settings.model_dump(mode="json")))
    paths = {name: tmp_path / f"{name}.jsonl" for name in ("questions", "evidence", "retrieval")}
    index = tmp_path / "index.npz"
    index.write_bytes(b"unused: the retriever is faked")
    write_jsonl(paths["questions"], [QUESTION])
    write_jsonl(paths["evidence"], [LOST_GRAVITY, MACK_RIDES])
    initial = RetrievalResult(
        question_id=QUESTION.question_id,
        query=QUESTION.question,
        stage="initial",
        top_k=settings.retrieval.initial_top_k,
        evidence=[LOST_GRAVITY],
    )
    write_jsonl(paths["retrieval"], [initial])
    monkeypatch.setattr(run_adaptive, "create_retriever", lambda *a, **k: FakeRetriever(MACK_RIDES))
    monkeypatch.setattr(
        run_adaptive,
        "create_model_components",
        lambda *a, **k: SimpleNamespace(
            client=SimpleNamespace(model_path=tmp_path),
            assessor=assessor,
            answer_generator=FakeAnswerGenerator(),
        ),
    )
    argv = ["run_adaptive.py", "--config", str(config), "--run-id", "adaptive"]
    argv += ["--variant", "adaptive_retrieval_abstention", "--index", str(index)]
    for name, path in paths.items():
        argv += [f"--{name}", str(path)]
    return settings, config, initial, argv


def test_adaptive_run_records_recovery_and_evaluates(monkeypatch, tmp_path) -> None:
    settings, config, _, argv = prepare_run(monkeypatch, tmp_path, GoldAwareAssessor())
    monkeypatch.setattr(sys, "argv", argv)

    assert run_adaptive.main() == 0

    directory = settings.output_root / "adaptive"
    record = json.loads((directory / "results.jsonl").read_text().splitlines()[0])
    metadata = json.loads((directory / "run_metadata.json").read_text())
    assert record["system_variant"] == "adaptive_retrieval_abstention"
    assert len(record["rounds"]) == 2 and record["final_answer"]["answer"] == "Germany"
    assert metadata["summary"]["two_round_records"] == 1
    assert metadata["scope"] == "two_round_adaptive_retrieval"

    monkeypatch.setattr(
        sys, "argv", ["evaluate_results.py", "--config", str(config), "--run-id", "adaptive"]
    )
    assert evaluate_results.main() == 0
    report = json.loads((directory / "evaluation.json").read_text())
    assert report["adaptive"]["recovery_rate"]["value"] == 1.0
    assert report["answer_correctness"]["exact_match_attempted"]["value"] == 1.0


def test_baseline_round_zero_is_reused(monkeypatch, tmp_path) -> None:
    # Baseline already asked for more evidence, so only round 1 calls the assessor.
    assessor = GoldAwareAssessor()
    settings, _, initial, argv = prepare_run(monkeypatch, tmp_path, assessor)
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "run_metadata.json").write_text(
        json.dumps({"config": settings.model_dump(mode="json")})
    )
    case = run_initial_rag_abstention_case(
        QUESTION,
        initial,
        RAGAbstentionResponder(GoldAwareAssessor(), FakeAnswerGenerator()),
        run_id="baseline",
    )
    assert case.experiment is not None
    write_jsonl(baseline / "results.jsonl", [case.experiment])
    monkeypatch.setattr(sys, "argv", argv + ["--baseline-results", str(baseline / "results.jsonl")])

    assert run_adaptive.main() == 0

    metadata = json.loads((settings.output_root / "adaptive" / "run_metadata.json").read_text())
    assert metadata["summary"]["round0_reused"] == 1
    assert assessor.calls == 1
