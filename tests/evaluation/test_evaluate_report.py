"""Evaluation report includes answer correctness and adaptive recovery"""

from pathlib import Path

from scripts.evaluate_results import evaluate

from reliable_rag.contracts.records import ExperimentRecord

FIXTURE = Path(__file__).parents[1] / "fixtures" / "example_experiment.jsonl"


def test_report_adds_answer_correctness_and_adaptive_sections() -> None:
    # Fixture is a recovered two round case that answers "Germany" correctly
    record = ExperimentRecord.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    question = record.question
    initial = record.rounds[0].retrieval

    report = evaluate([question], [initial], [record], [], attempted=1, ks=[1])

    correctness = report["answer_correctness"]

    assert correctness["exact_match_answered"] == {"numerator": 1, "denominator": 1, "value": 1.0}
    assert correctness["token_f1_attempted"]["value"] == 1.0
    assert report["adaptive"]["recovery_rate"]["value"] == 1.0
    assert report["adaptive"]["followup_support_recall"]["value"] == 1.0
    assert report["adaptive"]["retrieval_rounds"] == {"2": 1}
    assert "not_implemented" not in report
