"""Tests for metrics with fixed operational definitions and explicit denominators."""

import json
from pathlib import Path

import pytest

from reliable_rag.contracts.enums import AbstentionReason, SufficiencyStatus
from reliable_rag.contracts.records import ExperimentRecord
from reliable_rag.evaluation.metrics import (
    answer_coverage,
    answer_exact_match,
    answer_token_f1,
    correct_abstention_rate,
    exact_match,
    false_abstention_rate,
    followup_support_recall,
    normalize_answer,
    recovery_rate,
    sufficiency_classification_metrics,
    supporting_evidence_recall_at_k,
    token_f1,
    unsupported_answer_rate,
    unsupported_answer_rate_among_answered,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "example_experiment.jsonl"


def test_supporting_evidence_recall_at_k_preserves_rank_cutoff() -> None:
    at_one = supporting_evidence_recall_at_k(
        ["support-a", "support-b"], ["distractor", "support-a", "support-b"], k=1
    )
    at_three = supporting_evidence_recall_at_k(
        ["support-a", "support-b"], ["distractor", "support-a", "support-b"], k=3
    )
    assert (at_one.numerator, at_one.denominator, at_one.value) == (0, 2, 0.0)
    assert (at_three.numerator, at_three.denominator, at_three.value) == (2, 2, 1.0)
    with pytest.raises(ValueError, match="k must be at least 1"):
        supporting_evidence_recall_at_k(["support-a"], [], k=0)


def test_coverage_and_unsupported_rates_use_documented_denominators() -> None:
    record_data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    supported = ExperimentRecord.model_validate(record_data)
    unsupported_data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    unsupported_data["final_evidence_ids"] = ["lost-gravity::0"]
    unsupported_data["final_answer"]["used_evidence_ids"] = ["lost-gravity::0"]
    unsupported_data["gold_sufficient_finally"] = False
    unsupported_data["model_predicted_sufficient_finally"] = None
    unsupported = ExperimentRecord.model_validate(unsupported_data)

    coverage = answer_coverage([supported, unsupported])
    unsafe_rate = unsupported_answer_rate([supported, unsupported])
    answered_rate = unsupported_answer_rate_among_answered([supported, unsupported])
    assert (coverage.numerator, coverage.denominator, coverage.value) == (2, 2, 1.0)
    assert (unsafe_rate.numerator, unsafe_rate.denominator, unsafe_rate.value) == (1, 1, 1.0)
    assert (answered_rate.numerator, answered_rate.denominator, answered_rate.value) == (
        1,
        2,
        0.5,
    )
    assert answer_coverage([]).value is None


def test_sufficiency_metrics_use_need_more_as_positive_class() -> None:
    metrics = sufficiency_classification_metrics(
        [False, False, True, True],
        [
            SufficiencyStatus.SUFFICIENT,
            SufficiencyStatus.NEED_MORE_EVIDENCE,
            SufficiencyStatus.NEED_MORE_EVIDENCE,
            SufficiencyStatus.SUFFICIENT,
        ],
    )
    assert (metrics.true_positive, metrics.false_positive) == (1, 1)
    assert (metrics.true_negative, metrics.false_negative) == (1, 1)
    assert metrics.accuracy == 0.5
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1 == 0.5
    assert metrics.false_sufficiency_rate == 0.5
    assert metrics.false_insufficiency_rate == 0.5
    with pytest.raises(ValueError, match="equal lengths"):
        sufficiency_classification_metrics([True], [])


def test_correct_and_false_abstention_rates_have_distinct_denominators() -> None:
    supported_data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    supported = ExperimentRecord.model_validate(supported_data)

    unsupported_data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    unsupported_data["final_evidence_ids"] = ["lost-gravity::0"]
    unsupported_data["final_answer"]["used_evidence_ids"] = ["lost-gravity::0"]
    unsupported_data["gold_sufficient_finally"] = False
    unsupported_data["model_predicted_sufficient_finally"] = None
    unsupported_answer = ExperimentRecord.model_validate(unsupported_data)

    correct_abstention_data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    correct_abstention_data["final_evidence_ids"] = ["lost-gravity::0"]
    correct_abstention_data["final_answer"] = {
        "question_id": "lost-gravity-country",
        "answer": None,
        "abstained": True,
        "abstention_reason": AbstentionReason.INSUFFICIENT_EVIDENCE.value,
        "used_evidence_ids": [],
    }
    correct_abstention_data["gold_sufficient_finally"] = False
    correct_abstention_data["model_predicted_sufficient_finally"] = None
    correct_abstention = ExperimentRecord.model_validate(correct_abstention_data)

    false_abstention_data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    false_abstention_data["final_answer"] = {
        "question_id": "lost-gravity-country",
        "answer": None,
        "abstained": True,
        "abstention_reason": AbstentionReason.INSUFFICIENT_EVIDENCE.value,
        "used_evidence_ids": [],
    }
    false_abstention_data["model_predicted_sufficient_finally"] = None
    false_abstention = ExperimentRecord.model_validate(false_abstention_data)

    cases = [supported, unsupported_answer, correct_abstention, false_abstention]
    assert (
        correct_abstention_rate(cases).numerator,
        correct_abstention_rate(cases).denominator,
    ) == (
        1,
        2,
    )
    assert (false_abstention_rate(cases).numerator, false_abstention_rate(cases).denominator) == (
        1,
        2,
    )


def test_normalize_answer_removes_case_punctuation_and_articles() -> None:
    assert normalize_answer("    The Oberoi Group. ") == "oberoi group"


def test_exact_match_ignores_formatting_but_not_content() -> None:
    assert exact_match("germany", "Germany")
    assert exact_match("the Oberoi Group.", "Oberoi Group")
    assert not exact_match("New Delhi", "Delhi")


def test_token_f1_gives_partial_credit_for_overlap() -> None:
    assert token_f1("New Delhi", "Delhi") == pytest.approx(2 / 3)
    assert token_f1("Germany", "Germany") == 1.0
    assert token_f1("France", "Germany") == 0.0


def test_token_f1_requires_exact_yes_no() -> None:
    assert token_f1("yes", "yes") == 1.0
    assert token_f1("no", "yes") == 0.0
    assert token_f1("Yes, both are magazines", "yes") == 0.0


def test_answer_metrics_use_answered_or_all_record_denominators() -> None:
    answered = ExperimentRecord.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    abstained = answered.model_copy(
        update={
            "final_answer": answered.final_answer.model_copy(
                update={
                    "answer": None,
                    "abstained": True,
                    "abstention_reason": AbstentionReason.INSUFFICIENT_EVIDENCE,
                    "used_evidence_ids": [],
                }
            )
        }
    )

    records = [answered, abstained]

    exact = answer_exact_match(records)
    f1 = answer_token_f1(records)
    exact_all = answer_exact_match(records, answered_only=False)
    f1_all = answer_token_f1(records, answered_only=False)

    assert (exact.numerator, exact.denominator) == (1, 1)
    assert (f1.total, f1.count, f1.value) == (1.0, 1, 1.0)
    assert (exact_all.numerator, exact_all.denominator, exact_all.value) == (1, 2, 0.5)
    assert f1_all.value == 0.5
    assert answer_exact_match([]).value is None


def test_recovery_metrics_count_round_one_repairs() -> None:
    recovered = ExperimentRecord.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    already_complete = recovered.model_copy(
        update={"gold_sufficient_initially": True, "rounds": recovered.rounds[:1]}
    )
    records = [recovered, already_complete]

    recovery = recovery_rate(records)
    followup = followup_support_recall(records)

    assert (recovery.numerator, recovery.denominator, recovery.value) == (1, 1, 1.0)
    assert (followup.numerator, followup.denominator) == (1, 1)
    assert recovery_rate([already_complete]).value is None
