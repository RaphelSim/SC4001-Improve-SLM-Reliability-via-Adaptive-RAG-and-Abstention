"""Small denominator-explicit metrics whose definitions are fixed by the protocol."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from reliable_rag.contracts.enums import SufficiencyStatus
from reliable_rag.contracts.records import ExperimentRecord


@dataclass(frozen=True)
class MetricResult:
    """A rate with its numerator and denominator retained for transparent reporting."""

    numerator: int
    denominator: int

    @property
    def value(self) -> float | None:
        """Return the rate, or None when the denominator is zero."""
        return self.numerator / self.denominator if self.denominator else None


@dataclass(frozen=True)
class SufficiencyClassificationMetrics:
    """Binary classification counts and rates with NEED_MORE_EVIDENCE as positive."""

    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @property
    def accuracy(self) -> float | None:
        total = self.true_positive + self.false_positive + self.true_negative + self.false_negative
        correct = self.true_positive + self.true_negative
        return correct / total if total else None

    @property
    def precision(self) -> float | None:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else None

    @property
    def recall(self) -> float | None:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else None

    @property
    def f1(self) -> float | None:
        denominator = 2 * self.true_positive + self.false_positive + self.false_negative
        return 2 * self.true_positive / denominator if denominator else None

    @property
    def false_sufficiency_rate(self) -> float | None:
        """Among gold-insufficient cases, the fraction predicted sufficient."""
        denominator = self.true_positive + self.false_negative
        return self.false_negative / denominator if denominator else None

    @property
    def false_insufficiency_rate(self) -> float | None:
        """Among gold-sufficient cases, the fraction predicted insufficient."""
        denominator = self.false_positive + self.true_negative
        return self.false_positive / denominator if denominator else None


def sufficiency_classification_metrics(
    gold_sufficient: Sequence[bool], predicted_statuses: Sequence[SufficiencyStatus]
) -> SufficiencyClassificationMetrics:
    """Compute binary sufficiency counts with NEED_MORE_EVIDENCE as the positive class."""
    if len(gold_sufficient) != len(predicted_statuses):
        raise ValueError("gold_sufficient and predicted_statuses must have equal lengths")
    true_positive = false_positive = true_negative = false_negative = 0
    for is_gold_sufficient, predicted_status in zip(
        gold_sufficient, predicted_statuses, strict=True
    ):
        predicted_insufficient = predicted_status == SufficiencyStatus.NEED_MORE_EVIDENCE
        if not is_gold_sufficient and predicted_insufficient:
            true_positive += 1
        elif is_gold_sufficient and predicted_insufficient:
            false_positive += 1
        elif is_gold_sufficient:
            true_negative += 1
        else:
            false_negative += 1
    return SufficiencyClassificationMetrics(
        true_positive=true_positive,
        false_positive=false_positive,
        true_negative=true_negative,
        false_negative=false_negative,
    )


def supporting_evidence_recall_at_k(
    gold_support_ids: Sequence[str], retrieved_evidence_ids: Sequence[str], k: int
) -> MetricResult:
    """Measure the fraction of gold supports present among the first k ranked results."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if not gold_support_ids:
        raise ValueError("gold_support_ids must not be empty")
    if len(gold_support_ids) != len(set(gold_support_ids)):
        raise ValueError("gold_support_ids must be unique")
    top_ids = set(retrieved_evidence_ids[:k])
    return MetricResult(
        numerator=len(set(gold_support_ids).intersection(top_ids)),
        denominator=len(gold_support_ids),
    )


def answer_coverage(records: Sequence[ExperimentRecord]) -> MetricResult:
    """Return answered cases divided by all records."""
    return MetricResult(
        numerator=sum(not record.final_answer.abstained for record in records),
        denominator=len(records),
    )


def correct_abstention_rate(records: Sequence[ExperimentRecord]) -> MetricResult:
    """Return abstentions among final controlled-insufficient cases / all such cases."""
    insufficient = [record for record in records if not record.gold_sufficient_finally]
    return MetricResult(
        numerator=sum(record.final_answer.abstained for record in insufficient),
        denominator=len(insufficient),
    )


def false_abstention_rate(records: Sequence[ExperimentRecord]) -> MetricResult:
    """Return abstentions among final controlled-sufficient cases / all such cases."""
    sufficient = [record for record in records if record.gold_sufficient_finally]
    return MetricResult(
        numerator=sum(record.final_answer.abstained for record in sufficient),
        denominator=len(sufficient),
    )


def unsupported_answer_rate(records: Sequence[ExperimentRecord]) -> MetricResult:
    """Return non-abstained outputs on insufficient final evidence / insufficient cases."""
    insufficient = [record for record in records if not record.gold_sufficient_finally]
    return MetricResult(
        numerator=sum(not record.final_answer.abstained for record in insufficient),
        denominator=len(insufficient),
    )


def unsupported_answer_rate_among_answered(records: Sequence[ExperimentRecord]) -> MetricResult:
    """Return unsupported non-abstained outputs divided by all non-abstained outputs."""
    answered = [record for record in records if not record.final_answer.abstained]
    return MetricResult(
        numerator=sum(not record.gold_sufficient_finally for record in answered),
        denominator=len(answered),
    )
