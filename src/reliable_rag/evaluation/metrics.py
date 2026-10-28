"""Small denominator-explicit metrics whose definitions are fixed by the protocol."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

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
