"""Small denominator-explicit metrics whose definitions are fixed by the protocol."""

from __future__ import annotations

import re
import string
from collections import Counter
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
class MeanScore:
    """An average of per-case scores with its total and case count retained"""

    total: float
    count: int

    @property
    def value(self) -> float | None:
        """Return the mean score, or None when there are no cases."""
        return self.total / self.count if self.count else None


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


# Answer correctness follows HotpotQA evaluation script
# Answers are normalised before comparison, so
# E.g., "The Oberoi Group." and "oberoi group" matches
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_YES_NO = {"yes", "no", "noanswer"}


def normalize_answer(text: str) -> str:
    """Lowercase, remove punctuation, articles (a, an, the) and extra whitespace"""
    text = text.lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = _ARTICLES.sub(" ", text)

    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> bool:
    """Return whether normalised prediction is equal to normalised gold answer"""
    return normalize_answer(prediction) == normalize_answer(gold)


def token_f1(prediction: str, gold: str) -> float:
    """Return word-overlap F1 between normalised prediction and gold answer"""
    normalized_prediction = normalize_answer(prediction)
    normalized_gold = normalize_answer(gold)

    # yes/no only gets credit for exact match
    # "no" would get partial credit against "yes, no" style answers
    if (
        normalized_prediction in _YES_NO or normalized_gold in _YES_NO
    ) and normalized_prediction != normalized_gold:
        return 0.0

    prediction_tokens = normalized_prediction.split()
    gold_tokens = normalized_gold.split()
    common = sum((Counter(prediction_tokens) & Counter(gold_tokens)).values())

    if common == 0:
        return 0.0

    precision = common / len(prediction_tokens)
    recall = common / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def answer_exact_match(
    records: Sequence[ExperimentRecord], *, answered_only: bool = True
) -> MetricResult:
    """
    Return exact matches over answered records, or over all records when not answered_only

    With `answered_only = False` abstentions count as wrong, so score also shows coverage
    """

    answered = [record for record in records if not record.final_answer.abstained]

    return MetricResult(
        numerator=sum(
            exact_match(record.final_answer.answer or "", record.question.gold_answer)
            for record in answered
        ),
        denominator=len(answered) if answered_only else len(records),
    )


def answer_token_f1(
    records: Sequence[ExperimentRecord], *, answered_only: bool = True
) -> MeanScore:
    """
    Return mean token F1 over answered records, or over all records when not answered_only

    With `answered_only = False` abstentions score 0, giving-coverage-adjusted F1
    """

    answered = [record for record in records if not record.final_answer.abstained]

    return MeanScore(
        total=sum(
            token_f1(record.final_answer.answer or "", record.question.gold_answer)
            for record in answered
        ),
        count=len(answered) if answered_only else len(records),
    )


# Adaptive retrieval metrics
def recovery_rate(records: Sequence[ExperimentRecord]) -> MetricResult:
    """
    Return initially insufficient records that end gold-sufficient / initially insufficient

    Counts every reason recovery can fail:
    - Model judged evidence sufficient and never searched
    - Model's follow-up query missed the paragraph
    - Follow-up found nothing new

    Single round systems always score 0, since initial and final evidence match
    """

    initially_insufficient = [record for record in records if not record.gold_sufficient_initially]

    return MetricResult(
        numerator=sum(record.gold_sufficient_finally for record in initially_insufficient),
        denominator=len(initially_insufficient),
    )


def followup_support_recall(records: Sequence[ExperimentRecord]) -> MetricResult:
    """
    Return missing gold supports found by follow-up / supports missing after round 0

    Only records with stored round-1 retrieval count, so this measures how well
    the model's search_query finds what is missing, separately from whether it is asked at all
    """

    found = missing_total = 0

    for record in records:
        if record.gold_sufficient_initially or len(record.rounds) < 2:
            continue

        missing = set(record.question.gold_support_ids).difference(
            record.initial_evidence.evidence_ids
        )

        followup_ids = {chunk.evidence_id for chunk in record.rounds[1].retrieval.evidence}
        found += len(missing.intersection(followup_ids))
        missing_total += len(missing)

    return MetricResult(numerator=found, denominator=missing_total)
