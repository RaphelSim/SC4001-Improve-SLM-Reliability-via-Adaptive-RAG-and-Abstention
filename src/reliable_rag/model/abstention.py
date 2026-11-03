"""RAG + Abstention baseline policy; this module does not invoke retrieval."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from reliable_rag.contracts.enums import AbstentionReason, SufficiencyStatus
from reliable_rag.contracts.interfaces import AnswerGenerator, SufficiencyAssessor
from reliable_rag.contracts.records import (
    AnswerResult,
    EvidenceSet,
    QuestionRecord,
    SufficiencyDecision,
)
from reliable_rag.model.errors import ModelOutputError

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RAGAbstentionOutcome:
    """One sufficiency decision and answer, including a safe structured model failure."""

    sufficiency_decision: SufficiencyDecision | None
    answer: AnswerResult
    error: ModelOutputError | None = None


class RAGAbstentionResponder:
    """Answer when evidence is sufficient and abstain when it is not or output is invalid."""

    def __init__(
        self,
        assessor: SufficiencyAssessor,
        answer_generator: AnswerGenerator,
    ) -> None:
        self._assessor = assessor
        self._answer_generator = answer_generator

    def answer(self, question: QuestionRecord, evidence: EvidenceSet) -> AnswerResult:
        """Return only the answer for callers that do not need the sufficiency decision."""
        return self.run(question, evidence).answer

    def run(self, question: QuestionRecord, evidence: EvidenceSet) -> RAGAbstentionOutcome:
        """Run sufficiency then answer-or-abstain once, without retrieval side effects."""
        if question.question_id != evidence.question_id:
            raise ValueError("question and EvidenceSet question_id values must match")
        try:
            decision = self._assessor.assess(question, evidence)
            if not isinstance(decision, SufficiencyDecision):
                raise ModelOutputError(
                    task="sufficiency_assessment",
                    question_id=question.question_id,
                    attempts=1,
                    reason="assessor did not return a SufficiencyDecision",
                )
            if decision.question_id != question.question_id:
                raise ModelOutputError(
                    task="sufficiency_assessment",
                    question_id=question.question_id,
                    attempts=1,
                    reason="decision question_id does not match input question",
                )
            if len(decision.supporting_evidence_ids) != len(set(decision.supporting_evidence_ids)):
                raise ModelOutputError(
                    task="sufficiency_assessment",
                    question_id=question.question_id,
                    attempts=1,
                    reason="decision contains duplicate supporting evidence IDs",
                )
            unknown_support_ids = set(decision.supporting_evidence_ids) - set(evidence.evidence_ids)
            if unknown_support_ids:
                raise ModelOutputError(
                    task="sufficiency_assessment",
                    question_id=question.question_id,
                    attempts=1,
                    reason="decision cited evidence IDs absent from supplied evidence",
                )
            if (
                decision.status == SufficiencyStatus.SUFFICIENT
                and not decision.supporting_evidence_ids
            ):
                raise ModelOutputError(
                    task="sufficiency_assessment",
                    question_id=question.question_id,
                    attempts=1,
                    reason="sufficient decision did not cite supporting evidence",
                )
        except ModelOutputError as exc:
            return RAGAbstentionOutcome(
                sufficiency_decision=None,
                answer=self._abstain(
                    question.question_id,
                    AbstentionReason.INVALID_MODEL_OUTPUT,
                    task=exc.task,
                ),
                error=exc,
            )

        if decision.status == SufficiencyStatus.NEED_MORE_EVIDENCE:
            LOGGER.info(
                "Abstained question_id=%s status=%s reason=%s",
                question.question_id,
                decision.status.value,
                AbstentionReason.INSUFFICIENT_EVIDENCE.value,
            )
            return RAGAbstentionOutcome(
                sufficiency_decision=decision,
                answer=AnswerResult(
                    question_id=question.question_id,
                    answer=None,
                    abstained=True,
                    abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
                    used_evidence_ids=[],
                ),
            )

        try:
            answer = self._answer_generator.generate(question, evidence)
            if not isinstance(answer, AnswerResult):
                raise ModelOutputError(
                    task="final_answer",
                    question_id=question.question_id,
                    attempts=1,
                    reason="answer generator did not return an AnswerResult",
                )
            if answer.question_id != question.question_id:
                raise ModelOutputError(
                    task="final_answer",
                    question_id=question.question_id,
                    attempts=1,
                    reason="answer question_id does not match input question",
                )
            if not set(answer.used_evidence_ids).issubset(evidence.evidence_ids):
                raise ModelOutputError(
                    task="final_answer",
                    question_id=question.question_id,
                    attempts=1,
                    reason="answer cited evidence IDs absent from supplied evidence",
                )
            if len(answer.used_evidence_ids) != len(set(answer.used_evidence_ids)):
                raise ModelOutputError(
                    task="final_answer",
                    question_id=question.question_id,
                    attempts=1,
                    reason="answer contains duplicate used evidence IDs",
                )
            if not answer.abstained and not answer.used_evidence_ids:
                raise ModelOutputError(
                    task="final_answer",
                    question_id=question.question_id,
                    attempts=1,
                    reason="substantive answer did not cite used evidence",
                )
            return RAGAbstentionOutcome(sufficiency_decision=decision, answer=answer)
        except ModelOutputError as exc:
            return RAGAbstentionOutcome(
                sufficiency_decision=decision,
                answer=self._abstain(
                    question.question_id,
                    AbstentionReason.INVALID_MODEL_OUTPUT,
                    task=exc.task,
                ),
                error=exc,
            )

    @staticmethod
    def _abstain(
        question_id: str,
        reason: AbstentionReason,
        *,
        task: str,
    ) -> AnswerResult:
        """Create a typed abstention and log only identifiers and structured labels."""
        LOGGER.warning(
            "Abstained question_id=%s reason=%s task=%s",
            question_id,
            reason.value,
            task,
        )
        return AnswerResult(
            question_id=question_id,
            answer=None,
            abstained=True,
            abstention_reason=reason,
            used_evidence_ids=[],
        )
