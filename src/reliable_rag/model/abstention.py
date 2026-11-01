"""RAG + Abstention baseline policy; this module does not invoke retrieval."""

from __future__ import annotations

import logging

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
        """Run sufficiency then answer-or-abstain without any retrieval side effects."""
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
            return self._abstain(
                question.question_id,
                AbstentionReason.INVALID_MODEL_OUTPUT,
                task=exc.task,
            )

        if decision.status == SufficiencyStatus.NEED_MORE_EVIDENCE:
            LOGGER.info(
                "Abstained question_id=%s status=%s reason=%s",
                question.question_id,
                decision.status.value,
                AbstentionReason.INSUFFICIENT_EVIDENCE.value,
            )
            return AnswerResult(
                question_id=question.question_id,
                answer=None,
                abstained=True,
                abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
                used_evidence_ids=[],
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
            if not answer.abstained and not answer.used_evidence_ids:
                raise ModelOutputError(
                    task="final_answer",
                    question_id=question.question_id,
                    attempts=1,
                    reason="substantive answer did not cite used evidence",
                )
            return answer
        except ModelOutputError as exc:
            return self._abstain(
                question.question_id,
                AbstentionReason.INVALID_MODEL_OUTPUT,
                task=exc.task,
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
