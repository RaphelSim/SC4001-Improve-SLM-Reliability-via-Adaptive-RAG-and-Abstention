"""Bounded two-round adaptive retrieval controller with optional abstention"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from reliable_rag.contracts.enums import (
    AbstentionReason,
    EvidenceCondition,
    RetrievalStage,
    SufficiencyStatus,
    SystemVariant,
)
from reliable_rag.contracts.ids import make_experiment_id
from reliable_rag.contracts.interfaces import AnswerGenerator, Retriever, SufficiencyAssessor
from reliable_rag.contracts.records import (
    AdaptiveRoundRecord,
    AnswerResult,
    EvidenceChunk,
    EvidenceSet,
    ExperimentRecord,
    QuestionRecord,
    RetrievalRequest,
    RetrievalResult,
)
from reliable_rag.model.abstention import RAGAbstentionOutcome, RAGAbstentionResponder
from reliable_rag.model.errors import ModelOutputError

LOGGER = logging.getLogger(__name__)
CONTROLLER_VERSION = "adaptive-v1"


@dataclass(frozen=True)
class AdaptiveCaseResult:
    """Per-question controller output: the record when assessable, plus any model errors."""

    experiment: ExperimentRecord | None
    final_answer: AnswerResult
    errors: list[ModelOutputError] = field(default_factory=list)


def merge_evidence(
    current: list[EvidenceChunk],
    retrieved: list[EvidenceChunk],
) -> list[EvidenceChunk]:
    """Keep current evidence order, then append retrieved chunks with unseen IDs"""
    seen = {chunk.evidence_id for chunk in current}
    merged = list(current)
    for chunk in retrieved:
        if chunk.evidence_id not in seen:
            merged.append(chunk)
            seen.add(chunk.evidence_id)

    return merged


def label_evidence(question: QuestionRecord, evidence: list[EvidenceChunk]) -> EvidenceSet:
    """Build EvidenceSet where condition reflects the gold support coverage"""
    support_ids = set(question.gold_support_ids)
    present_support_ids = support_ids.intersection(chunk.evidence_id for chunk in evidence)
    if present_support_ids == support_ids:
        condition = EvidenceCondition.COMPLETE
    elif present_support_ids:
        condition = EvidenceCondition.PARTIALLY_INCOMPLETE
    else:
        condition = EvidenceCondition.FULLY_INCOMPLETE
    return EvidenceSet(
        question_id=question.question_id,
        condition=condition,
        evidence=evidence,
        gold_support_ids=question.gold_support_ids,
    )


class AdaptiveRetrievalController:
    """
    Assess, retrieve again on insufficiency, then answer or abstain

    ``allow_abstention=True`` is adaptive_retrieval_abstention system
    ``allow_abstention=False`` is adaptive_retrieval system,
                    which answers even when evidence is insufficient

    """

    def __init__(
        self,
        retriever: Retriever,
        assessor: SufficiencyAssessor,
        answer_generator: AnswerGenerator,
        *,
        allow_abstention: bool = True,
        followup_top_k: int = 3,
    ) -> None:
        if followup_top_k < 1:
            raise ValueError("followup_top_k must be positive")

        self._retriever = retriever
        self._answer_generator = answer_generator
        self._responder = RAGAbstentionResponder(assessor, answer_generator)
        self.allow_abstention = allow_abstention
        self.followup_top_k = followup_top_k

    @property
    def system_variant(self) -> SystemVariant:
        """Return the system label stamped on record the configuration produces"""
        if self.allow_abstention:
            return SystemVariant.ADAPTIVE_RETRIEVAL_ABSTENTION
        return SystemVariant.ADAPTIVE_RETRIEVAL

    def run(
        self,
        question: QuestionRecord,
        initial_retrieval: RetrievalResult,
        *,
        run_id: str,
        initial_outcome: RAGAbstentionOutcome | None = None,
    ) -> AdaptiveCaseResult:
        """
        Run one case from saved initial retrieval, with at most one follow-up

        `initial_outcome` reuses a saved round-0 baseline outcome for the same question, retrieval, model and prompts
        """

        if not run_id.strip():
            raise ValueError("run_id must not be empty")
        if initial_retrieval.stage != RetrievalStage.INITIAL:
            raise ValueError("adaptive control starts from an initial retrieval result")
        if initial_retrieval.question_id != question.question_id:
            raise ValueError("question and retrieval question_id values must match")

        # label what round 0 found (complete / partial / fully incomplete)
        initial = label_evidence(question, initial_retrieval.evidence)

        # First step is baseline assessment, to judge whether round 0 evidence is enough
        # if reply stays invalid after one repair retry, no decision, so no record is written
        first = initial_outcome or self._responder.run(question, initial)
        if first.sufficiency_decision is None:
            return AdaptiveCaseResult(
                experiment=None,
                final_answer=first.answer,
                errors=[first.error] if first.error else [],
            )
        round0 = AdaptiveRoundRecord(
            round_index=0,
            retrieval=initial_retrieval,
            sufficiency_decision=first.sufficiency_decision,
        )

        # Second step is if there is sufficient evidence, the responder has generated an answer
        # This stops after one round and costs no more than baseline
        if first.sufficiency_decision.status == SufficiencyStatus.SUFFICIENT:
            errors = [first.error] if first.error else []
            return self._result(question, initial, [round0], initial, first.answer, run_id, errors)

        # Third step is follow up search, to find missing paragraph with search_query
        query = first.sufficiency_decision.search_query
        if query is None:  # SufficiencyDecision requires it for need_more_evidence
            raise AssertionError("need_more_evidence decision without search_query")
        followup = self._retriever.retrieve(
            RetrievalRequest(
                question_id=question.question_id,
                query=query,
                stage=RetrievalStage.FOLLOW_UP,
                top_k=self.followup_top_k,
                exclude_evidence_ids=initial.evidence_ids,
            )
        )
        merged = label_evidence(question, merge_evidence(initial.evidence, followup.evidence))

        # Fourth step is if nothing is found, a second assessment would get identical input
        # With greedy decoding, there will be an identical reply, so this model call is skipped
        if merged.evidence_ids == initial.evidence_ids:
            answer = self._final_without_support(
                question, initial, AbstentionReason.RETRIEVAL_FAILED
            )
            return self._result(
                question, initial, [round0], initial, answer, run_id, [], followup_empty=True
            )

        # Fifth step is the second assessment, to ask model again with merged evidence
        # If the reply is invalid, round 1 cannot be stored, so keep round 0 and abstain
        second = self._responder.run(question, merged)
        if second.sufficiency_decision is None:
            return self._result(
                question,
                initial,
                [round0],
                initial,
                second.answer,
                run_id,
                [second.error] if second.error else [],
            )
        round1 = AdaptiveRoundRecord(
            round_index=1,
            retrieval=followup,
            sufficiency_decision=second.sufficiency_decision,
        )

        # Sixth step (recovered) is where the merged evidence is sufficient
        # Recovery rate counts these
        if second.sufficiency_decision.status == SufficiencyStatus.SUFFICIENT:
            errors = [second.error] if second.error else []
            return self._result(
                question, initial, [round0, round1], merged, second.answer, run_id, errors
            )

        # Sixth step (not recovered) is where it is still insufficient, there are no rounds left
        # So abstain, or force answer in adaptive_retrieval
        answer = self._final_without_support(
            question, merged, AbstentionReason.MAX_RETRIEVAL_ROUNDS_REACHED
        )
        return self._result(question, initial, [round0, round1], merged, answer, run_id, [])

    def _final_without_support(
        self,
        question: QuestionRecord,
        evidence: EvidenceSet,
        reason: AbstentionReason,
    ) -> AnswerResult:
        """Abstain, or in adaptive_retrieval answer from the best evidence available"""
        if self.allow_abstention:
            LOGGER.info("Abstained question_id=%s reason=%s", question.question_id, reason.value)
            return AnswerResult(
                question_id=question.question_id,
                answer=None,
                abstained=True,
                abstention_reason=reason,
                used_evidence_ids=[],
            )

        try:
            answer = self._answer_generator.generate(question, evidence)
        except ModelOutputError as exc:
            LOGGER.warning(
                "Abstained question_id=%s reason=%s task=%s",
                question.question_id,
                exc.abstention_reason.value,
                exc.task,
            )
            return self._abstain(question.question_id, exc.abstention_reason)
        if not set(answer.used_evidence_ids).issubset(evidence.evidence_ids):
            return self._abstain(question.question_id, AbstentionReason.INVALID_MODEL_OUTPUT)

        return answer

    @staticmethod
    def _abstain(question_id: str, reason: AbstentionReason) -> AnswerResult:
        """Create a typed abstention"""
        return AnswerResult(
            question_id=question_id,
            answer=None,
            abstained=True,
            abstention_reason=reason,
            used_evidence_ids=[],
        )

    def _result(
        self,
        question: QuestionRecord,
        initial: EvidenceSet,
        rounds: list[AdaptiveRoundRecord],
        final: EvidenceSet,
        answer: AnswerResult,
        run_id: str,
        errors: list[ModelOutputError],
        *,
        followup_empty: bool = False,
    ) -> AdaptiveCaseResult:
        experiment = ExperimentRecord(
            experiment_id=make_experiment_id(
                question.question_id,
                initial.condition,
                self.system_variant,
                run_id,
            ),
            question=question,
            condition=initial.condition,
            system_variant=self.system_variant,
            initial_evidence=initial,
            rounds=rounds,
            final_evidence_ids=final.evidence_ids,
            final_answer=answer,
            gold_sufficient_initially=initial.gold_sufficient,
            gold_sufficient_finally=final.gold_sufficient,
            model_predicted_sufficient_initially=(
                rounds[0].sufficiency_decision.status == SufficiencyStatus.SUFFICIENT
            ),
            model_predicted_sufficient_finally=(
                rounds[-1].sufficiency_decision.status == SufficiencyStatus.SUFFICIENT
            ),
            metadata={
                "controller_version": CONTROLLER_VERSION,
                "allow_abstention": self.allow_abstention,
                "retrieval_rounds": len(rounds),
                "followup_empty": followup_empty,
            },
        )
        return AdaptiveCaseResult(experiment=experiment, final_answer=answer, errors=errors)
