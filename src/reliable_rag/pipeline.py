"""Small, non-adaptive integration path for initial retrieval and RAG + Abstention."""

from __future__ import annotations

from dataclasses import dataclass

from reliable_rag.contracts.enums import (
    EvidenceCondition,
    RetrievalStage,
    SufficiencyStatus,
    SystemVariant,
)
from reliable_rag.contracts.ids import make_experiment_id
from reliable_rag.contracts.records import (
    AdaptiveRoundRecord,
    EvidenceSet,
    ExperimentRecord,
    QuestionRecord,
    RetrievalResult,
)
from reliable_rag.model.abstention import RAGAbstentionOutcome, RAGAbstentionResponder


@dataclass(frozen=True)
class InitialRAGAbstentionResult:
    """Integration output for one initial retrieval result and one model assessment."""

    evidence: EvidenceSet
    outcome: RAGAbstentionOutcome
    experiment: ExperimentRecord | None


def evidence_set_from_initial_retrieval(
    question: QuestionRecord,
    retrieval: RetrievalResult,
) -> EvidenceSet:
    """Adapt initial ranked evidence to the sufficiency assessment contract."""
    if question.question_id != retrieval.question_id:
        raise ValueError("question and retrieval question_id values must match")
    if retrieval.stage != RetrievalStage.INITIAL:
        raise ValueError("RAG + Abstention baseline requires an initial retrieval result")

    retrieved_ids = {chunk.evidence_id for chunk in retrieval.evidence}
    candidate_ids = set(question.candidate_evidence_ids)
    if candidate_ids:
        if not set(question.gold_support_ids).issubset(candidate_ids):
            raise ValueError("question gold support IDs must be included in candidate evidence IDs")
        if not retrieved_ids.issubset(candidate_ids):
            raise ValueError(
                "retrieval result contains IDs outside the question's candidate evidence"
            )

    support_ids = set(question.gold_support_ids)
    present_support_ids = support_ids.intersection(retrieved_ids)
    if present_support_ids == support_ids:
        condition = EvidenceCondition.COMPLETE
    elif present_support_ids:
        condition = EvidenceCondition.PARTIALLY_INCOMPLETE
    else:
        condition = EvidenceCondition.FULLY_INCOMPLETE

    return EvidenceSet(
        question_id=question.question_id,
        condition=condition,
        evidence=retrieval.evidence,
        gold_support_ids=question.gold_support_ids,
    )


def run_initial_rag_abstention_case(
    question: QuestionRecord,
    retrieval: RetrievalResult,
    responder: RAGAbstentionResponder,
    *,
    run_id: str,
) -> InitialRAGAbstentionResult:
    """Run one initial-retrieval case and build a standard record when assessable.

    This is a one-retrieval-round baseline. It deliberately leaves follow-up search,
    evidence merging, and adaptive control outside this baseline.
    """
    if not run_id.strip():
        raise ValueError("run_id must not be empty")

    evidence = evidence_set_from_initial_retrieval(question, retrieval)
    outcome = responder.run(question, evidence)
    decision = outcome.sufficiency_decision
    if decision is None:
        return InitialRAGAbstentionResult(evidence=evidence, outcome=outcome, experiment=None)

    model_sufficient = decision.status == SufficiencyStatus.SUFFICIENT
    experiment = ExperimentRecord(
        experiment_id=make_experiment_id(
            question.question_id,
            evidence.condition,
            SystemVariant.RAG_ABSTENTION,
            run_id,
        ),
        question=question,
        condition=evidence.condition,
        system_variant=SystemVariant.RAG_ABSTENTION,
        initial_evidence=evidence,
        rounds=[
            AdaptiveRoundRecord(
                round_index=0,
                retrieval=retrieval,
                sufficiency_decision=decision,
            )
        ],
        final_evidence_ids=evidence.evidence_ids,
        final_answer=outcome.answer,
        gold_sufficient_initially=evidence.gold_sufficient,
        gold_sufficient_finally=evidence.gold_sufficient,
        model_predicted_sufficient_initially=model_sufficient,
        model_predicted_sufficient_finally=model_sufficient,
    )
    return InitialRAGAbstentionResult(evidence=evidence, outcome=outcome, experiment=experiment)
