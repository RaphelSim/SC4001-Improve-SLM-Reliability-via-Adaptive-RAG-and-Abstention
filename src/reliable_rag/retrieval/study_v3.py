"""V3 adaptive retrieval: Top-2 unseen returns and cumulative unique model context."""

from __future__ import annotations

from reliable_rag.contracts.enums import RetrievalStage, SufficiencyStatus
from reliable_rag.contracts.records import (
    ModelEvidence,
    QuestionRecord,
    RetrievalRequest,
    RetrievalResult,
)
from reliable_rag.contracts.retrieval_study import RetrievalStudyRecord, StudyRound
from reliable_rag.model.errors import ModelOutputError
from reliable_rag.retrieval.study import _finish, _normalize_query


def _top_unseen(question, retriever, query, number, seen, top_k):
    """Rank once using the shared retriever, then fill slots below seen hits."""
    stage = RetrievalStage.INITIAL if number == 1 else RetrievalStage.FOLLOW_UP
    ranked = retriever.retrieve(
        RetrievalRequest(
            question_id=question.question_id,
            query=query,
            stage=stage,
            top_k=len(question.candidate_evidence_ids),
        )
    )
    if ranked.question_id != question.question_id or ranked.query != query:
        raise ValueError("retriever returned a mismatched question or query")
    selected = []
    skipped = []
    considered = []
    for chunk in ranked.evidence:
        considered.append(chunk.evidence_id)
        if chunk.evidence_id in seen:
            skipped.append(chunk.evidence_id)
        else:
            selected.append(chunk)
            if len(selected) == top_k:
                break
    result = RetrievalResult(
        question_id=question.question_id,
        query=query,
        stage=stage,
        top_k=top_k,
        evidence=selected,
    )
    return result, skipped, considered


def run_v3_case(
    question: QuestionRecord,
    *,
    retriever,
    assessor,
    top_k: int = 2,
    max_rounds: int = 5,
) -> RetrievalStudyRecord:
    """Run inference without consulting the question's evaluator-only gold fields."""
    if top_k < 1 or max_rounds < 1 or not question.candidate_evidence_ids:
        raise ValueError("V3 requires positive Top-k/rounds and a nonempty candidate pool")
    rounds: list[StudyRound] = []
    accumulated = []
    seen: set[str] = set()
    query = question.question
    controller_calls = 0
    repair_calls = 0
    regeneration_calls = 0

    def finish(reason: str, *, error=None) -> RetrievalStudyRecord:
        if rounds:
            rounds[-1].stop_reason = reason
        record = _finish(question, "adaptive", rounds, error=error)
        values = record.model_dump(mode="json")
        values.update(
            controller_version="v3",
            stop_reason=reason,
            controller_model_calls=controller_calls,
            repair_model_calls=repair_calls,
            query_regeneration_model_calls=regeneration_calls,
            total_reasoning_model_calls=controller_calls + repair_calls + regeneration_calls,
        )
        return RetrievalStudyRecord.model_validate(values)

    for number in range(1, max_rounds + 1):
        retrieval, skipped, considered = _top_unseen(
            question, retriever, query, number, seen, top_k
        )
        new_ids = [chunk.evidence_id for chunk in retrieval.evidence]
        if not new_ids:
            return finish("retrieval_pool_exhausted")
        seen.update(new_ids)
        accumulated.extend(retrieval.evidence)
        item = StudyRound(
            round_number=number,
            retrieval=retrieval,
            new_unique_ids=new_ids,
            duplicate_ids=[],
            ranked_seen_ids_skipped=skipped,
            ranked_candidate_ids_considered=considered,
        )
        rounds.append(item)
        model_evidence = ModelEvidence(question_id=question.question_id, evidence=accumulated)
        previous_queries = [r.retrieval.query for r in rounds]
        try:
            controller_calls += 1
            try:
                decision = assessor.assess(
                    question,
                    model_evidence,
                    previous_search_queries=previous_queries,
                )
            finally:
                repair_calls += assessor.last_repair_calls
                item.repair_used |= assessor.last_repair_calls > 0
                item.raw_model_responses.extend(assessor.raw_outputs)
            proposed = decision.search_query
            if proposed is None:
                item.model_status_after_round = SufficiencyStatus.SUFFICIENT
                return finish("sufficient")
            item.model_status_after_round = SufficiencyStatus.NEED_MORE_EVIDENCE
            item.next_query = proposed
            attempted = {_normalize_query(previous) for previous in previous_queries}
            if number == max_rounds:
                return finish("max_rounds_reached")
            if _normalize_query(proposed) in attempted:
                item.loop_guard_activated = True
                item.query_regeneration_used = True
                regeneration_calls += 1
                try:
                    replacement = assessor.assess(
                        question,
                        model_evidence,
                        previous_search_queries=previous_queries,
                        rejected_query=proposed,
                    )
                finally:
                    repair_calls += assessor.last_repair_calls
                    item.repair_used |= assessor.last_repair_calls > 0
                    item.raw_model_responses.extend(assessor.raw_outputs)
                proposed = replacement.search_query
                item.next_query = proposed
                if proposed is None or _normalize_query(proposed) in attempted:
                    return finish("repeated_query")
            if len(seen) == len(question.candidate_evidence_ids):
                return finish("retrieval_pool_exhausted")
            query = proposed
        except Exception as exc:
            reason = (
                "model_output_error" if isinstance(exc, ModelOutputError) else "controller_error"
            )
            return finish(reason, error=exc)
    raise AssertionError("unreachable: V3 always returns within max_rounds")
