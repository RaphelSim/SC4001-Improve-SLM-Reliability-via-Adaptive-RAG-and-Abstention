"""Single-shot and bounded adaptive retrieval over the same scoped retriever."""

from __future__ import annotations

from reliable_rag.contracts.enums import RetrievalStage, SufficiencyStatus
from reliable_rag.contracts.records import ModelEvidence, QuestionRecord, RetrievalRequest
from reliable_rag.contracts.retrieval_study import RetrievalStudyRecord, StudyRound


def _finish(question: QuestionRecord, system: str, rounds: list[StudyRound], *, error=None):
    """Evaluation-only annotation after controller decisions have been made."""
    gold_ids = question.gold_paragraph_ids
    if not gold_ids:
        raise ValueError("prepared question lacks frozen gold_paragraph_ids")
    gold = set(gold_ids)
    raw = [chunk.evidence_id for item in rounds for chunk in item.retrieval.evidence]
    unique = list(dict.fromkeys(raw))
    seen: set[str] = set()
    annotated = []
    for item in rounds:
        seen.update(chunk.evidence_id for chunk in item.retrieval.evidence)
        annotated.append(item.model_copy(update={"gold_complete_after_round": gold.issubset(seen)}))
    gold_retrieved = [item for item in unique if item in gold]
    noise = [item for item in unique if item not in gold]
    final_status = (
        None if system == "baseline" else "ERROR" if error else rounds[-1].model_status_after_round
    )
    return RetrievalStudyRecord(
        question_id=question.question_id,
        question=question.question,
        gold_paragraph_ids=gold_ids,
        candidate_paragraph_ids=question.candidate_evidence_ids,
        system=system,
        rounds=annotated,
        raw_retrieved_ids=raw,
        unique_retrieved_ids=unique,
        gold_retrieved_ids=gold_retrieved,
        noise_ids=noise,
        gold_evidence_percentage=100 * len(gold_retrieved) / len(gold),
        noise_percentage=100 * len(noise) / len(unique) if unique else 0,
        raw_paragraphs_retrieved=len(raw),
        unique_paragraphs_retrieved=len(unique),
        duplicate_retrieval_count=len(raw) - len(unique),
        retrieval_rounds=len(rounds),
        final_model_status=final_status,
        error_type=type(error).__name__ if error else None,
        error_message=str(error) if error else None,
    )


def run_retrieval_case(
    question: QuestionRecord,
    *,
    retriever,
    assessor=None,
    system: str,
    top_k: int,
    max_rounds: int = 1,
) -> RetrievalStudyRecord:
    """Controller never reads gold labels; _finish adds them after stopping."""
    if system not in {"baseline", "adaptive"} or top_k < 1 or max_rounds < 1:
        raise ValueError("invalid study policy")
    if system == "adaptive" and assessor is None:
        raise ValueError("adaptive policy requires a sufficiency assessor")
    rounds: list[StudyRound] = []
    accumulated = []
    seen: set[str] = set()
    query = question.question
    for number in range(1, 2 if system == "baseline" else max_rounds + 1):
        retrieval = retriever.retrieve(
            RetrievalRequest(
                question_id=question.question_id,
                query=query,
                stage=RetrievalStage.INITIAL if number == 1 else RetrievalStage.FOLLOW_UP,
                top_k=top_k,
            )
        )
        if retrieval.question_id != question.question_id or retrieval.query != query:
            raise ValueError("retriever returned a mismatched question or query")
        new_ids, duplicate_ids = [], []
        for chunk in retrieval.evidence:
            if chunk.evidence_id in seen:
                duplicate_ids.append(chunk.evidence_id)
            else:
                seen.add(chunk.evidence_id)
                new_ids.append(chunk.evidence_id)
                accumulated.append(chunk)
        item = StudyRound(
            round_number=number,
            retrieval=retrieval,
            new_unique_ids=new_ids,
            duplicate_ids=duplicate_ids,
        )
        rounds.append(item)
        if system == "baseline":
            break
        try:
            decision = assessor.assess(
                question,
                ModelEvidence(question_id=question.question_id, evidence=accumulated),
            )
            if decision.status == SufficiencyStatus.SUFFICIENT and decision.missing_information:
                raise ValueError(
                    "sufficient controller decision must have null missing_information"
                )
            item.model_status_after_round = decision.status
            item.missing_information = decision.missing_information
            item.next_query = decision.search_query if number < max_rounds else None
            item.raw_model_responses = list(getattr(assessor, "raw_outputs", []))
        except Exception as exc:
            item.raw_model_responses = list(getattr(assessor, "raw_outputs", []))
            return _finish(question, system, rounds, error=exc)
        if decision.status == SufficiencyStatus.SUFFICIENT or number == max_rounds:
            break
        query = decision.search_query
    return _finish(question, system, rounds)
