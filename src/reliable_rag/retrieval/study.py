"""Single-shot and bounded adaptive retrieval over the same scoped retriever."""

from __future__ import annotations

import logging

from reliable_rag.contracts.enums import RetrievalStage, SufficiencyStatus
from reliable_rag.contracts.records import ModelEvidence, QuestionRecord, RetrievalRequest
from reliable_rag.contracts.retrieval_study import RetrievalStudyRecord, StudyRound

LOGGER = logging.getLogger(__name__)


def _normalize_query(value: str) -> str:
    """Conservative exact normalization for the inference-time loop guard."""
    return " ".join("".join(c.lower() if c.isalnum() else " " for c in value).split())


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
    cumulative_unique: list[str] = []
    cumulative_raw = 0
    for item in rounds:
        round_ids = [chunk.evidence_id for chunk in item.retrieval.evidence]
        cumulative_raw += len(round_ids)
        seen.update(round_ids)
        cumulative_unique = list(dict.fromkeys([*cumulative_unique, *round_ids]))
        cumulative_gold = sorted(gold.intersection(cumulative_unique))
        annotated.append(
            item.model_copy(
                update={
                    "gold_complete_after_round": gold.issubset(seen),
                    "cumulative_raw_paragraphs": cumulative_raw,
                    "cumulative_unique_paragraph_ids": cumulative_unique,
                    "cumulative_gold_paragraph_ids": cumulative_gold,
                    "cumulative_gold_evidence_percentage": 100 * len(cumulative_gold) / len(gold),
                    "cumulative_unique_noise_percentage": 100
                    * (len(cumulative_unique) - len(cumulative_gold))
                    / len(cumulative_unique)
                    if cumulative_unique
                    else 0,
                }
            )
        )
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
    history: list[dict] = []
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
            retrieval_stagnated=not new_ids,
        )
        rounds.append(item)
        history_item = {
            "round_number": number,
            "query": query,
            "retrieved_ids": [chunk.evidence_id for chunk in retrieval.evidence],
            "new_unique_ids": new_ids,
            "duplicate_ids": duplicate_ids,
            "retrieval_stagnated": not new_ids,
        }
        history.append(history_item)
        if system == "baseline":
            break
        try:
            model_evidence = ModelEvidence(question_id=question.question_id, evidence=accumulated)
            if getattr(assessor, "is_v2", False):
                decision = assessor.assess(
                    question,
                    model_evidence,
                    retrieval_history=history,
                    retrieval_stagnated=not new_ids,
                    force_query_reformulation=False,
                )
                item.raw_model_responses.extend(list(getattr(assessor, "raw_outputs", [])))
                item.required_information = list(decision.required_information)
                item.established_facts = list(decision.established_facts)
                item.missing_information_items = list(decision.missing_information_items)
                if decision.status == SufficiencyStatus.NEED_MORE_EVIDENCE and number < max_rounds:
                    failed_queries = {
                        _normalize_query(entry["query"])
                        for entry in history
                        if entry["retrieval_stagnated"]
                    }
                    if _normalize_query(decision.search_query) in failed_queries:
                        item.loop_guard_activated = True
                        LOGGER.warning(
                            "V2 query loop guard activated question_id=%s round=%s repeated_query=%r",
                            question.question_id,
                            number,
                            decision.search_query,
                        )
                        history_item["loop_guard_activated"] = True
                        history_item["rejected_repeated_query"] = decision.search_query
                        history_item["controller_status"] = decision.status.value
                        history_item["missing_information"] = decision.missing_information
                        history_item["required_information"] = list(decision.required_information)
                        history_item["established_facts"] = list(decision.established_facts)
                        history_item["missing_information_items"] = list(
                            decision.missing_information_items
                        )
                        decision = assessor.assess(
                            question,
                            model_evidence,
                            retrieval_history=history,
                            retrieval_stagnated=True,
                            force_query_reformulation=True,
                        )
                        item.raw_model_responses.extend(list(getattr(assessor, "raw_outputs", [])))
                        item.required_information = list(decision.required_information)
                        item.established_facts = list(decision.established_facts)
                        item.missing_information_items = list(decision.missing_information_items)
                        if _normalize_query(decision.search_query or "") in failed_queries:
                            raise ValueError(
                                "V2 query regeneration repeated a failed stagnant query"
                            )
            else:
                decision = assessor.assess(question, model_evidence)
            if decision.status == SufficiencyStatus.SUFFICIENT and decision.missing_information:
                raise ValueError(
                    "sufficient controller decision must have null missing_information"
                )
            item.model_status_after_round = decision.status
            item.missing_information = decision.missing_information
            item.next_query = decision.search_query if number < max_rounds else None
            history_item["controller_status"] = decision.status.value
            history_item["missing_information"] = decision.missing_information
            history_item["next_query"] = item.next_query
            history_item["required_information"] = item.required_information
            history_item["established_facts"] = item.established_facts
            history_item["missing_information_items"] = item.missing_information_items
            if not item.raw_model_responses:
                item.raw_model_responses = list(getattr(assessor, "raw_outputs", []))
        except Exception as exc:
            if not item.raw_model_responses:
                item.raw_model_responses = list(getattr(assessor, "raw_outputs", []))
            return _finish(question, system, rounds, error=exc)
        if decision.status == SufficiencyStatus.SUFFICIENT or number == max_rounds:
            break
        query = decision.search_query
    return _finish(question, system, rounds)
