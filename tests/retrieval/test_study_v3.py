"""V3 unseen retrieval, cumulative context, and clean stopping outcomes."""

import json

import pytest

from reliable_rag.contracts.enums import DatasetSplit, SufficiencyStatus
from reliable_rag.contracts.records import EvidenceChunk, QuestionRecord, RetrievalResult
from reliable_rag.model.errors import ModelOutputError
from reliable_rag.model.retrieval_query_v3 import V3QueryDecision
from reliable_rag.retrieval.study_v3 import run_v3_case


def question(ids=None, gold=None):
    ids = ids or ["3", "4", "5", "7"]
    gold = gold or ["5", "7"]
    return QuestionRecord(
        question_id="q1",
        question="What nationality was the named brother?",
        gold_answer="X",
        split=DatasetSplit.DEVELOPMENT,
        gold_support_ids=gold,
        gold_paragraph_ids=gold,
        candidate_evidence_ids=ids,
    )


class RankedRetriever:
    def __init__(self, rankings):
        self.rankings = rankings
        self.requests = []

    def retrieve(self, request):
        self.requests.append(request)
        ranking = self.rankings[len(self.requests) - 1]
        return RetrievalResult(
            question_id=request.question_id,
            query=request.query,
            stage=request.stage,
            top_k=request.top_k,
            evidence=[
                EvidenceChunk(
                    evidence_id=evidence_id,
                    document_id=evidence_id,
                    title=evidence_id,
                    text=f"Text {evidence_id}",
                    retrieval_score=1 - rank / 10,
                    retrieval_rank=rank + 1,
                )
                for rank, evidence_id in enumerate(ranking[: request.top_k])
            ],
        )


class QueryAssessor:
    def __init__(self, queries, repairs=None):
        self.queries = queries
        self.repairs = repairs or [0] * len(queries)
        self.calls = []
        self.raw_outputs = []
        self.last_repair_calls = 0

    def assess(self, question, evidence, *, previous_search_queries, rejected_query=None):
        assert not hasattr(evidence, "gold_paragraph_ids")
        self.calls.append(
            (list(evidence.evidence_ids), list(previous_search_queries), rejected_query)
        )
        index = len(self.calls) - 1
        self.last_repair_calls = self.repairs[index]
        self.raw_outputs = [json.dumps({"search_query": self.queries[index]})]
        if self.queries[index] == "ERROR":
            raise ModelOutputError(
                task="v3_retrieval_query",
                question_id=question.question_id,
                attempts=2,
                reason="invalid JSON",
            )
        return V3QueryDecision(search_query=self.queries[index])


@pytest.mark.parametrize("second_ranking", [["3", "5", "7", "4"], ["3", "4", "5", "7"]])
def test_top_two_unseen_fills_slots_and_keeps_cumulative_context(second_ranking):
    retriever = RankedRetriever([["3", "4", "5", "7"], second_ranking])
    assessor = QueryAssessor(["named brother nationality", None])
    record = run_v3_case(question(), retriever=retriever, assessor=assessor)
    assert record.raw_retrieved_ids == ["3", "4", "5", "7"]
    assert record.rounds[1].new_unique_ids == ["5", "7"]
    assert record.rounds[1].ranked_seen_ids_skipped == (
        ["3"] if second_ranking[1] == "5" else ["3", "4"]
    )
    assert record.rounds[1].retrieval.evidence[0].retrieval_rank > 1
    assert assessor.calls[1][0] == ["3", "4", "5", "7"]
    assert assessor.calls[1][1] == [question().question, "named brother nationality"]
    assert record.duplicate_retrieval_count == 0
    assert record.raw_paragraphs_retrieved == record.unique_paragraphs_retrieved == 4
    assert record.gold_evidence_percentage == 100
    assert record.stop_reason == "sufficient"


def test_partial_pool_and_exhaustion_are_clean():
    q = question(ids=["3", "4", "5"], gold=["5"])
    record = run_v3_case(
        q,
        retriever=RankedRetriever([["3", "4", "5"], ["3", "4", "5"]]),
        assessor=QueryAssessor(["find 5", "another query"]),
    )
    assert [len(r.retrieval.evidence) for r in record.rounds] == [2, 1]
    assert record.stop_reason == "retrieval_pool_exhausted"
    assert record.final_model_status == SufficiencyStatus.NEED_MORE_EVIDENCE
    assert record.error_type is None


def test_repeated_query_stops_cleanly_after_one_regeneration():
    q = question()
    assessor = QueryAssessor([q.question, q.question])
    record = run_v3_case(q, retriever=RankedRetriever([["3", "4", "5", "7"]]), assessor=assessor)
    assert record.stop_reason == "repeated_query"
    assert record.error_type is None
    assert record.rounds[0].query_regeneration_used
    assert record.controller_model_calls == 1
    assert record.query_regeneration_model_calls == 1
    assert record.total_reasoning_model_calls == 2
    assert assessor.calls[1][2] == q.question


def test_max_five_rounds_retains_requested_query_and_never_runs_sixth():
    ids = [str(i) for i in range(12)]
    q = question(ids=ids, gold=["10", "11"])
    ranking = [ids for _ in range(5)]
    retriever = RankedRetriever(ranking)
    assessor = QueryAssessor([f"query {i}" for i in range(5)])
    record = run_v3_case(q, retriever=retriever, assessor=assessor)
    assert record.retrieval_rounds == len(retriever.requests) == 5
    assert record.raw_paragraphs_retrieved == record.unique_paragraphs_retrieved == 10
    assert record.stop_reason == "max_rounds_reached"
    assert record.rounds[-1].next_query == "query 4"


def test_repair_and_model_output_error_accounting():
    retriever = RankedRetriever([["3", "4", "5", "7"]])
    record = run_v3_case(
        question(), retriever=retriever, assessor=QueryAssessor([None], repairs=[1])
    )
    assert record.rounds[0].repair_used
    assert (
        record.controller_model_calls,
        record.repair_model_calls,
        record.total_reasoning_model_calls,
    ) == (1, 1, 2)
    error = run_v3_case(
        question(),
        retriever=RankedRetriever([["3", "4", "5", "7"]]),
        assessor=QueryAssessor(["ERROR"], repairs=[1]),
    )
    assert error.stop_reason == "model_output_error"
    assert error.final_model_status == "ERROR"
    assert error.raw_paragraphs_retrieved == 2
    assert error.total_reasoning_model_calls == 2
