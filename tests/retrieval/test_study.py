"""Regression checks for the study's labels, duplicate costs, and stopping rules."""

import json

from reliable_rag.config import load_settings
from reliable_rag.contracts.enums import DatasetSplit, SufficiencyStatus
from reliable_rag.contracts.records import (
    EvidenceChunk,
    QuestionRecord,
    RetrievalResult,
    SufficiencyDecision,
)
from reliable_rag.contracts.retrieval_study import RetrievalStudyRecord
from reliable_rag.evaluation.retrieval_study import summarize
from reliable_rag.io.jsonl import read_jsonl, write_jsonl
from reliable_rag.model.errors import ModelOutputError
from reliable_rag.model.sufficiency import SLMSufficiencyAssessor
from reliable_rag.retrieval.study import run_retrieval_case


def _question():
    return QuestionRecord(
        question_id="q1",
        question="Where was the maker based?",
        gold_answer="X",
        split=DatasetSplit.DEVELOPMENT,
        gold_support_ids=["2", "3"],
        gold_paragraph_ids=["2", "3"],
        candidate_evidence_ids=["1", "2", "3", "4"],
    )


class Retriever:
    def __init__(self, rounds):
        self.rounds = rounds
        self.requests = []

    def retrieve(self, request):
        self.requests.append(request)
        ids = self.rounds[len(self.requests) - 1]
        return RetrievalResult(
            question_id=request.question_id,
            query=request.query,
            stage=request.stage,
            top_k=request.top_k,
            evidence=[
                EvidenceChunk(
                    evidence_id=i,
                    document_id=i,
                    title=i,
                    text=f"Text {i}",
                    retrieval_score=float(len(ids) - rank),
                    retrieval_rank=rank + 1,
                )
                for rank, i in enumerate(ids)
            ],
        )


class Assessor:
    def __init__(self, statuses):
        self.statuses = statuses
        self.seen = []
        self.raw_outputs = []

    def assess(self, question, evidence):
        assert not hasattr(evidence, "gold_support_ids")
        self.seen.append([item.evidence_id for item in evidence.evidence])
        status = self.statuses[len(self.seen) - 1]
        self.raw_outputs = [f"raw round {len(self.seen)}"]
        if status == "ERROR":
            raise ModelOutputError(
                task="sufficiency_assessment",
                question_id=question.question_id,
                attempts=2,
                reason="invalid JSON",
            )
        return SufficiencyDecision(
            question_id=question.question_id,
            status=status,
            missing_information=None if status == SufficiencyStatus.SUFFICIENT else "missing fact",
            search_query=None
            if status == SufficiencyStatus.SUFFICIENT
            else f"target {len(self.seen)}",
            supporting_evidence_ids=[evidence.evidence[0].evidence_id],
        )


class V2Assessor:
    is_v2 = True

    def __init__(self, decisions):
        self.decisions = decisions
        self.calls = []
        self.raw_outputs = []

    def assess(self, question, evidence, **context):
        self.calls.append((list(evidence.evidence_ids), context))
        self.raw_outputs = [json.dumps({"call": len(self.calls)})]
        return self.decisions[len(self.calls) - 1]


def v2_decision(status, search_query=None):
    return SufficiencyDecision(
        question_id="q1",
        status=status,
        missing_information=None if status == SufficiencyStatus.SUFFICIENT else "actor relation",
        search_query=search_query,
        supporting_evidence_ids=["1"],
        required_information=["identify the actor"],
        established_facts=[],
        missing_information_items=[]
        if status == SufficiencyStatus.SUFFICIENT
        else ["identify the actor"],
    )


def test_baseline_top_five_never_calls_model():
    retriever = Retriever([["1", "3", "4", "2"]])
    record = run_retrieval_case(_question(), retriever=retriever, system="baseline", top_k=5)
    assert record.retrieval_rounds == 1
    assert retriever.requests[0].top_k == 5
    assert record.final_model_status is None
    assert record.gold_evidence_percentage == 100


def test_adaptive_raw_duplicates_and_unique_noise(tmp_path):
    retriever = Retriever([["1", "3"], ["3", "4"], ["2", "4"]])
    assessor = Assessor(
        [
            SufficiencyStatus.NEED_MORE_EVIDENCE,
            SufficiencyStatus.NEED_MORE_EVIDENCE,
            SufficiencyStatus.SUFFICIENT,
        ]
    )
    record = run_retrieval_case(
        _question(),
        retriever=retriever,
        assessor=assessor,
        system="adaptive",
        top_k=2,
        max_rounds=5,
    )
    assert record.raw_retrieved_ids == ["1", "3", "3", "4", "2", "4"]
    assert record.unique_retrieved_ids == ["1", "3", "4", "2"]
    assert record.gold_evidence_percentage == 100
    assert record.noise_percentage == 50
    assert record.raw_paragraphs_retrieved == 6
    assert record.duplicate_retrieval_count == 2
    assert record.retrieval_rounds == 3
    assert assessor.seen == [["1", "3"], ["1", "3", "4"], ["1", "3", "4", "2"]]
    assert [item.gold_complete_after_round for item in record.rounds] == [False, False, True]
    assert [r.query for r in retriever.requests] == [_question().question, "target 1", "target 2"]
    path = tmp_path / "results.jsonl"
    write_jsonl(path, [record])
    assert read_jsonl(path, RetrievalStudyRecord)[0] == record
    summary = summarize([record])
    assert summary["stopping_round_counts"] == {"1": 0, "2": 0, "3": 1, "4": 0, "5": 0}
    assert summary["duplicate_percentage_raw"] == 100 * 2 / 6


def test_adaptive_stops_at_five_without_gold_oracle():
    retriever = Retriever([["1", "3"]] * 5)
    assessor = Assessor([SufficiencyStatus.NEED_MORE_EVIDENCE] * 5)
    record = run_retrieval_case(
        _question(),
        retriever=retriever,
        assessor=assessor,
        system="adaptive",
        top_k=2,
        max_rounds=5,
    )
    assert record.retrieval_rounds == 5
    assert record.final_model_status == SufficiencyStatus.NEED_MORE_EVIDENCE
    assert record.rounds[-1].next_query is None
    assert record.raw_paragraphs_retrieved == 10
    assert record.unique_paragraphs_retrieved == 2


def test_model_failure_preserves_partial_retrieval():
    record = run_retrieval_case(
        _question(),
        retriever=Retriever([["1", "3"]]),
        assessor=Assessor(["ERROR"]),
        system="adaptive",
        top_k=2,
        max_rounds=5,
    )
    assert record.final_model_status == "ERROR"
    assert record.retrieval_rounds == 1
    assert record.raw_paragraphs_retrieved == 2
    assert record.rounds[0].raw_model_responses == ["raw round 1"]


def test_real_assessor_prompt_contains_no_gold_labels():
    class Client:
        prompts = []

        def generate(self, prompt, *, max_new_tokens=None):
            self.prompts.append(prompt)
            return json.dumps(
                {
                    "question_id": "q1",
                    "status": "sufficient",
                    "missing_information": None,
                    "search_query": None,
                    "supporting_evidence_ids": ["1"],
                }
            )

    from reliable_rag.contracts.records import ModelEvidence

    settings = load_settings("configs/retrieval_study.yaml")
    client = Client()
    assessor = SLMSufficiencyAssessor(client, settings)
    assessor.assess(
        _question(),
        ModelEvidence(
            question_id="q1",
            evidence=[EvidenceChunk(evidence_id="1", document_id="1", text="Text 1")],
        ),
    )
    assert "gold_support_ids" not in client.prompts[0]
    assert "gold_paragraph_ids" not in client.prompts[0]
    assert '"evidence_id": "2"' not in client.prompts[0]


def test_v2_retrieval_history_stagnation_and_loop_guard_regeneration():
    retriever = Retriever([["1", "3"], ["1", "3"], ["2"]])
    assessor = V2Assessor(
        [
            v2_decision(SufficiencyStatus.NEED_MORE_EVIDENCE, "bridge actor"),
            v2_decision(SufficiencyStatus.NEED_MORE_EVIDENCE, "bridge actor"),
            v2_decision(SufficiencyStatus.NEED_MORE_EVIDENCE, "bridge person performer"),
            v2_decision(SufficiencyStatus.SUFFICIENT),
        ]
    )
    record = run_retrieval_case(
        _question(),
        retriever=retriever,
        assessor=assessor,
        system="adaptive",
        top_k=2,
        max_rounds=5,
    )
    assert record.final_model_status == SufficiencyStatus.SUFFICIENT
    assert [request.query for request in retriever.requests] == [
        _question().question,
        "bridge actor",
        "bridge person performer",
    ]
    assert assessor.calls[1][1]["retrieval_stagnated"] is True
    assert assessor.calls[1][1]["force_query_reformulation"] is False
    assert assessor.calls[2][1]["force_query_reformulation"] is True
    assert assessor.calls[2][1]["retrieval_history"][1]["duplicate_ids"] == ["1", "3"]
    assert record.rounds[1].retrieval_stagnated is True
    assert record.rounds[1].loop_guard_activated is True
    assert len(record.rounds[1].raw_model_responses) == 2


def test_v2_output_contains_per_round_evaluator_trajectory():
    retriever = Retriever([["1", "2"], ["3"]])
    assessor = V2Assessor(
        [
            v2_decision(SufficiencyStatus.NEED_MORE_EVIDENCE, "another query"),
            v2_decision(SufficiencyStatus.SUFFICIENT),
        ]
    )
    record = run_retrieval_case(
        _question(),
        retriever=retriever,
        assessor=assessor,
        system="adaptive",
        top_k=2,
        max_rounds=5,
    )
    assert record.rounds[0].cumulative_raw_paragraphs == 2
    assert record.rounds[1].cumulative_raw_paragraphs == 3
    assert record.rounds[1].cumulative_unique_paragraph_ids == ["1", "2", "3"]
    assert record.rounds[1].cumulative_gold_evidence_percentage == 100


def test_v2_prompt_has_history_and_never_receives_evaluator_fields():
    class Client:
        prompts = []

        def generate(self, prompt, *, max_new_tokens=None):
            self.prompts.append(prompt)
            return json.dumps(
                {
                    "question_id": "q1",
                    "required_information": ["the maker's country"],
                    "established_facts": [],
                    "missing_information_items": ["the maker's country"],
                    "status": "need_more_evidence",
                    "missing_information": "the maker's country",
                    "search_query": "maker headquarters country",
                    "supporting_evidence_ids": ["1"],
                }
            )

    from reliable_rag.contracts.records import ModelEvidence

    settings = load_settings("configs/retrieval_study_v2_1.yaml")
    client = Client()
    assessor = SLMSufficiencyAssessor(client, settings)
    decision = assessor.assess(
        _question(),
        ModelEvidence(
            question_id="q1",
            evidence=[EvidenceChunk(evidence_id="1", document_id="1", text="Maker is Acme.")],
        ),
        retrieval_history=[
            {
                "round_number": 1,
                "query": "maker country",
                "retrieved_ids": ["1"],
                "new_unique_ids": ["1"],
                "duplicate_ids": [],
            }
        ],
        retrieval_stagnated=True,
    )
    assert decision.status == SufficiencyStatus.NEED_MORE_EVIDENCE
    rendered = client.prompts[0]
    assert '"query": "maker country"' in rendered
    assert "Did the latest retrieval stagnate (zero new unique IDs)? true" in rendered
    assert "gold_paragraph_ids" not in rendered
    assert "gold_complete_after_round" not in rendered
    assert "gold_evidence_percentage" not in rendered
