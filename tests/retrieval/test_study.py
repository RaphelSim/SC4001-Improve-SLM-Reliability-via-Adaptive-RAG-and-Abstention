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
