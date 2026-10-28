"""Contract serialization and conditional validation tests."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from reliable_rag.contracts.enums import (
    AbstentionReason,
    DatasetSplit,
    EvidenceCondition,
    RetrievalStage,
    SufficiencyStatus,
    SystemVariant,
)
from reliable_rag.contracts.ids import make_evidence_id, make_experiment_id
from reliable_rag.contracts.records import (
    AnswerResult,
    EvidenceChunk,
    EvidenceSet,
    ExperimentRecord,
    SufficiencyDecision,
)

FIXTURES = Path(__file__).parents[1] / "fixtures"


def test_enum_values_are_stable_serialized_strings() -> None:
    assert DatasetSplit.DEVELOPMENT.value == "development"
    assert EvidenceCondition.PARTIALLY_INCOMPLETE.value == "partially_incomplete"
    assert SystemVariant.ADAPTIVE_RETRIEVAL_ABSTENTION.value == "adaptive_retrieval_abstention"
    assert SufficiencyStatus.NEED_MORE_EVIDENCE.value == "need_more_evidence"
    assert AbstentionReason.MAX_RETRIEVAL_ROUNDS_REACHED.value == "max_retrieval_rounds_reached"
    assert RetrievalStage.FOLLOW_UP.value == "follow_up"
    serialized = json.dumps({"condition": EvidenceCondition.COMPLETE})
    assert serialized == '{"condition": "complete"}'


def test_evidence_chunk_rejects_blank_text_and_invalid_rank() -> None:
    base = {"evidence_id": "doc::0", "document_id": "doc", "text": "Useful fact."}
    assert EvidenceChunk.model_validate(base).title is None
    with pytest.raises(ValidationError, match="must not be empty"):
        EvidenceChunk.model_validate({**base, "text": "   "})
    with pytest.raises(ValidationError):
        EvidenceChunk.model_validate({**base, "retrieval_rank": 0})


def test_evidence_set_enforces_controlled_support_condition() -> None:
    complete = json.loads((FIXTURES / "complete_evidence.json").read_text(encoding="utf-8"))
    partial = json.loads((FIXTURES / "partial_evidence.json").read_text(encoding="utf-8"))
    assert EvidenceSet.model_validate(complete).gold_sufficient is True
    assert EvidenceSet.model_validate(partial).gold_sufficient is False
    partial["condition"] = "complete"
    with pytest.raises(ValidationError, match="complete evidence must contain"):
        EvidenceSet.model_validate(partial)


def test_sufficiency_decision_conditional_fields() -> None:
    sufficient = SufficiencyDecision(
        question_id="q1", status=SufficiencyStatus.SUFFICIENT, supporting_evidence_ids=["d::0"]
    )
    assert sufficient.search_query is None
    with pytest.raises(ValidationError, match="search_query must be null"):
        SufficiencyDecision(
            question_id="q1",
            status="sufficient",
            search_query="unneeded query",
        )
    with pytest.raises(ValidationError, match="missing_information and search_query are required"):
        SufficiencyDecision(question_id="q1", status="need_more_evidence")


def test_answer_result_abstention_validation() -> None:
    answer = AnswerResult(question_id="q1", answer="yes", abstained=False)
    abstention = AnswerResult(
        question_id="q1",
        answer=None,
        abstained=True,
        abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
    )
    assert answer.answer == "yes"
    assert abstention.answer is None
    with pytest.raises(ValidationError, match="abstention_reason is required"):
        AnswerResult(question_id="q1", answer=None, abstained=True)
    with pytest.raises(ValidationError, match="answer must be null"):
        AnswerResult(
            question_id="q1",
            answer="guess",
            abstained=True,
            abstention_reason="insufficient_evidence",
        )


def test_fixture_experiment_validates_and_demonstrates_recovery() -> None:
    line = (FIXTURES / "example_experiment.jsonl").read_text(encoding="utf-8").strip()
    record = ExperimentRecord.model_validate_json(line)
    assert record.gold_sufficient_initially is False
    assert record.gold_sufficient_finally is True
    assert record.initial_evidence.evidence_ids == ["lost-gravity::0"]
    assert record.final_evidence_ids == ["lost-gravity::0", "mack-rides::0"]
    assert record.final_answer.answer == "Germany"
    assert record.rounds[0].sufficiency_decision.status == SufficiencyStatus.NEED_MORE_EVIDENCE
    assert record.rounds[1].retrieval.evidence[0].evidence_id == "mack-rides::0"


def test_stable_id_helpers() -> None:
    assert make_evidence_id("mack-rides", 0) == "mack-rides::0"
    assert (
        make_experiment_id("q1", EvidenceCondition.COMPLETE, SystemVariant.STANDARD_RAG, "run-2")
        == "q1::complete::standard_rag::run-2"
    )
    with pytest.raises(ValueError, match="chunk_index"):
        make_evidence_id("doc", -1)
