"""Grounded answer generation and RAG + Abstention policy tests."""

import json

import pytest

from reliable_rag.contracts.enums import (
    AbstentionReason,
    SufficiencyStatus,
)
from reliable_rag.contracts.interfaces import AnswerGenerator as AnswerGeneratorProtocol
from reliable_rag.contracts.records import AnswerResult, SufficiencyDecision
from reliable_rag.model.abstention import RAGAbstentionResponder
from reliable_rag.model.answer import SLMAnswerGenerator, parse_answer_response
from reliable_rag.model.errors import ModelOutputError
from reliable_rag.model.structured_output import StructuredOutputValidationError
from tests.model.conftest import FakeSLMClient, load_case


class FakeAssessor:
    def __init__(self, decision=None, error: Exception | None = None) -> None:
        self.decision = decision
        self.error = error
        self.calls = 0

    def assess(self, question, evidence):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.decision


class FakeAnswerGenerator:
    def __init__(self, result: AnswerResult) -> None:
        self.result = result
        self.calls = 0

    def generate(self, question, evidence):
        self.calls += 1
        return self.result


def test_answer_generator_validates_answer_and_used_evidence(
    member2_cases: dict,  # type: ignore[type-arg]
    settings,
) -> None:
    case = next(item for item in member2_cases["cases"] if item["case_id"] == "multi_hop_complete")
    question, evidence = load_case(case)
    output = json.dumps(
        {
            "question_id": question.question_id,
            "answer": "Germany",
            "abstained": False,
            "abstention_reason": None,
            "used_evidence_ids": evidence.evidence_ids,
        }
    )
    client = FakeSLMClient(output)
    generator = SLMAnswerGenerator(client, settings)
    assert isinstance(generator, AnswerGeneratorProtocol)
    result = generator.generate(question, evidence)
    assert result.answer == "Germany"
    assert result.abstained is False
    assert set(result.used_evidence_ids).issubset(evidence.evidence_ids)


def test_answer_parser_rejects_empty_answers_and_unknown_evidence(
    member2_cases: dict,  # type: ignore[type-arg]
) -> None:
    case = next(item for item in member2_cases["cases"] if item["case_id"] == "multi_hop_complete")
    question, evidence = load_case(case)
    empty_answer = json.dumps(
        {
            "question_id": question.question_id,
            "answer": " ",
            "abstained": False,
            "abstention_reason": None,
            "used_evidence_ids": evidence.evidence_ids,
        }
    )
    with pytest.raises(StructuredOutputValidationError, match="must not be empty"):
        parse_answer_response(empty_answer, question=question, evidence=evidence)

    unknown_evidence = json.dumps(
        {
            "question_id": question.question_id,
            "answer": "Germany",
            "abstained": False,
            "abstention_reason": None,
            "used_evidence_ids": ["missing-id"],
        }
    )
    with pytest.raises(StructuredOutputValidationError, match="absent from supplied evidence"):
        parse_answer_response(unknown_evidence, question=question, evidence=evidence)

    wrong_abstention_reason = json.dumps(
        {
            "question_id": question.question_id,
            "answer": None,
            "abstained": True,
            "abstention_reason": "retrieval_failed",
            "used_evidence_ids": [],
        }
    )
    with pytest.raises(
        StructuredOutputValidationError, match="only abstain for insufficient_evidence"
    ):
        parse_answer_response(wrong_abstention_reason, question=question, evidence=evidence)


def test_answer_generator_surfaces_empty_output_as_invalid_after_retry(
    member2_cases: dict,  # type: ignore[type-arg]
    settings,
) -> None:
    case = next(item for item in member2_cases["cases"] if item["case_id"] == "multi_hop_complete")
    question, evidence = load_case(case)
    settings = settings.model_copy(
        update={"slm": settings.slm.model_copy(update={"format_retry_limit": 0})}
    )
    output = json.dumps(
        {
            "question_id": question.question_id,
            "answer": "",
            "abstained": False,
            "abstention_reason": None,
            "used_evidence_ids": evidence.evidence_ids,
        }
    )
    generator = SLMAnswerGenerator(FakeSLMClient(output), settings)
    with pytest.raises(ModelOutputError) as error:
        generator.generate(question, evidence)
    assert error.value.task == "final_answer"
    assert error.value.attempts == 1


def test_need_more_evidence_abstains_without_calling_answer_generator(
    member2_cases: dict,  # type: ignore[type-arg]
) -> None:
    case = next(item for item in member2_cases["cases"] if item["case_id"] == "multi_hop_partial")
    question, evidence = load_case(case)
    decision = SufficiencyDecision(
        question_id=question.question_id,
        status=SufficiencyStatus.NEED_MORE_EVIDENCE,
        missing_information="The country where Mack Rides is based.",
        search_query="Mack Rides headquarters country",
        supporting_evidence_ids=evidence.evidence_ids,
    )
    answer_generator = FakeAnswerGenerator(
        AnswerResult(question_id=question.question_id, answer="unused", abstained=False)
    )
    responder = RAGAbstentionResponder(FakeAssessor(decision), answer_generator)
    result = responder.answer(question, evidence)
    assert result.abstained is True
    assert result.answer is None
    assert result.abstention_reason == AbstentionReason.INSUFFICIENT_EVIDENCE
    assert result.used_evidence_ids == []
    assert answer_generator.calls == 0


def test_sufficient_evidence_calls_answer_generator(
    member2_cases: dict,  # type: ignore[type-arg]
) -> None:
    case = next(item for item in member2_cases["cases"] if item["case_id"] == "multi_hop_complete")
    question, evidence = load_case(case)
    decision = SufficiencyDecision(
        question_id=question.question_id,
        status=SufficiencyStatus.SUFFICIENT,
        supporting_evidence_ids=evidence.evidence_ids,
    )
    answer = AnswerResult(
        question_id=question.question_id,
        answer="Germany",
        abstained=False,
        used_evidence_ids=evidence.evidence_ids,
    )
    answer_generator = FakeAnswerGenerator(answer)
    responder = RAGAbstentionResponder(FakeAssessor(decision), answer_generator)
    assert responder.answer(question, evidence) == answer
    assert answer_generator.calls == 1


def test_invalid_assessor_output_becomes_typed_invalid_output_abstention(
    member2_cases: dict,  # type: ignore[type-arg]
) -> None:
    case = next(item for item in member2_cases["cases"] if item["case_id"] == "multi_hop_partial")
    question, evidence = load_case(case)
    error = ModelOutputError(
        task="sufficiency_assessment",
        question_id=question.question_id,
        attempts=2,
        reason="invalid JSON",
    )
    answer_generator = FakeAnswerGenerator(
        AnswerResult(question_id=question.question_id, answer="unused", abstained=False)
    )
    responder = RAGAbstentionResponder(FakeAssessor(error=error), answer_generator)
    result = responder.answer(question, evidence)
    assert result.abstained is True
    assert result.answer is None
    assert result.abstention_reason == AbstentionReason.INVALID_MODEL_OUTPUT
    assert answer_generator.calls == 0
