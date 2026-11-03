"""Parser, prompt, and dependency-injected assessor tests."""

import json

import pytest

from reliable_rag.contracts.enums import SufficiencyStatus
from reliable_rag.contracts.interfaces import SufficiencyAssessor
from reliable_rag.model.errors import ModelOutputError
from reliable_rag.model.slm import SLMClient
from reliable_rag.model.structured_output import StructuredOutputValidationError
from reliable_rag.model.sufficiency import (
    SLMSufficiencyAssessor,
    parse_sufficiency_response,
)
from tests.model.conftest import FakeSLMClient, decision_response, load_case


def test_sufficiency_implementation_accepts_backend_neutral_fake(
    settings,
) -> None:
    client = FakeSLMClient("{}")
    assert isinstance(client, SLMClient)
    assert isinstance(SLMSufficiencyAssessor(client, settings), SufficiencyAssessor)


def test_valid_sufficient_and_insufficient_json_parse(
    sufficiency_cases: dict,  # type: ignore[type-arg]
) -> None:
    cases = {case["case_id"]: case for case in sufficiency_cases["cases"]}
    question, evidence = load_case(cases["multi_hop_complete"])
    sufficient = parse_sufficiency_response(
        decision_response(question, evidence, SufficiencyStatus.SUFFICIENT),
        question=question,
        evidence=evidence,
    )
    assert sufficient.status == SufficiencyStatus.SUFFICIENT
    assert sufficient.search_query is None
    assert set(sufficient.supporting_evidence_ids) == set(evidence.evidence_ids)

    question, evidence = load_case(cases["multi_hop_partial"])
    insufficient = parse_sufficiency_response(
        decision_response(question, evidence, SufficiencyStatus.NEED_MORE_EVIDENCE),
        question=question,
        evidence=evidence,
    )
    assert insufficient.status == SufficiencyStatus.NEED_MORE_EVIDENCE
    assert insufficient.missing_information
    assert insufficient.search_query


def test_parser_rejects_malformed_schema_invalid_and_wrong_question_id(
    sufficiency_cases: dict,  # type: ignore[type-arg]
) -> None:
    case = next(
        item for item in sufficiency_cases["cases"] if item["case_id"] == "multi_hop_partial"
    )
    question, evidence = load_case(case)
    with pytest.raises(StructuredOutputValidationError, match="invalid JSON"):
        parse_sufficiency_response("{bad json", question=question, evidence=evidence)

    schema_invalid = json.dumps(
        {
            "question_id": question.question_id,
            "status": "need_more_evidence",
            "missing_information": "A missing fact.",
            "search_query": None,
            "supporting_evidence_ids": evidence.evidence_ids,
        }
    )
    with pytest.raises(
        StructuredOutputValidationError,
        match="missing_information and search_query are required",
    ):
        parse_sufficiency_response(schema_invalid, question=question, evidence=evidence)

    wrong_question = json.loads(
        decision_response(question, evidence, SufficiencyStatus.NEED_MORE_EVIDENCE)
    )
    wrong_question["question_id"] = "different-question"
    with pytest.raises(StructuredOutputValidationError, match="question_id mismatch"):
        parse_sufficiency_response(json.dumps(wrong_question), question=question, evidence=evidence)


def test_assessor_runs_synthetic_cases_and_prompt_excludes_gold_annotations(
    sufficiency_cases: dict,  # type: ignore[type-arg]
    settings,
) -> None:
    for case in sufficiency_cases["cases"]:
        question, evidence = load_case(case)
        expected = SufficiencyStatus(case["expected_status"])
        client = FakeSLMClient(decision_response(question, evidence, expected))
        assessor = SLMSufficiencyAssessor(client, settings)
        decision = assessor.assess(question, evidence)
        assert decision.status == expected
        assert len(client.prompts) == 1
        prompt = client.prompts[0]
        assert "ONLY the supplied evidence" in prompt
        assert "pretrained or general factual knowledge is not evidence" in prompt
        assert '"gold_support_ids"' not in prompt
        if case["case_id"] == "parametric_knowledge_trap":
            assert "Paris" not in prompt


def test_same_assessor_can_reassess_updated_evidence_without_retrieval_state(
    sufficiency_cases: dict,  # type: ignore[type-arg]
    settings,
) -> None:
    case_by_id = {case["case_id"]: case for case in sufficiency_cases["cases"]}
    partial_question, partial_evidence = load_case(case_by_id["multi_hop_partial"])
    complete_question, complete_evidence = load_case(case_by_id["multi_hop_complete"])
    client = FakeSLMClient(
        decision_response(
            partial_question,
            partial_evidence,
            SufficiencyStatus.NEED_MORE_EVIDENCE,
        ),
        decision_response(
            complete_question,
            complete_evidence,
            SufficiencyStatus.SUFFICIENT,
        ),
    )
    assessor = SLMSufficiencyAssessor(client, settings)
    first = assessor.assess(partial_question, partial_evidence)
    second = assessor.assess(complete_question, complete_evidence)
    assert first.status == SufficiencyStatus.NEED_MORE_EVIDENCE
    assert second.status == SufficiencyStatus.SUFFICIENT
    assert len(client.prompts) == 2


def test_assessor_uses_one_constrained_retry_then_returns_decision(
    sufficiency_cases: dict,  # type: ignore[type-arg]
    settings,
) -> None:
    case = next(
        item for item in sufficiency_cases["cases"] if item["case_id"] == "multi_hop_partial"
    )
    question, evidence = load_case(case)
    client = FakeSLMClient(
        "not JSON",
        decision_response(question, evidence, SufficiencyStatus.NEED_MORE_EVIDENCE),
    )
    assessor = SLMSufficiencyAssessor(client, settings)
    decision = assessor.assess(question, evidence)
    assert decision.status == SufficiencyStatus.NEED_MORE_EVIDENCE
    assert len(client.prompts) == 2
    assert "repair JSON formatting only" in client.prompts[1]
    assert client.token_limits == [settings.slm.max_new_tokens] * 2


def test_assessor_surfaces_structured_failure_after_one_retry(
    sufficiency_cases: dict,  # type: ignore[type-arg]
    settings,
) -> None:
    case = next(
        item for item in sufficiency_cases["cases"] if item["case_id"] == "obviously_sufficient"
    )
    question, evidence = load_case(case)
    settings = settings.model_copy(
        update={"slm": settings.slm.model_copy(update={"format_retry_limit": 1})}
    )
    client = FakeSLMClient("{", "{still-not-json")
    assessor = SLMSufficiencyAssessor(client, settings)
    with pytest.raises(ModelOutputError) as error:
        assessor.assess(question, evidence)
    assert len(client.prompts) == 2
    assert error.value.attempts == 2
    assert error.value.as_dict()["failure_reason"] == "invalid_model_output"


def test_assessor_rejects_unknown_evidence_ids(
    sufficiency_cases: dict,  # type: ignore[type-arg]
) -> None:
    case = next(
        item for item in sufficiency_cases["cases"] if item["case_id"] == "obviously_sufficient"
    )
    question, evidence = load_case(case)
    output = json.loads(decision_response(question, evidence, SufficiencyStatus.SUFFICIENT))
    output["supporting_evidence_ids"] = ["not-present"]
    with pytest.raises(StructuredOutputValidationError, match="absent from supplied evidence"):
        parse_sufficiency_response(json.dumps(output), question=question, evidence=evidence)
