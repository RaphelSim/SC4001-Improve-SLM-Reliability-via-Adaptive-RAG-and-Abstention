"""Dense retrieval and structured sufficiency/answer integration tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tests.model.conftest import FakeSLMClient, load_case

from reliable_rag.contracts.enums import (
    AbstentionReason,
    RetrievalStage,
    SufficiencyStatus,
)
from reliable_rag.contracts.records import RetrievalRequest, RetrievalResult
from reliable_rag.model.abstention import RAGAbstentionResponder
from reliable_rag.model.answer import SLMAnswerGenerator
from reliable_rag.model.sufficiency import SLMSufficiencyAssessor
from reliable_rag.pipeline import run_initial_rag_abstention_case
from reliable_rag.retrieval import DenseRetriever

ROOT = Path(__file__).parents[2]
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "sufficiency_cases.json"


class ConstantEmbedder:
    """Tiny deterministic embedding stand-in; no sentence-transformer download needed."""

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))


def _model_output(question_id: str, status: SufficiencyStatus, evidence_ids: list[str]) -> str:
    sufficient = status == SufficiencyStatus.SUFFICIENT
    return json.dumps(
        {
            "question_id": question_id,
            "status": status.value,
            "missing_information": None if sufficient else "The country is not stated.",
            "search_query": None if sufficient else "Mack Rides headquarters country",
            "supporting_evidence_ids": evidence_ids,
        }
    )


def _answer_output(question_id: str, evidence_ids: list[str]) -> str:
    return json.dumps(
        {
            "question_id": question_id,
            "answer": "Germany",
            "abstained": False,
            "abstention_reason": None,
            "used_evidence_ids": evidence_ids,
        }
    )


def test_dense_retrieval_and_models_form_one_baseline_case() -> None:
    from reliable_rag.config import load_settings

    sufficiency_cases = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    settings = load_settings(ROOT / "configs" / "base.yaml", environ={})
    case_data = next(
        case for case in sufficiency_cases["cases"] if case["case_id"] == "multi_hop_complete"
    )
    question, complete_evidence = load_case(case_data)
    retriever = DenseRetriever(complete_evidence.evidence, ConstantEmbedder())
    retrieval = retriever.retrieve(
        RetrievalRequest(
            question_id=question.question_id,
            query=question.question,
            stage=RetrievalStage.INITIAL,
            top_k=len(question.candidate_evidence_ids),
        )
    )
    outputs = FakeSLMClient(
        _model_output(
            question.question_id,
            SufficiencyStatus.SUFFICIENT,
            [chunk.evidence_id for chunk in retrieval.evidence],
        ),
        _answer_output(
            question.question_id,
            [chunk.evidence_id for chunk in retrieval.evidence],
        ),
    )
    responder = RAGAbstentionResponder(
        SLMSufficiencyAssessor(outputs, settings),
        SLMAnswerGenerator(outputs, settings),
    )

    result = run_initial_rag_abstention_case(
        question,
        retrieval,
        responder,
        run_id="integration-test",
    )

    assert result.evidence.gold_sufficient is True
    assert result.outcome.sufficiency_decision is not None
    assert result.outcome.sufficiency_decision.status == SufficiencyStatus.SUFFICIENT
    assert result.outcome.answer.answer == "Germany"
    assert result.experiment is not None
    assert result.experiment.rounds[0].retrieval == retrieval
    assert result.experiment.system_variant.value == "rag_abstention"
    assert result.experiment.gold_sufficient_finally is True
    assert len(outputs.prompts) == 2


def test_partial_retrieval_abstains_and_records_one_round() -> None:
    from reliable_rag.config import load_settings

    sufficiency_cases = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    settings = load_settings(ROOT / "configs" / "base.yaml", environ={})
    case_data = next(
        case for case in sufficiency_cases["cases"] if case["case_id"] == "multi_hop_partial"
    )
    question, partial_evidence = load_case(case_data)
    retrieval = RetrievalResult(
        question_id=question.question_id,
        query=question.question,
        stage=RetrievalStage.INITIAL,
        top_k=3,
        evidence=partial_evidence.evidence,
    )
    outputs = FakeSLMClient(
        _model_output(
            question.question_id,
            SufficiencyStatus.NEED_MORE_EVIDENCE,
            partial_evidence.evidence_ids,
        )
    )
    responder = RAGAbstentionResponder(
        SLMSufficiencyAssessor(outputs, settings),
        SLMAnswerGenerator(outputs, settings),
    )

    result = run_initial_rag_abstention_case(
        question,
        retrieval,
        responder,
        run_id="integration-test",
    )

    assert result.evidence.gold_sufficient is False
    assert result.outcome.answer.abstained is True
    assert result.outcome.answer.abstention_reason == AbstentionReason.INSUFFICIENT_EVIDENCE
    assert result.experiment is not None
    assert len(result.experiment.rounds) == 1
    assert result.experiment.model_predicted_sufficient_finally is False
    assert len(outputs.prompts) == 1  # The answer generator is gated off.


def test_invalid_sufficiency_output_returns_abstention_without_experiment_record() -> None:
    from reliable_rag.config import load_settings

    sufficiency_cases = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    settings = load_settings(ROOT / "configs" / "base.yaml", environ={})
    case_data = next(
        case for case in sufficiency_cases["cases"] if case["case_id"] == "multi_hop_partial"
    )
    question, partial_evidence = load_case(case_data)
    retrieval = RetrievalResult(
        question_id=question.question_id,
        query=question.question,
        stage=RetrievalStage.INITIAL,
        top_k=3,
        evidence=partial_evidence.evidence,
    )
    outputs = FakeSLMClient("not json", "still not json")
    responder = RAGAbstentionResponder(
        SLMSufficiencyAssessor(outputs, settings),
        SLMAnswerGenerator(outputs, settings),
    )

    result = run_initial_rag_abstention_case(
        question,
        retrieval,
        responder,
        run_id="integration-test",
    )

    assert result.experiment is None
    assert result.outcome.sufficiency_decision is None
    assert result.outcome.error is not None
    assert result.outcome.answer.abstention_reason == AbstentionReason.INVALID_MODEL_OUTPUT
    assert len(outputs.prompts) == 2
