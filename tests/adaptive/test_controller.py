"""Adaptive controller tests with injected fake retriever, assessor and answer generator."""

import json
from pathlib import Path

import pytest

from reliable_rag.adaptive import AdaptiveCaseResult, AdaptiveRetrievalController, merge_evidence
from reliable_rag.contracts.enums import (
    AbstentionReason,
    EvidenceCondition,
    RetrievalStage,
    SufficiencyStatus,
    SystemVariant,
)
from reliable_rag.contracts.records import (
    AnswerResult,
    EvidenceChunk,
    ExperimentRecord,
    QuestionRecord,
    RetrievalRequest,
    RetrievalResult,
    SufficiencyDecision,
)
from reliable_rag.model.errors import ModelOutputError

FIXTURES = Path(__file__).parents[1] / "fixtures"
QUESTION = QuestionRecord.model_validate(
    json.loads((FIXTURES / "example_question.json").read_text(encoding="utf-8"))
)
LOST_GRAVITY = EvidenceChunk(
    evidence_id="lost-gravity::0",
    document_id="lost-gravity",
    title="Lost Gravity",
    text="Lost Gravity was manufactured by Mack Rides.",
)
MACK_RIDES = EvidenceChunk(
    evidence_id="mack-rides::0",
    document_id="mack-rides",
    title="Mack Rides",
    text="Mack Rides is headquartered in Germany.",
)
DISTRACTOR = EvidenceChunk(
    evidence_id="park-distractor::0",
    document_id="park-distractor",
    title="Park Distractor",
    text="The park opened a new water ride.",
)


def initial(*chunks: EvidenceChunk) -> RetrievalResult:
    """Build a saved round-0 retrieval result containing the given chunks."""
    return RetrievalResult(
        question_id=QUESTION.question_id,
        query=QUESTION.question,
        stage=RetrievalStage.INITIAL,
        top_k=3,
        evidence=list(chunks),
    )


class FakeRetriever:
    """Return queued follow-up chunks, honouring exclusions, and record each request."""

    def __init__(self, *chunks: EvidenceChunk) -> None:
        self.chunks = list(chunks)
        self.requests: list[RetrievalRequest] = []

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        self.requests.append(request)
        found = [c for c in self.chunks if c.evidence_id not in request.exclude_evidence_ids]
        return RetrievalResult(
            question_id=request.question_id,
            query=request.query,
            stage=request.stage,
            top_k=request.top_k,
            evidence=found[: request.top_k],
        )


class GoldAwareAssessor:
    """An ideal assessor: sufficient exactly when all gold support is present."""

    def __init__(self, invalid_on_call: int | None = None) -> None:
        self.calls = 0
        # Simulate unrepaired model output on a chosen call (1 = round 0, 2 = round 1).
        self.invalid_on_call = invalid_on_call

    def assess(self, question, evidence):
        self.calls += 1
        if self.calls == self.invalid_on_call:
            raise ModelOutputError(
                task="sufficiency_assessment",
                question_id=question.question_id,
                attempts=2,
                reason="missing keys: search_query",
            )
        present = [i for i in evidence.evidence_ids if i in question.gold_support_ids]
        if evidence.gold_sufficient:
            return SufficiencyDecision(
                question_id=question.question_id,
                status=SufficiencyStatus.SUFFICIENT,
                supporting_evidence_ids=present,
            )
        return SufficiencyDecision(
            question_id=question.question_id,
            status=SufficiencyStatus.NEED_MORE_EVIDENCE,
            missing_information="Where Mack Rides is based.",
            search_query="Mack Rides headquarters country",
            supporting_evidence_ids=present,
        )


class FakeAnswerGenerator:
    """Answer Germany when gold support is visible, otherwise a wrong guess (USA)."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, question, evidence):
        self.calls += 1
        return AnswerResult(
            question_id=question.question_id,
            answer="Germany" if evidence.gold_sufficient else "USA",
            abstained=False,
            used_evidence_ids=evidence.evidence_ids[:1],
        )


def record_of(result: AdaptiveCaseResult) -> ExperimentRecord:
    """Return the case's record, failing the test if none was written."""
    assert result.experiment is not None
    return result.experiment


def make_controller(retriever, assessor=None, answer_generator=None, *, allow_abstention=True):
    return AdaptiveRetrievalController(
        retriever,
        assessor or GoldAwareAssessor(),
        answer_generator or FakeAnswerGenerator(),
        allow_abstention=allow_abstention,
    )


def test_complete_evidence_answers_without_follow_up() -> None:
    # Both gold paragraphs found in round 0, so no extra search is spent.
    retriever = FakeRetriever(DISTRACTOR)
    result = make_controller(retriever).run(
        QUESTION, initial(LOST_GRAVITY, MACK_RIDES), run_id="test"
    )

    record = record_of(result)
    assert len(record.rounds) == 1
    assert retriever.requests == []
    assert record.final_answer.answer == "Germany"
    assert record.system_variant == SystemVariant.ADAPTIVE_RETRIEVAL_ABSTENTION


def test_partial_evidence_is_recovered_in_round_one() -> None:
    # The core case: round 0 has only Lost Gravity; the follow-up finds Mack Rides.
    retriever = FakeRetriever(MACK_RIDES, DISTRACTOR)
    result = make_controller(retriever).run(QUESTION, initial(LOST_GRAVITY), run_id="test")

    request = retriever.requests[0]
    assert request.query == "Mack Rides headquarters country"
    assert request.stage == RetrievalStage.FOLLOW_UP
    assert request.exclude_evidence_ids == ["lost-gravity::0"]
    record = record_of(result)
    assert record.condition == EvidenceCondition.PARTIALLY_INCOMPLETE
    assert [r.round_index for r in record.rounds] == [0, 1]
    assert record.gold_sufficient_initially is False
    assert record.gold_sufficient_finally is True
    assert record.model_predicted_sufficient_finally is True
    assert record.final_answer.answer == "Germany"


def test_unrecovered_evidence_abstains_after_bounded_rounds() -> None:
    # The follow-up only finds a distractor, so the main system abstains.
    result = make_controller(FakeRetriever(DISTRACTOR)).run(
        QUESTION, initial(LOST_GRAVITY), run_id="test"
    )

    record = record_of(result)
    assert len(record.rounds) == 2
    assert record.final_answer.abstained is True
    assert record.final_answer.abstention_reason == AbstentionReason.MAX_RETRIEVAL_ROUNDS_REACHED


def test_adaptive_retrieval_variant_forces_an_answer() -> None:
    # Same situation without abstention: the answer is given but unsupported.
    answer_generator = FakeAnswerGenerator()
    result = make_controller(
        FakeRetriever(DISTRACTOR), answer_generator=answer_generator, allow_abstention=False
    ).run(QUESTION, initial(LOST_GRAVITY), run_id="test")

    record = record_of(result)
    assert record.system_variant == SystemVariant.ADAPTIVE_RETRIEVAL
    assert record.final_answer.answer == "USA"
    assert record.gold_sufficient_finally is False
    assert answer_generator.calls == 1


def test_empty_follow_up_skips_second_assessment() -> None:
    # Nothing new retrieved: no second model call, abstain as retrieval_failed.
    assessor = GoldAwareAssessor()
    result = make_controller(FakeRetriever(), assessor=assessor).run(
        QUESTION, initial(LOST_GRAVITY), run_id="test"
    )

    record = record_of(result)
    assert assessor.calls == 1
    assert len(record.rounds) == 1
    assert record.final_answer.abstention_reason == AbstentionReason.RETRIEVAL_FAILED
    assert record.metadata["followup_empty"] is True


def test_invalid_round_zero_assessment_writes_no_record() -> None:
    # Matches the baseline: an invalid assessment is an error, not a decision.
    result = make_controller(
        FakeRetriever(MACK_RIDES), assessor=GoldAwareAssessor(invalid_on_call=1)
    ).run(QUESTION, initial(LOST_GRAVITY), run_id="test")

    assert result.experiment is None
    assert result.final_answer.abstention_reason == AbstentionReason.INVALID_MODEL_OUTPUT
    assert result.errors[0].task == "sufficiency_assessment"


def test_invalid_round_one_assessment_keeps_round_zero_record() -> None:
    result = make_controller(
        FakeRetriever(MACK_RIDES), assessor=GoldAwareAssessor(invalid_on_call=2)
    ).run(QUESTION, initial(LOST_GRAVITY), run_id="test")

    record = record_of(result)
    assert len(record.rounds) == 1
    assert record.final_answer.abstention_reason == AbstentionReason.INVALID_MODEL_OUTPUT
    assert len(result.errors) == 1


def test_merge_keeps_order_and_drops_duplicates() -> None:
    merged = merge_evidence([LOST_GRAVITY], [LOST_GRAVITY, MACK_RIDES])

    assert [c.evidence_id for c in merged] == ["lost-gravity::0", "mack-rides::0"]


def test_rejects_non_initial_retrieval() -> None:
    followup = initial(LOST_GRAVITY).model_copy(update={"stage": RetrievalStage.FOLLOW_UP})

    with pytest.raises(ValueError, match="initial retrieval"):
        make_controller(FakeRetriever()).run(QUESTION, followup, run_id="test")
