"""Pydantic v2 models forming the project's canonical serialized contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from reliable_rag.contracts.enums import (
    AbstentionReason,
    DatasetSplit,
    EvidenceCondition,
    RetrievalStage,
    SufficiencyStatus,
    SystemVariant,
)


class ContractModel(BaseModel):
    """Base settings shared by persisted project records."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, use_enum_values=False)


def _require_nonempty(value: str) -> str:
    """Strip a string and reject empty or whitespace-only values."""
    value = value.strip()
    if not value:
        raise ValueError("must not be empty")
    return value


class EvidenceChunk(ContractModel):
    """A text unit with stable source identity and optional retrieval annotations."""

    evidence_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    title: str | None = None
    text: str = Field(min_length=1)
    retrieval_score: float | None = None
    retrieval_rank: int | None = Field(default=None, ge=1)
    retrieval_stage: RetrievalStage | None = None

    _clean_ids = field_validator("evidence_id", "document_id")(_require_nonempty)

    @field_validator("text")
    @classmethod
    def text_must_contain_content(cls, value: str) -> str:
        return _require_nonempty(value)


class QuestionRecord(ContractModel):
    """A question and its gold answer/support annotations for one split."""

    question_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    gold_answer: str = Field(min_length=1)
    split: DatasetSplit
    gold_support_ids: list[str] = Field(min_length=1)
    candidate_evidence_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    _clean_ids = field_validator("question_id")(_require_nonempty)

    @field_validator("question", "gold_answer")
    @classmethod
    def text_must_contain_content(cls, value: str) -> str:
        return _require_nonempty(value)

    @field_validator("gold_support_ids", "candidate_evidence_ids")
    @classmethod
    def ids_must_be_unique_and_nonempty(cls, values: list[str]) -> list[str]:
        cleaned = [_require_nonempty(value) for value in values]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("IDs must be unique")
        return cleaned


class EvidenceSet(ContractModel):
    """Evidence visible to the SLM at a specific point in the pipeline."""

    question_id: str = Field(min_length=1)
    condition: EvidenceCondition
    evidence: list[EvidenceChunk] = Field(default_factory=list)
    gold_support_ids: list[str] = Field(min_length=1)

    @field_validator("question_id")
    @classmethod
    def question_id_must_be_nonempty(cls, value: str) -> str:
        return _require_nonempty(value)

    @field_validator("gold_support_ids")
    @classmethod
    def support_ids_must_be_unique_and_nonempty(cls, values: list[str]) -> list[str]:
        cleaned = [_require_nonempty(value) for value in values]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("gold_support_ids must be unique")
        return cleaned

    @model_validator(mode="after")
    def validate_evidence_state(self) -> EvidenceSet:
        evidence_ids = [chunk.evidence_id for chunk in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence must not contain duplicate evidence_id values")
        supports = set(self.gold_support_ids)
        present_supports = supports.intersection(evidence_ids)
        if self.condition == EvidenceCondition.COMPLETE and present_supports != supports:
            raise ValueError("complete evidence must contain every gold_support_id")
        if self.condition == EvidenceCondition.PARTIALLY_INCOMPLETE and not (
            present_supports and len(present_supports) < len(supports)
        ):
            raise ValueError(
                "partially_incomplete evidence must contain some, but not all, gold supports"
            )
        if self.condition == EvidenceCondition.FULLY_INCOMPLETE and present_supports:
            raise ValueError("fully_incomplete evidence must contain no gold support IDs")
        return self

    @property
    def evidence_ids(self) -> list[str]:
        """Return the evidence identifiers in their presented order."""
        return [chunk.evidence_id for chunk in self.evidence]

    @property
    def gold_sufficient(self) -> bool:
        """Whether controlled gold support is complete in this evidence state."""
        return set(self.gold_support_ids).issubset(self.evidence_ids)


class RetrievalRequest(ContractModel):
    """One retriever invocation; one request corresponds to one retrieval round."""

    question_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    stage: RetrievalStage
    top_k: int = Field(ge=1)
    exclude_evidence_ids: list[str] = Field(default_factory=list)

    @field_validator("question_id", "query")
    @classmethod
    def required_text_must_contain_content(cls, value: str) -> str:
        return _require_nonempty(value)


class RetrievalResult(ContractModel):
    """Validated output from a retriever for one request."""

    question_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    stage: RetrievalStage
    top_k: int = Field(ge=1)
    evidence: list[EvidenceChunk] = Field(default_factory=list)

    @field_validator("question_id", "query")
    @classmethod
    def required_text_must_contain_content(cls, value: str) -> str:
        return _require_nonempty(value)

    @model_validator(mode="after")
    def evidence_ids_must_be_unique(self) -> RetrievalResult:
        ids = [chunk.evidence_id for chunk in self.evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("retrieval result must not contain duplicate evidence IDs")
        if len(ids) > self.top_k:
            raise ValueError("retrieval result cannot contain more evidence than top_k")
        return self


class SufficiencyDecision(ContractModel):
    """Concise model assessment; this stores a decision, not hidden reasoning."""

    question_id: str = Field(min_length=1)
    status: SufficiencyStatus
    missing_information: str | None = None
    search_query: str | None = None
    supporting_evidence_ids: list[str] = Field(default_factory=list)

    @field_validator("question_id")
    @classmethod
    def question_id_must_be_nonempty(cls, value: str) -> str:
        return _require_nonempty(value)

    @field_validator("missing_information", "search_query")
    @classmethod
    def optional_text_must_be_nonempty_if_present(cls, value: str | None) -> str | None:
        return None if value is None else _require_nonempty(value)

    @model_validator(mode="after")
    def validate_status_fields(self) -> SufficiencyDecision:
        if self.status == SufficiencyStatus.SUFFICIENT:
            if self.search_query is not None:
                raise ValueError("search_query must be null when status is sufficient")
        elif not self.missing_information or not self.search_query:
            raise ValueError(
                "missing_information and search_query are required when more evidence is needed"
            )
        return self


class AnswerResult(ContractModel):
    """A substantive answer or a typed abstention with evidence provenance."""

    question_id: str = Field(min_length=1)
    answer: str | None = None
    abstained: bool
    abstention_reason: AbstentionReason | None = None
    used_evidence_ids: list[str] = Field(default_factory=list)

    @field_validator("question_id")
    @classmethod
    def question_id_must_be_nonempty(cls, value: str) -> str:
        return _require_nonempty(value)

    @field_validator("answer")
    @classmethod
    def answer_must_be_nonempty_if_present(cls, value: str | None) -> str | None:
        return None if value is None else _require_nonempty(value)

    @model_validator(mode="after")
    def validate_abstention_fields(self) -> AnswerResult:
        if self.abstained:
            if self.answer is not None:
                raise ValueError("answer must be null for abstentions")
            if self.abstention_reason is None:
                raise ValueError("abstention_reason is required when abstained is true")
        elif self.answer is None:
            raise ValueError("answer must be non-empty when abstained is false")
        elif self.abstention_reason is not None:
            raise ValueError("abstention_reason must be null when abstained is false")
        return self


class AdaptiveRoundRecord(ContractModel):
    """One retrieval and sufficiency decision within the bounded experiment flow."""

    round_index: int = Field(ge=0, le=1)
    retrieval: RetrievalResult
    sufficiency_decision: SufficiencyDecision

    @model_validator(mode="after")
    def round_and_question_must_match(self) -> AdaptiveRoundRecord:
        if self.retrieval.question_id != self.sufficiency_decision.question_id:
            raise ValueError("retrieval and sufficiency decision question_id values must match")
        expected_stage = (
            RetrievalStage.INITIAL if self.round_index == 0 else RetrievalStage.FOLLOW_UP
        )
        if self.retrieval.stage != expected_stage:
            raise ValueError(
                f"round_index {self.round_index} requires stage {expected_stage.value!r}"
            )
        return self


class ExperimentRecord(ContractModel):
    """Canonical raw per-case result; aggregate metrics belong in separate outputs."""

    experiment_id: str = Field(min_length=1)
    question: QuestionRecord
    condition: EvidenceCondition
    system_variant: SystemVariant
    initial_evidence: EvidenceSet
    rounds: list[AdaptiveRoundRecord] = Field(min_length=1, max_length=2)
    final_evidence_ids: list[str]
    final_answer: AnswerResult
    gold_sufficient_initially: bool
    gold_sufficient_finally: bool
    model_predicted_sufficient_initially: bool | None = None
    model_predicted_sufficient_finally: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("experiment_id")
    @classmethod
    def experiment_id_must_be_nonempty(cls, value: str) -> str:
        return _require_nonempty(value)

    @field_validator("final_evidence_ids")
    @classmethod
    def final_ids_must_be_unique_and_nonempty(cls, values: list[str]) -> list[str]:
        cleaned = [_require_nonempty(value) for value in values]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("final_evidence_ids must be unique")
        return cleaned

    @model_validator(mode="after")
    def validate_record_consistency(self) -> ExperimentRecord:
        if self.initial_evidence.question_id != self.question.question_id:
            raise ValueError("initial_evidence question_id must match question.question_id")
        if self.initial_evidence.condition != self.condition:
            raise ValueError("initial_evidence condition must match experiment condition")
        if self.initial_evidence.gold_support_ids != self.question.gold_support_ids:
            raise ValueError("initial_evidence gold_support_ids must match the question")
        if self.final_answer.question_id != self.question.question_id:
            raise ValueError("final_answer question_id must match question.question_id")
        if [record.round_index for record in self.rounds] != list(range(len(self.rounds))):
            raise ValueError("rounds must be contiguous and start at round_index 0")
        if any(record.retrieval.question_id != self.question.question_id for record in self.rounds):
            raise ValueError("all retrieval rounds must match question.question_id")
        available_ids = set(self.initial_evidence.evidence_ids)
        for record in self.rounds:
            available_ids.update(chunk.evidence_id for chunk in record.retrieval.evidence)
        if not set(self.final_evidence_ids).issubset(available_ids):
            raise ValueError("final_evidence_ids must come from initial or retrieved evidence")
        if not set(self.final_answer.used_evidence_ids).issubset(self.final_evidence_ids):
            raise ValueError(
                "final_answer.used_evidence_ids must be included in final_evidence_ids"
            )
        support_ids = set(self.question.gold_support_ids)
        expected_initial = support_ids.issubset(self.initial_evidence.evidence_ids)
        expected_final = support_ids.issubset(self.final_evidence_ids)
        if self.gold_sufficient_initially != expected_initial:
            raise ValueError("gold_sufficient_initially must reflect initial gold support coverage")
        if self.gold_sufficient_finally != expected_final:
            raise ValueError("gold_sufficient_finally must reflect final gold support coverage")
        if self.model_predicted_sufficient_initially is not None:
            first = self.rounds[0].sufficiency_decision.status == SufficiencyStatus.SUFFICIENT
            if self.model_predicted_sufficient_initially != first:
                raise ValueError("initial model prediction must match round 0 sufficiency decision")
        if self.model_predicted_sufficient_finally is not None:
            last = self.rounds[-1].sufficiency_decision.status == SufficiencyStatus.SUFFICIENT
            if self.model_predicted_sufficient_finally != last:
                raise ValueError("final model prediction must match the last round decision")
        return self
