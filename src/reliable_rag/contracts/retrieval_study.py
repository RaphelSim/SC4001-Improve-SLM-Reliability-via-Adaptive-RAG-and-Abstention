"""Question-level records for paragraph retrieval experiments."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from reliable_rag.contracts.enums import RetrievalStage, SufficiencyStatus
from reliable_rag.contracts.records import ContractModel, RetrievalResult


class StudyRound(ContractModel):
    round_number: int = Field(ge=1)
    retrieval: RetrievalResult
    new_unique_ids: list[str]
    duplicate_ids: list[str]
    model_status_after_round: SufficiencyStatus | None = None
    missing_information: str | None = None
    next_query: str | None = None
    raw_model_responses: list[str] = Field(default_factory=list)
    gold_complete_after_round: bool | None = None
    required_information: list[str] = Field(default_factory=list)
    established_facts: list[dict] = Field(default_factory=list)
    missing_information_items: list[str] = Field(default_factory=list)
    retrieval_stagnated: bool = False
    loop_guard_activated: bool = False
    ranked_seen_ids_skipped: list[str] = Field(default_factory=list)
    ranked_candidate_ids_considered: list[str] = Field(default_factory=list)
    repair_used: bool = False
    query_regeneration_used: bool = False
    stop_reason: str | None = None
    cumulative_raw_paragraphs: int = 0
    cumulative_unique_paragraph_ids: list[str] = Field(default_factory=list)
    cumulative_gold_paragraph_ids: list[str] = Field(default_factory=list)
    cumulative_gold_evidence_percentage: float = Field(default=0, ge=0, le=100)
    cumulative_unique_noise_percentage: float = Field(default=0, ge=0, le=100)


class RetrievalStudyRecord(ContractModel):
    """One baseline or adaptive observation, including raw and unique retrieval cost."""

    question_id: str
    question: str
    gold_paragraph_ids: list[str] = Field(min_length=1)
    candidate_paragraph_ids: list[str] = Field(min_length=1)
    system: Literal["baseline", "adaptive"]
    rounds: list[StudyRound] = Field(min_length=1)
    raw_retrieved_ids: list[str]
    unique_retrieved_ids: list[str]
    gold_retrieved_ids: list[str]
    noise_ids: list[str]
    gold_evidence_percentage: float = Field(ge=0, le=100)
    noise_percentage: float = Field(ge=0, le=100)
    raw_paragraphs_retrieved: int = Field(ge=0)
    unique_paragraphs_retrieved: int = Field(ge=0)
    duplicate_retrieval_count: int = Field(ge=0)
    retrieval_rounds: int = Field(ge=1)
    final_model_status: SufficiencyStatus | Literal["ERROR"] | None = None
    error_type: str | None = None
    error_message: str | None = None
    controller_version: str | None = None
    stop_reason: str | None = None
    controller_model_calls: int = Field(default=0, ge=0)
    repair_model_calls: int = Field(default=0, ge=0)
    query_regeneration_model_calls: int = Field(default=0, ge=0)
    total_reasoning_model_calls: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def counts_and_labels_agree(self) -> RetrievalStudyRecord:
        if self.retrieval_rounds != len(self.rounds):
            raise ValueError("retrieval_rounds must equal the number of rounds")
        if [r.round_number for r in self.rounds] != list(range(1, len(self.rounds) + 1)):
            raise ValueError("round numbers must start at one and be contiguous")
        raw = [c.evidence_id for r in self.rounds for c in r.retrieval.evidence]
        unique = list(dict.fromkeys(raw))
        seen: set[str] = set()
        gold = set(self.gold_paragraph_ids)
        for item in self.rounds:
            if item.retrieval.question_id != self.question_id:
                raise ValueError("retrieval round question ID mismatch")
            expected_stage = (
                RetrievalStage.INITIAL if item.round_number == 1 else RetrievalStage.FOLLOW_UP
            )
            if item.retrieval.stage != expected_stage:
                raise ValueError("retrieval stage does not match round number")
            ids = [chunk.evidence_id for chunk in item.retrieval.evidence]
            expected_new = [evidence_id for evidence_id in ids if evidence_id not in seen]
            expected_duplicate = [evidence_id for evidence_id in ids if evidence_id in seen]
            if item.new_unique_ids != expected_new or item.duplicate_ids != expected_duplicate:
                raise ValueError("round new/duplicate IDs do not match prior retrieval history")
            seen.update(ids)
            if item.gold_complete_after_round != gold.issubset(seen):
                raise ValueError("round gold completeness does not match frozen labels")
        if self.raw_retrieved_ids != raw or self.unique_retrieved_ids != unique:
            raise ValueError("raw and unique IDs must reflect the saved round history")
        if self.raw_paragraphs_retrieved != len(raw) or self.unique_paragraphs_retrieved != len(
            unique
        ):
            raise ValueError("retrieval counts must reflect saved IDs")
        if self.duplicate_retrieval_count != len(raw) - len(unique):
            raise ValueError("duplicate count must equal raw minus unique")
        if set(self.gold_retrieved_ids) != gold.intersection(unique):
            raise ValueError("gold retrieved IDs do not match frozen gold labels")
        if set(self.noise_ids) != set(unique) - gold:
            raise ValueError("noise IDs do not match unique non-gold IDs")
        if (
            abs(self.gold_evidence_percentage - 100 * len(self.gold_retrieved_ids) / len(gold))
            > 1e-8
        ):
            raise ValueError("gold evidence percentage is inconsistent")
        expected_noise = 100 * len(self.noise_ids) / len(unique) if unique else 0
        if abs(self.noise_percentage - expected_noise) > 1e-8:
            raise ValueError("noise percentage is inconsistent")
        if not set(unique).issubset(self.candidate_paragraph_ids):
            raise ValueError("retrieval escaped the question candidate corpus")
        if self.system == "baseline" and (
            len(self.rounds) != 1 or self.final_model_status is not None
        ):
            raise ValueError("baseline must have one retrieval and no model status")
        if self.system == "adaptive" and self.final_model_status is None:
            raise ValueError("adaptive final model status is required")
        if self.final_model_status == "ERROR" and not self.error_type:
            raise ValueError("ERROR requires an error type")
        if self.total_reasoning_model_calls != (
            self.controller_model_calls
            + self.repair_model_calls
            + self.query_regeneration_model_calls
        ):
            raise ValueError("reasoning-model call totals are inconsistent")
        if self.controller_version == "v3":
            if self.system != "adaptive" or self.stop_reason is None:
                raise ValueError("V3 requires adaptive system and a stop reason")
            if self.duplicate_retrieval_count or any(r.duplicate_ids for r in self.rounds):
                raise ValueError("V3 must return only unseen paragraphs")
        return self
