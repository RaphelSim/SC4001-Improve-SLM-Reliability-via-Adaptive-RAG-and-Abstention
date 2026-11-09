"""Evidence-only sufficiency assessment using the shared structured decision contract."""

from __future__ import annotations

import json
import logging

from reliable_rag.config.settings import Settings
from reliable_rag.contracts.enums import SufficiencyStatus
from reliable_rag.contracts.interfaces import SufficiencyAssessor
from reliable_rag.contracts.records import (
    EvidenceSet,
    ModelEvidence,
    QuestionRecord,
    SufficiencyDecision,
)
from reliable_rag.model.errors import ModelOutputError
from reliable_rag.model.slm import SLMClient
from reliable_rag.model.structured_output import (
    StructuredOutputValidationError,
    build_retry_prompt,
    load_prompt_template,
    parse_json_model,
    render_prompt,
)

LOGGER = logging.getLogger(__name__)
TASK_NAME = "sufficiency_assessment"
SUFFICIENCY_KEYS = {
    "question_id",
    "status",
    "missing_information",
    "search_query",
    "supporting_evidence_ids",
}
V2_KEYS = SUFFICIENCY_KEYS | {
    "required_information",
    "established_facts",
    "missing_information_items",
}


def parse_sufficiency_response(
    raw_output: str,
    *,
    question: QuestionRecord,
    evidence: EvidenceSet | ModelEvidence,
    v2: bool = False,
) -> SufficiencyDecision:
    """Parse a strict decision and ensure it only cites evidence supplied to the SLM."""
    decision = parse_json_model(
        raw_output,
        SufficiencyDecision,
        required_keys=V2_KEYS if v2 else SUFFICIENCY_KEYS,
    )
    if decision.question_id != question.question_id:
        raise StructuredOutputValidationError(
            f"question_id mismatch: expected {question.question_id!r}"
        )
    if len(decision.supporting_evidence_ids) != len(set(decision.supporting_evidence_ids)):
        raise StructuredOutputValidationError("supporting_evidence_ids must not contain duplicates")
    unknown_ids = set(decision.supporting_evidence_ids) - set(evidence.evidence_ids)
    if unknown_ids:
        raise StructuredOutputValidationError(
            f"supporting_evidence_ids contains IDs absent from supplied evidence: "
            f"{', '.join(sorted(unknown_ids))}"
        )
    if decision.status == SufficiencyStatus.SUFFICIENT and not decision.supporting_evidence_ids:
        raise StructuredOutputValidationError(
            "a sufficient decision must identify at least one supporting evidence ID"
        )
    if v2:
        if not decision.required_information:
            raise StructuredOutputValidationError("required_information must not be empty")
        allowed = set(evidence.evidence_ids)
        for fact in decision.established_facts:
            if isinstance(fact, dict) and fact.get("evidence_ids") == []:
                raise StructuredOutputValidationError(
                    "unsupported facts must be removed from established_facts; keep their "
                    "requirement in missing_information_items and use need_more_evidence"
                )
            if (
                set(fact) != {"requirement", "fact", "evidence_ids"}
                or not fact["fact"]
                or not fact["evidence_ids"]
            ):
                raise StructuredOutputValidationError(
                    "each established_fact needs requirement, fact, and evidence_ids"
                )
            if fact["requirement"] not in decision.required_information:
                raise StructuredOutputValidationError(
                    "established_fact requirement must match required_information"
                )
            if set(fact["evidence_ids"]) - allowed:
                raise StructuredOutputValidationError(
                    "established_fact cites an ID absent from supplied evidence"
                )
        if decision.status == SufficiencyStatus.SUFFICIENT and (
            decision.missing_information_items
            or {fact["requirement"] for fact in decision.established_facts}
            != set(decision.required_information)
        ):
            raise StructuredOutputValidationError(
                "sufficient requires grounded facts for every requirement and no missing information"
            )
        if (
            decision.status == SufficiencyStatus.NEED_MORE_EVIDENCE
            and not decision.missing_information_items
        ):
            raise StructuredOutputValidationError(
                "need_more_evidence requires specific missing_information_items"
            )
    return decision


class SLMSufficiencyAssessor(SufficiencyAssessor):
    """Stateless adapter that asks an injected SLM whether supplied evidence is enough."""

    def __init__(self, client: SLMClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings
        self._prompt_template = load_prompt_template(settings.slm.sufficiency_prompt_path)
        self._retry_template = load_prompt_template(settings.slm.structured_retry_prompt_path)
        self.raw_outputs: list[str] = []

    @property
    def is_v2(self) -> bool:
        return self._settings.slm.prompt_version.startswith("retrieval-controller-v2")

    def assess(
        self,
        question: QuestionRecord,
        evidence: EvidenceSet | ModelEvidence,
        *,
        retrieval_history: list[dict] | None = None,
        retrieval_stagnated: bool = False,
        force_query_reformulation: bool = False,
    ) -> SufficiencyDecision:
        """Assess only the question and evidence text; never expose gold annotations in the prompt."""
        if question.question_id != evidence.question_id:
            raise ValueError("question and EvidenceSet question_id values must match")
        evidence_payload = [
            {"evidence_id": chunk.evidence_id, "title": chunk.title, "text": chunk.text}
            for chunk in evidence.evidence
        ]
        prompt = render_prompt(
            self._prompt_template,
            {
                "question_id_json": json.dumps(question.question_id),
                "question_json": json.dumps(question.question, ensure_ascii=False),
                "evidence_json": json.dumps(evidence_payload, ensure_ascii=False),
                "retrieval_history_json": json.dumps(retrieval_history or [], ensure_ascii=False),
                "retrieval_stagnated_json": json.dumps(retrieval_stagnated),
                "force_query_reformulation_json": json.dumps(force_query_reformulation),
            },
        )
        self.raw_outputs = []
        raw_output = self._client.generate(
            prompt,
            max_new_tokens=self._settings.slm.max_new_tokens,
        )
        self.raw_outputs.append(raw_output)
        retry_limit = self._settings.slm.format_retry_limit
        for attempt_index in range(retry_limit + 1):
            try:
                decision = parse_sufficiency_response(
                    raw_output,
                    question=question,
                    evidence=evidence,
                    v2=self.is_v2,
                )
            except StructuredOutputValidationError as exc:
                LOGGER.warning(
                    "Invalid model output question_id=%s task=%s attempt=%s "
                    "prompt_version=%s model_id=%s error=%s",
                    question.question_id,
                    TASK_NAME,
                    attempt_index + 1,
                    self._settings.slm.prompt_version,
                    self._settings.slm.model_id,
                    exc,
                )
                if attempt_index >= retry_limit:
                    raise ModelOutputError(
                        task=TASK_NAME,
                        question_id=question.question_id,
                        attempts=attempt_index + 1,
                        reason=str(exc),
                    ) from exc
                retry_prompt = build_retry_prompt(
                    self._retry_template,
                    task=TASK_NAME,
                    schema=SufficiencyDecision.model_json_schema(),
                    invalid_output=raw_output,
                    validation_error=str(exc),
                    original_prompt=prompt,
                )
                LOGGER.info(
                    "Retrying structured output question_id=%s task=%s next_attempt=%s "
                    "prompt_version=%s model_id=%s",
                    question.question_id,
                    TASK_NAME,
                    attempt_index + 2,
                    self._settings.slm.prompt_version,
                    self._settings.slm.model_id,
                )
                raw_output = self._client.generate(
                    retry_prompt,
                    max_new_tokens=self._settings.slm.max_new_tokens,
                )
                self.raw_outputs.append(raw_output)
                continue
            LOGGER.info(
                "Assessed evidence question_id=%s status=%s prompt_version=%s model_id=%s",
                question.question_id,
                decision.status.value,
                self._settings.slm.prompt_version,
                self._settings.slm.model_id,
            )
            if attempt_index:
                LOGGER.info(
                    "Retry succeeded question_id=%s task=%s attempt=%s",
                    question.question_id,
                    TASK_NAME,
                    attempt_index + 1,
                )
            return decision
        raise AssertionError("unreachable: retry loop returns or raises")
