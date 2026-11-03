"""Evidence-only sufficiency assessment using the shared structured decision contract."""

from __future__ import annotations

import json
import logging

from reliable_rag.config.settings import Settings
from reliable_rag.contracts.enums import SufficiencyStatus
from reliable_rag.contracts.interfaces import SufficiencyAssessor
from reliable_rag.contracts.records import EvidenceSet, QuestionRecord, SufficiencyDecision
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


def parse_sufficiency_response(
    raw_output: str,
    *,
    question: QuestionRecord,
    evidence: EvidenceSet,
) -> SufficiencyDecision:
    """Parse a strict decision and ensure it only cites evidence supplied to the SLM."""
    decision = parse_json_model(
        raw_output,
        SufficiencyDecision,
        required_keys=SUFFICIENCY_KEYS,
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
    return decision


class SLMSufficiencyAssessor(SufficiencyAssessor):
    """Stateless adapter that asks an injected SLM whether supplied evidence is enough."""

    def __init__(self, client: SLMClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings
        self._prompt_template = load_prompt_template(settings.slm.sufficiency_prompt_path)
        self._retry_template = load_prompt_template(settings.slm.structured_retry_prompt_path)

    def assess(self, question: QuestionRecord, evidence: EvidenceSet) -> SufficiencyDecision:
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
            },
        )
        raw_output = self._client.generate(
            prompt,
            max_new_tokens=self._settings.slm.max_new_tokens,
        )
        retry_limit = self._settings.slm.format_retry_limit
        for attempt_index in range(retry_limit + 1):
            try:
                decision = parse_sufficiency_response(
                    raw_output,
                    question=question,
                    evidence=evidence,
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
