"""Grounded final answer generation using the shared AnswerResult contract."""

from __future__ import annotations

import json
import logging

from reliable_rag.config.settings import Settings
from reliable_rag.contracts.enums import AbstentionReason
from reliable_rag.contracts.interfaces import AnswerGenerator as AnswerGeneratorProtocol
from reliable_rag.contracts.records import AnswerResult, EvidenceSet, QuestionRecord
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
TASK_NAME = "final_answer"
ANSWER_KEYS = {
    "question_id",
    "answer",
    "abstained",
    "abstention_reason",
    "used_evidence_ids",
}


def parse_answer_response(
    raw_output: str,
    *,
    question: QuestionRecord,
    evidence: EvidenceSet,
) -> AnswerResult:
    """Parse an AnswerResult and ensure all cited IDs came from the supplied EvidenceSet."""
    result = parse_json_model(raw_output, AnswerResult, required_keys=ANSWER_KEYS)
    if result.question_id != question.question_id:
        raise StructuredOutputValidationError(
            f"question_id mismatch: expected {question.question_id!r}"
        )
    if len(result.used_evidence_ids) != len(set(result.used_evidence_ids)):
        raise StructuredOutputValidationError("used_evidence_ids must not contain duplicates")
    unknown_ids = set(result.used_evidence_ids) - set(evidence.evidence_ids)
    if unknown_ids:
        raise StructuredOutputValidationError(
            f"used_evidence_ids contains IDs absent from supplied evidence: "
            f"{', '.join(sorted(unknown_ids))}"
        )
    if not result.abstained and not result.used_evidence_ids:
        raise StructuredOutputValidationError(
            "a substantive answer must identify at least one used evidence ID"
        )
    if result.abstained and result.abstention_reason != AbstentionReason.INSUFFICIENT_EVIDENCE:
        raise StructuredOutputValidationError(
            "the answer generator may only abstain for insufficient_evidence"
        )
    return result


class SLMAnswerGenerator(AnswerGeneratorProtocol):
    """Generate a concise grounded answer without invoking retrieval."""

    def __init__(self, client: SLMClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings
        self._prompt_template = load_prompt_template(settings.slm.final_answer_prompt_path)
        self._retry_template = load_prompt_template(settings.slm.structured_retry_prompt_path)

    def generate(self, question: QuestionRecord, evidence: EvidenceSet) -> AnswerResult:
        """Generate from supplied text and return a validated answer or typed abstention."""
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
                result = parse_answer_response(
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
                    schema=AnswerResult.model_json_schema(),
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
                "Generated answer question_id=%s abstained=%s prompt_version=%s model_id=%s",
                question.question_id,
                result.abstained,
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
            return result
        raise AssertionError("unreachable: retry loop returns or raises")
