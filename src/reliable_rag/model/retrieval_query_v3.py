"""Compact, evidence-only V3 retrieval controller."""

from __future__ import annotations

import json

from pydantic import field_validator

from reliable_rag.config.settings import Settings
from reliable_rag.contracts.records import ContractModel, ModelEvidence, QuestionRecord
from reliable_rag.model.errors import ModelOutputError
from reliable_rag.model.slm import SLMClient
from reliable_rag.model.structured_output import (
    StructuredOutputValidationError,
    build_retry_prompt,
    load_prompt_template,
    parse_json_model,
    render_prompt,
)


class V3QueryDecision(ContractModel):
    """Null stops; a nonempty query requests another retrieval round."""

    search_query: str | None

    @field_validator("search_query")
    @classmethod
    def nonempty_query(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("search_query must be nonempty or null")
        return value


def parse_v3_response(raw_output: str) -> V3QueryDecision:
    return parse_json_model(raw_output, V3QueryDecision, required_keys={"search_query"})


class V3QueryAssessor:
    """Produce one query/null using only the original question and unique evidence."""

    def __init__(self, client: SLMClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings
        self._prompt_template = load_prompt_template(settings.slm.sufficiency_prompt_path)
        self._retry_template = load_prompt_template(settings.slm.structured_retry_prompt_path)
        self.raw_outputs: list[str] = []
        self.last_repair_calls = 0

    def assess(
        self,
        question: QuestionRecord,
        evidence: ModelEvidence,
        *,
        previous_search_queries: list[str],
        rejected_query: str | None = None,
    ) -> V3QueryDecision:
        if question.question_id != evidence.question_id:
            raise ValueError("question and model evidence question_id values must match")
        prompt = render_prompt(
            self._prompt_template,
            {
                "question_json": json.dumps(question.question, ensure_ascii=False),
                "evidence_json": json.dumps(
                    [
                        {"evidence_id": c.evidence_id, "title": c.title, "text": c.text}
                        for c in evidence.evidence
                    ],
                    ensure_ascii=False,
                ),
                "previous_search_queries_json": json.dumps(
                    previous_search_queries, ensure_ascii=False
                ),
                "rejected_query_json": json.dumps(rejected_query, ensure_ascii=False),
            },
        )
        self.raw_outputs = []
        self.last_repair_calls = 0
        raw = self._client.generate(prompt, max_new_tokens=self._settings.slm.max_new_tokens)
        self.raw_outputs.append(raw)
        for attempt in range(self._settings.slm.format_retry_limit + 1):
            try:
                return parse_v3_response(raw)
            except StructuredOutputValidationError as exc:
                if attempt >= self._settings.slm.format_retry_limit:
                    raise ModelOutputError(
                        task="v3_retrieval_query",
                        question_id=question.question_id,
                        attempts=attempt + 1,
                        reason=str(exc),
                    ) from exc
                repair_prompt = build_retry_prompt(
                    self._retry_template,
                    task="v3_retrieval_query",
                    schema=V3QueryDecision.model_json_schema(),
                    invalid_output=raw,
                    validation_error=str(exc),
                    original_prompt=prompt,
                )
                self.last_repair_calls += 1
                raw = self._client.generate(
                    repair_prompt, max_new_tokens=self._settings.slm.max_new_tokens
                )
                self.raw_outputs.append(raw)
        raise AssertionError("unreachable")
