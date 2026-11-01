"""Shared synthetic cases and fake SLM client for Member 2 tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from reliable_rag.config import Settings, load_settings
from reliable_rag.contracts.enums import SufficiencyStatus
from reliable_rag.contracts.records import EvidenceSet, QuestionRecord

FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "member2_sufficiency_cases.json"
ROOT = Path(__file__).parents[2]


class FakeSLMClient:
    """Queue deterministic fake outputs and retain prompts for contract assertions."""

    def __init__(self, *outputs: str) -> None:
        self.outputs = list(outputs)
        self.prompts: list[str] = []
        self.token_limits: list[int | None] = []

    def generate(self, prompt: str, *, max_new_tokens: int | None = None) -> str:
        self.prompts.append(prompt)
        self.token_limits.append(max_new_tokens)
        if not self.outputs:
            raise AssertionError("fake SLM has no queued response")
        return self.outputs.pop(0)


@pytest.fixture
def member2_cases() -> dict[str, Any]:
    """Load the explicitly synthetic sufficiency case collection."""
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def settings() -> Settings:
    """Load the development config, which selects a mock runtime and prompt v1."""
    return load_settings(ROOT / "configs" / "development.yaml", environ={})


def load_case(case_data: dict[str, Any]) -> tuple[QuestionRecord, EvidenceSet]:
    """Validate a fixture case using the shared contracts."""
    return (
        QuestionRecord.model_validate(case_data["question"]),
        EvidenceSet.model_validate(case_data["evidence"]),
    )


def decision_response(
    question: QuestionRecord,
    evidence: EvidenceSet,
    status: SufficiencyStatus,
) -> str:
    """Create a valid fake structured model response for one fixture case."""
    sufficient = status == SufficiencyStatus.SUFFICIENT
    return json.dumps(
        {
            "question_id": question.question_id,
            "status": status.value,
            "missing_information": None if sufficient else "The missing supporting fact.",
            "search_query": None
            if sufficient
            else "Targeted query for the missing supporting fact",
            "supporting_evidence_ids": evidence.evidence_ids,
        }
    )
