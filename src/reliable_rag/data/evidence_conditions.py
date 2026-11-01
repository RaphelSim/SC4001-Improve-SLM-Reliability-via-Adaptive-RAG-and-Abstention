"""Deterministic construction of controlled initial evidence states."""

from __future__ import annotations

import hashlib

from reliable_rag.contracts.enums import EvidenceCondition
from reliable_rag.contracts.records import EvidenceSet
from reliable_rag.data.hotpotqa import HotpotQAExample


def _selected_support(question_id: str, support_ids: list[str], seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{question_id}".encode("utf-8")).digest()
    return support_ids[int.from_bytes(digest[:8], "big") % len(support_ids)]


def construct_evidence_set(
    example: HotpotQAExample,
    condition: EvidenceCondition,
    *,
    seed: int = 42,
) -> EvidenceSet:
    """Construct one controlled evidence state without changing the retrieval corpus."""
    support_ids = example.question.gold_support_ids
    if condition == EvidenceCondition.COMPLETE:
        visible_ids = set(chunk.evidence_id for chunk in example.evidence)
    elif condition == EvidenceCondition.PARTIALLY_INCOMPLETE:
        removed = _selected_support(example.question.question_id, support_ids, seed)
        visible_ids = set(support_ids) - {removed}
        visible_ids.update(
            chunk.evidence_id
            for chunk in example.evidence
            if chunk.evidence_id not in support_ids
        )
    elif condition == EvidenceCondition.FULLY_INCOMPLETE:
        visible_ids = {
            chunk.evidence_id
            for chunk in example.evidence
            if chunk.evidence_id not in support_ids
        }
    else:
        raise ValueError("distractor_heavy construction is not enabled in the foundation")

    visible = tuple(chunk for chunk in example.evidence if chunk.evidence_id in visible_ids)
    return EvidenceSet(
        question_id=example.question.question_id,
        condition=condition,
        evidence=list(visible),
        gold_support_ids=support_ids,
    )
