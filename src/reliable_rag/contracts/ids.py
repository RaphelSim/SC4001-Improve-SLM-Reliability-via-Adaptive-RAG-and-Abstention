"""Canonical stable identifier helpers for generated records."""

from reliable_rag.contracts.enums import EvidenceCondition, SystemVariant


def make_evidence_id(document_id: str, chunk_index: int) -> str:
    """Build a deterministic evidence identifier from stable source and chunk identity."""
    document_id = document_id.strip()
    if not document_id:
        raise ValueError("document_id must not be empty")
    if chunk_index < 0:
        raise ValueError("chunk_index must be zero or greater")
    return f"{document_id}::{chunk_index}"


def make_experiment_id(
    question_id: str,
    condition: EvidenceCondition,
    system_variant: SystemVariant,
    run_id: str,
) -> str:
    """Build a stable, readable experiment identifier from canonical labels."""
    parts = (question_id.strip(), run_id.strip())
    if any(not part for part in parts):
        raise ValueError("question_id and run_id must not be empty")
    return f"{parts[0]}::{condition.value}::{system_variant.value}::{parts[1]}"
