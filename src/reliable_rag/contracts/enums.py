"""Stable string values shared across configs, JSONL, and module contracts."""

from enum import StrEnum


class DatasetSplit(StrEnum):
    """A project-level data partition; test remains held out until settings freeze."""

    DEVELOPMENT = "development"
    VALIDATION = "validation"
    TEST = "test"


class EvidenceCondition(StrEnum):
    """Controlled evidence state presented at the start of an experiment case."""

    COMPLETE = "complete"
    PARTIALLY_INCOMPLETE = "partially_incomplete"
    FULLY_INCOMPLETE = "fully_incomplete"
    DISTRACTOR_HEAVY = "distractor_heavy"


class SystemVariant(StrEnum):
    """A named system configuration used for comparison."""

    STANDARD_RAG = "standard_rag"
    RAG_ABSTENTION = "rag_abstention"
    ADAPTIVE_RETRIEVAL = "adaptive_retrieval"
    ADAPTIVE_RETRIEVAL_ABSTENTION = "adaptive_retrieval_abstention"


class SufficiencyStatus(StrEnum):
    """The model's structured assessment of the current evidence."""

    SUFFICIENT = "sufficient"
    NEED_MORE_EVIDENCE = "need_more_evidence"


class AbstentionReason(StrEnum):
    """Reason recorded when the answering system intentionally abstains."""

    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    RETRIEVAL_FAILED = "retrieval_failed"
    INVALID_MODEL_OUTPUT = "invalid_model_output"
    MAX_RETRIEVAL_ROUNDS_REACHED = "max_retrieval_rounds_reached"


class RetrievalStage(StrEnum):
    """Retrieval round label. Initial retrieval is round zero."""

    INITIAL = "initial"
    FOLLOW_UP = "follow_up"
