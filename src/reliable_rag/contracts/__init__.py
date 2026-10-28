"""Canonical data contracts and module interfaces."""

from reliable_rag.contracts.enums import (
    AbstentionReason,
    DatasetSplit,
    EvidenceCondition,
    RetrievalStage,
    SufficiencyStatus,
    SystemVariant,
)
from reliable_rag.contracts.ids import make_evidence_id, make_experiment_id
from reliable_rag.contracts.records import (
    AdaptiveRoundRecord,
    AnswerResult,
    EvidenceChunk,
    EvidenceSet,
    ExperimentRecord,
    QuestionRecord,
    RetrievalRequest,
    RetrievalResult,
    SufficiencyDecision,
)

__all__ = [
    "AbstentionReason",
    "AdaptiveRoundRecord",
    "AnswerResult",
    "DatasetSplit",
    "EvidenceChunk",
    "EvidenceCondition",
    "EvidenceSet",
    "ExperimentRecord",
    "QuestionRecord",
    "RetrievalRequest",
    "RetrievalResult",
    "RetrievalStage",
    "SufficiencyDecision",
    "SufficiencyStatus",
    "SystemVariant",
    "make_evidence_id",
    "make_experiment_id",
]
