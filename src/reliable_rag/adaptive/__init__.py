"""Adaptive follow-up retrieval with optional abstention."""

from reliable_rag.adaptive.controller import (
    AdaptiveCaseResult,
    AdaptiveRetrievalController,
    label_evidence,
    merge_evidence,
)

__all__ = [
    "AdaptiveCaseResult",
    "AdaptiveRetrievalController",
    "label_evidence",
    "merge_evidence",
]
