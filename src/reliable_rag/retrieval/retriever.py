"""Deterministic dense cosine retriever for a split-scoped evidence corpus."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from reliable_rag.contracts.records import EvidenceChunk, RetrievalRequest, RetrievalResult
from reliable_rag.retrieval.embedder import Embedder


class DenseRetriever:
    """Retrieve evidence chunks using cosine similarity and stable tie-breaking."""

    def __init__(
        self,
        evidence: Sequence[EvidenceChunk],
        embedder: Embedder,
        *,
        embeddings: np.ndarray | None = None,
    ) -> None:
        if not evidence:
            raise ValueError("evidence corpus must not be empty")
        ids = [chunk.evidence_id for chunk in evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence corpus must contain unique evidence IDs")
        if embeddings is None:
            embeddings = np.asarray(
                embedder.encode([chunk.text for chunk in evidence]), dtype=np.float32
            )
        else:
            embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(evidence):
            raise ValueError("embedder must return one vector per evidence chunk")
        norms = np.linalg.norm(embeddings, axis=1)
        if np.any(norms == 0):
            raise ValueError("evidence embeddings must not contain zero vectors")
        self._evidence = tuple(evidence)
        self._by_id = {chunk.evidence_id: chunk for chunk in evidence}
        self._embeddings = embeddings / norms[:, None]
        self._embedder = embedder

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """Return at most request.top_k ranked, non-excluded evidence chunks."""
        query_embedding = np.asarray(self._embedder.encode([request.query]), dtype=np.float32)
        return self.retrieve_encoded(request, query_embedding[0])

    def retrieve_encoded(
        self,
        request: RetrievalRequest,
        query_embedding: np.ndarray,
        *,
        candidate_evidence_ids: Sequence[str] | None = None,
    ) -> RetrievalResult:
        """Retrieve using a precomputed query vector and an optional candidate subset."""
        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        if query_embedding.shape != (1, self._embeddings.shape[1]):
            if query_embedding.shape != (self._embeddings.shape[1],):
                raise ValueError("query vector has an incompatible shape")
            query_embedding = query_embedding.reshape(1, -1)
        query_norm = np.linalg.norm(query_embedding[0])
        if query_norm == 0:
            raise ValueError("query embedding must not be a zero vector")
        candidate_ids = (
            set(candidate_evidence_ids) if candidate_evidence_ids is not None else None
        )
        scores = self._embeddings @ (query_embedding[0] / query_norm)
        excluded = set(request.exclude_evidence_ids)
        ranked = sorted(
            (
                (float(score), chunk.evidence_id, index)
                for index, (score, chunk) in enumerate(zip(scores, self._evidence))
                if chunk.evidence_id not in excluded
                and (candidate_ids is None or chunk.evidence_id in candidate_ids)
            ),
            key=lambda item: (-item[0], item[1]),
        )[: request.top_k]
        chunks = [
            self._by_id[evidence_id].model_copy(
                update={
                    "retrieval_score": score,
                    "retrieval_rank": rank,
                    "retrieval_stage": request.stage,
                }
            )
            for rank, (score, evidence_id, _) in enumerate(ranked, start=1)
        ]
        return RetrievalResult(
            question_id=request.question_id,
            query=request.query,
            stage=request.stage,
            top_k=request.top_k,
            evidence=chunks,
        )
