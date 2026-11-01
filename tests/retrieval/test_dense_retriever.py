"""Tests for deterministic dense retrieval."""

import numpy as np
import pytest

from reliable_rag.contracts.enums import RetrievalStage
from reliable_rag.contracts.records import EvidenceChunk, RetrievalRequest
from reliable_rag.retrieval import DenseRetriever


class FakeEmbedder:
    vectors = {
        "alpha": [1.0, 0.0],
        "beta": [0.8, 0.2],
        "gamma": [0.0, 1.0],
        "query": [1.0, 0.0],
    }

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


def test_dense_retriever_ranks_excludes_and_stamps_results() -> None:
    evidence = [
        EvidenceChunk(evidence_id="b::0", document_id="b", text="beta"),
        EvidenceChunk(evidence_id="a::0", document_id="a", text="alpha"),
        EvidenceChunk(evidence_id="g::0", document_id="g", text="gamma"),
    ]
    retriever = DenseRetriever(evidence, FakeEmbedder())
    result = retriever.retrieve(
        RetrievalRequest(
            question_id="q1",
            query="query",
            stage=RetrievalStage.FOLLOW_UP,
            top_k=2,
            exclude_evidence_ids=["a::0"],
        )
    )

    assert [chunk.evidence_id for chunk in result.evidence] == ["b::0", "g::0"]
    assert result.evidence[0].retrieval_rank == 1
    assert result.evidence[0].retrieval_stage == RetrievalStage.FOLLOW_UP
    assert result.evidence[0].retrieval_score == pytest.approx(0.9701425)
