#!/usr/bin/env python3
"""Run batch initial retrieval for a frozen question split."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from reliable_rag.contracts.enums import RetrievalStage
from reliable_rag.contracts.records import EvidenceChunk, QuestionRecord, RetrievalRequest
from reliable_rag.io.jsonl import read_jsonl, write_jsonl
from reliable_rag.retrieval import (
    DenseRetriever,
    SentenceTransformerEmbedder,
    load_embedding_cache,
)


def main() -> int:
    """Retrieve top-k evidence for every question using a persisted index."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.top_k < 1 or args.batch_size < 1:
        raise ValueError("top-k and batch size must be at least 1")

    questions = read_jsonl(args.questions, QuestionRecord)
    evidence = read_jsonl(args.evidence, EvidenceChunk)
    evidence_by_id = {chunk.evidence_id: chunk for chunk in evidence}
    cached_ids, embeddings, metadata = load_embedding_cache(args.embedding_cache)
    if cached_ids != [chunk.evidence_id for chunk in evidence]:
        raise ValueError("embedding cache evidence IDs do not match the evidence file")
    if int(metadata["evidence_count"]) != len(evidence):
        raise ValueError("embedding cache evidence count does not match the evidence file")

    embedder = SentenceTransformerEmbedder(args.model)
    retriever = DenseRetriever(evidence, embedder, embeddings=embeddings)
    results = []
    for start in range(0, len(questions), args.batch_size):
        batch = questions[start : start + args.batch_size]
        query_embeddings = embedder.encode([question.question for question in batch])
        for question, query_embedding in zip(batch, query_embeddings):
            missing = set(question.candidate_evidence_ids) - evidence_by_id.keys()
            if missing:
                raise ValueError(f"question {question.question_id} references unknown IDs")
            request = RetrievalRequest(
                question_id=question.question_id,
                query=question.question,
                stage=RetrievalStage.INITIAL,
                top_k=args.top_k,
            )
            results.append(
                retriever.retrieve_encoded(
                    request,
                    np.asarray(query_embedding),
                    candidate_evidence_ids=question.candidate_evidence_ids,
                )
            )
        if (start + len(batch)) % (args.batch_size * 100) == 0:
            print(f"Retrieved {start + len(batch)}/{len(questions)} questions")
    write_jsonl(args.output, results)
    print(f"Wrote {len(results)} retrieval results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
