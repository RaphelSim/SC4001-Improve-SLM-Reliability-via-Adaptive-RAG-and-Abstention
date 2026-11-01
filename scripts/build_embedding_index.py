#!/usr/bin/env python3
"""Build a persistent embedding cache for a prepared evidence JSONL file."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from reliable_rag.contracts.records import EvidenceChunk
from reliable_rag.io.jsonl import read_jsonl
from reliable_rag.retrieval import SentenceTransformerEmbedder, save_embedding_cache


def _corpus_fingerprint(evidence: list[EvidenceChunk]) -> str:
    digest = hashlib.sha256()
    for chunk in evidence:
        digest.update(chunk.evidence_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk.text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> int:
    """Encode all evidence paragraphs in batches and persist provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--dataset-revision", default="unspecified")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be at least 1")

    evidence = read_jsonl(args.evidence, EvidenceChunk)
    embedder = SentenceTransformerEmbedder(args.model)
    if not evidence:
        raise ValueError("evidence corpus must contain at least one chunk")

    first_batch = embedder.encode([chunk.text for chunk in evidence[: args.batch_size]])
    with tempfile.TemporaryDirectory(prefix="reliable-rag-embeddings-") as temp_dir:
        memmap_path = Path(temp_dir) / "embeddings.dat"
        embeddings = np.memmap(
            memmap_path,
            dtype=np.float32,
            mode="w+",
            shape=(len(evidence), first_batch.shape[1]),
        )
        embeddings[: len(first_batch)] = first_batch
        encoded = len(first_batch)
        print(f"Encoded {encoded}/{len(evidence)} evidence chunks", flush=True)
        for start in range(encoded, len(evidence), args.batch_size):
            batch = evidence[start : start + args.batch_size]
            embeddings[start : start + len(batch)] = embedder.encode(
                [chunk.text for chunk in batch]
            )
            encoded += len(batch)
            if encoded % (args.batch_size * 10) == 0 or encoded == len(evidence):
                print(f"Encoded {encoded}/{len(evidence)} evidence chunks", flush=True)
        embeddings.flush()
        metadata = {
            "embedding_model": args.model,
            "dataset_revision": args.dataset_revision,
            "evidence_count": str(len(evidence)),
            "embedding_dimension": str(embeddings.shape[1]),
            "corpus_fingerprint": _corpus_fingerprint(evidence),
        }
        save_embedding_cache(
            args.output,
            evidence_ids=[chunk.evidence_id for chunk in evidence],
            embeddings=embeddings,
            metadata=metadata,
        )
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
