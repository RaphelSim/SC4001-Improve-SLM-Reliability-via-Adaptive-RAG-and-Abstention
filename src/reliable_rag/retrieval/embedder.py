"""Embedding adapters and internal NumPy embedding-cache helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    """Encode text into a two-dimensional floating-point matrix."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        ...


class SentenceTransformerEmbedder:
    """Lazy adapter for the configured sentence-transformers model."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        vectors = self._model.encode(
            list(texts), convert_to_numpy=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype=np.float32)


def save_embedding_cache(
    path: str | Path,
    *,
    evidence_ids: Sequence[str],
    embeddings: np.ndarray,
    metadata: dict[str, str],
) -> None:
    """Persist internal embeddings and provenance; do not use for contract JSONL."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, evidence_ids=np.asarray(evidence_ids), embeddings=embeddings)
    destination.with_suffix(destination.suffix + ".json").write_text(
        json.dumps(metadata, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_embedding_cache(path: str | Path) -> tuple[list[str], np.ndarray, dict[str, str]]:
    """Load an embedding cache and its provenance metadata."""
    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        evidence_ids = [str(value) for value in data["evidence_ids"].tolist()]
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    metadata = json.loads(source.with_suffix(source.suffix + ".json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Embedding cache metadata for {source} must be an object")
    return evidence_ids, embeddings, metadata
