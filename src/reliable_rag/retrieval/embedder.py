"""Embedding adapters and internal NumPy embedding-cache helpers."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from reliable_rag.assets import resolve_model_path
from reliable_rag.contracts.records import EvidenceChunk


class Embedder(Protocol):
    """Encode text into a two-dimensional floating-point matrix."""

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Lazy adapter for the configured sentence-transformers model."""

    def __init__(
        self,
        model_name: str,
        *,
        revision: str | None = None,
        device: str = "cpu",
        cache_root: Path = Path("cache"),
        allow_model_download: bool = False,
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.cache_root = cache_root
        self.allow_model_download = allow_model_download
        self._model = None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            path = resolve_model_path(
                self.model_name,
                revision=self.revision,
                cache_root=self.cache_root,
                allow_download=self.allow_model_download,
            )
            self._model = SentenceTransformer(
                str(path), device=self.device, trust_remote_code=False
            )
        vectors = self._model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
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


def corpus_fingerprint(evidence: Sequence[EvidenceChunk]) -> str:
    """Hash ordered IDs and text so unchanged IDs cannot hide changed content."""
    digest = hashlib.sha256()
    for chunk in evidence:
        digest.update(chunk.evidence_id.encode())
        digest.update(b"\0")
        digest.update(chunk.text.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def validate_embedding_cache(
    evidence: Sequence[EvidenceChunk],
    evidence_ids: Sequence[str],
    embeddings: np.ndarray,
    metadata: dict[str, str],
    *,
    model_name: str,
    revision: str | None,
) -> None:
    """Reject stale or incompatible evidence/query embedding combinations."""
    expected = {
        "embedding_model": model_name,
        "embedding_revision": revision or "default",
        "corpus_fingerprint": corpus_fingerprint(evidence),
        "evidence_count": str(len(evidence)),
    }
    if metadata.get("encoder_preprocessing", "model-defaults-v1") != "model-defaults-v1":
        raise ValueError("embedding cache preprocessing mismatch")
    if (
        model_name == "sentence-transformers/all-MiniLM-L6-v2"
        and embeddings.ndim == 2
        and embeddings.shape[1] != 384
    ):
        raise ValueError("MiniLM embedding dimension must be 384")
    if list(evidence_ids) != [chunk.evidence_id for chunk in evidence]:
        raise ValueError("embedding cache evidence IDs/order do not match the corpus")
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"embedding cache {key} mismatch; rebuild the embedding index")
    if (
        embeddings.ndim != 2
        or embeddings.shape[0] != len(evidence)
        or embeddings.shape[1] < 1
        or metadata.get("embedding_dimension") != str(embeddings.shape[1])
        or not np.isfinite(embeddings).all()
        or np.any(np.linalg.norm(embeddings, axis=1) == 0)
    ):
        raise ValueError("embedding cache contains invalid vectors or dimensions")


def prepare_embedding_index(
    settings, evidence_path: Path, output: Path, *, batch_size: int = 64, force: bool = False
) -> bool:
    """Reuse only a validated corpus/model match; return True when rebuilding."""
    import logging
    import tempfile

    from reliable_rag.io.jsonl import read_jsonl

    logger = logging.getLogger(__name__)
    if (
        output.resolve() == evidence_path.resolve()
        or output.with_suffix(output.suffix + ".json").resolve() == evidence_path.resolve()
    ):
        raise ValueError("Index must not overwrite its evidence input")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    evidence = read_jsonl(evidence_path, EvidenceChunk)
    if not evidence:
        raise ValueError("Evidence corpus must not be empty")
    if not force:
        try:
            ids, vectors, metadata = load_embedding_cache(output)
            validate_embedding_cache(
                evidence,
                ids,
                vectors,
                metadata,
                model_name=settings.retrieval.embedding_model,
                revision=settings.retrieval.embedding_revision,
            )
            logger.info("Using cached embedding index: %s", output)
            return False
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            if output.exists():
                logger.info(
                    "Cached index does not match corpus/configuration. Rebuilding %s", output
                )
    embedder = SentenceTransformerEmbedder(
        settings.retrieval.embedding_model,
        revision=settings.retrieval.embedding_revision,
        device=settings.retrieval.embedding_device,
        cache_root=settings.cache_root,
    )
    first = embedder.encode([chunk.text for chunk in evidence[:batch_size]])
    with tempfile.TemporaryDirectory(prefix="reliable-rag-index-") as temporary:
        vectors = np.memmap(
            Path(temporary) / "vectors.dat",
            dtype=np.float32,
            mode="w+",
            shape=(len(evidence), first.shape[1]),
        )
        vectors[: len(first)] = first
        for start in range(len(first), len(evidence), batch_size):
            batch = evidence[start : start + batch_size]
            vectors[start : start + len(batch)] = embedder.encode([chunk.text for chunk in batch])
        vectors.flush()
        save_embedding_cache(
            output,
            evidence_ids=[chunk.evidence_id for chunk in evidence],
            embeddings=vectors,
            metadata={
                "embedding_model": settings.retrieval.embedding_model,
                "embedding_revision": settings.retrieval.embedding_revision or "default",
                "embedding_device": settings.retrieval.embedding_device,
                "encoder_preprocessing": "model-defaults-v1",
                "dataset_revision": settings.dataset.revision or "unspecified",
                "evidence_count": str(len(evidence)),
                "embedding_dimension": str(vectors.shape[1]),
                "corpus_fingerprint": corpus_fingerprint(evidence),
            },
        )
    logger.info("Built embedding index for %s paragraphs: %s", len(evidence), output)
    return True
