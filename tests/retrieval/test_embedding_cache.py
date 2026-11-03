"""Tests for internal embedding cache persistence."""

import numpy as np
import pytest

from reliable_rag.contracts.records import EvidenceChunk
from reliable_rag.retrieval import load_embedding_cache, save_embedding_cache
from reliable_rag.retrieval.embedder import corpus_fingerprint, validate_embedding_cache


def test_embedding_cache_round_trip(tmp_path) -> None:
    path = tmp_path / "embeddings.npz"
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    metadata = {"dataset_revision": "fixture-v1", "embedding_model": "test"}

    save_embedding_cache(
        path,
        evidence_ids=["a::0", "b::0"],
        embeddings=embeddings,
        metadata=metadata,
    )
    evidence_ids, loaded, loaded_metadata = load_embedding_cache(path)

    assert evidence_ids == ["a::0", "b::0"]
    assert np.array_equal(loaded, embeddings)
    assert loaded_metadata == metadata


@pytest.mark.parametrize("mismatch", ["text", "model", "revision", "vectors"])
def test_cache_validation_rejects_stale_content_and_incompatible_models(mismatch):
    evidence = [EvidenceChunk(evidence_id="a::0", document_id="a", text="original")]
    metadata = {
        "embedding_model": "test",
        "embedding_revision": "revision1",
        "evidence_count": "1",
        "embedding_dimension": "2",
        "corpus_fingerprint": corpus_fingerprint(evidence),
    }
    vectors = np.asarray([[1.0, 0.0]], dtype=np.float32)
    if mismatch == "text":
        evidence[0].text = "changed text with the same ID"
    elif mismatch == "model":
        metadata["embedding_model"] = "another-model"
    elif mismatch == "revision":
        metadata["embedding_revision"] = "another-revision"
    else:
        vectors[0, 0] = np.nan
    with pytest.raises(ValueError, match="embedding cache"):
        validate_embedding_cache(
            evidence, ["a::0"], vectors, metadata, model_name="test", revision="revision1"
        )


def test_index_reuse_checks_corpus_and_preprocessing_and_force(monkeypatch, tmp_path):
    from pathlib import Path

    from reliable_rag.config import load_settings
    from reliable_rag.io.jsonl import write_jsonl
    from reliable_rag.retrieval.embedder import prepare_embedding_index

    settings = load_settings(Path(__file__).parents[2] / "configs/base.yaml", environ={})
    settings.retrieval.embedding_model = "test"
    calls = []

    def encode(self, texts):
        calls.append(texts)
        return np.ones((len(texts), 2), dtype=np.float32)

    monkeypatch.setattr(
        "reliable_rag.retrieval.embedder.SentenceTransformerEmbedder.encode", encode
    )
    evidence_path, index = tmp_path / "evidence.jsonl", tmp_path / "index.npz"
    evidence = [EvidenceChunk(evidence_id="a::0", document_id="a", text="original")]
    write_jsonl(evidence_path, evidence)
    assert prepare_embedding_index(settings, evidence_path, index)
    assert not prepare_embedding_index(settings, evidence_path, index)
    assert len(calls) == 1
    evidence[0].text = "changed"
    write_jsonl(evidence_path, evidence)
    assert prepare_embedding_index(settings, evidence_path, index)
    assert prepare_embedding_index(settings, evidence_path, index, force=True)
    assert len(calls) == 3
    ids, vectors, metadata = load_embedding_cache(index)
    metadata["encoder_preprocessing"] = "different"
    with pytest.raises(ValueError, match="preprocessing"):
        validate_embedding_cache(
            evidence,
            ids,
            vectors,
            metadata,
            model_name="test",
            revision=settings.retrieval.embedding_revision,
        )
