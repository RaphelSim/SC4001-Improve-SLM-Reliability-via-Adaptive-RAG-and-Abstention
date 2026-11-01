"""Tests for internal embedding cache persistence."""

import numpy as np

from reliable_rag.retrieval import load_embedding_cache, save_embedding_cache


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
