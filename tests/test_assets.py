"""Model preparation checks exact cached revisions and incomplete snapshots."""

import json
from pathlib import Path

import pytest

from reliable_rag.assets import resolve_model_path


def snapshot(path):
    path.mkdir(exist_ok=True)
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (path / name).write_text("{}")
    (path / "model.safetensors").write_bytes(b"weights")
    return path


def test_hub_resolution_reuses_pinned_cache_even_when_download_allowed(monkeypatch, tmp_path):
    calls = []
    model = snapshot(tmp_path / "snapshot")

    def download(**kwargs):
        calls.append(kwargs)
        return str(model)

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    for allow in (False, True):
        assert (
            resolve_model_path(
                "org/model", revision="commit123", cache_root=tmp_path, allow_download=allow
            )
            == model
        )
    assert all(call["local_files_only"] for call in calls)
    assert all(call["revision"] == "commit123" for call in calls)


def test_incomplete_cache_is_repaired_only_when_explicitly_allowed(monkeypatch, tmp_path):
    calls = []
    model = tmp_path / "snapshot"
    model.mkdir()

    def download(**kwargs):
        calls.append(kwargs["local_files_only"])
        if not kwargs["local_files_only"]:
            snapshot(model)
        return str(model)

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    with pytest.raises(ValueError, match="incomplete"):
        resolve_model_path("org/model", revision="rev", cache_root=tmp_path)
    assert (
        resolve_model_path("org/model", revision="rev", cache_root=tmp_path, allow_download=True)
        == model
    )
    assert calls == [True, True, False]


def test_local_model_path_and_sharded_completeness(monkeypatch, tmp_path):
    model = snapshot(tmp_path)

    def forbid(**kwargs):
        raise AssertionError("Hub must not be used for a local model")

    monkeypatch.setattr("huggingface_hub.snapshot_download", forbid)
    assert (
        resolve_model_path(str(model), revision=None, cache_root=Path("unused")) == model.resolve()
    )
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": "missing.safetensors"}})
    )
    with pytest.raises(ValueError, match="weights"):
        resolve_model_path(str(model), revision=None, cache_root=Path("unused"))
