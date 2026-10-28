"""Configuration loading and validation tests."""

from pathlib import Path

import pytest

from reliable_rag.config import load_settings

ROOT = Path(__file__).parents[2]


def test_base_config_loads_with_development_defaults() -> None:
    settings = load_settings(ROOT / "configs" / "base.yaml", environ={})
    assert settings.project_seed == 42
    assert settings.retrieval.initial_top_k == 3
    assert settings.retrieval.max_total_retrieval_rounds == 2
    assert settings.slm.temperature == 0
    assert settings.slm.do_sample is False


def test_documented_environment_overrides_apply() -> None:
    settings = load_settings(
        ROOT / "configs" / "base.yaml",
        environ={
            "RELIABLE_RAG_DATA_ROOT": "/tmp/data-root",
            "RELIABLE_RAG_PROJECT_SEED": "100",
        },
    )
    assert settings.data_root == Path("/tmp/data-root")
    assert settings.project_seed == 100


def test_invalid_config_reports_field_and_fails(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        "project_seed: 42\n"
        "dataset: {dataset_name: test}\n"
        "retrieval: {embedding_model: test, initial_top_k: 0, followup_top_k: 3}\n"
        "slm: {model_id: mock, max_input_tokens: 10, max_new_tokens: 2, temperature: 0}\n"
        "experiment: {system_variant: standard_rag, evidence_condition: complete, run_id: test}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="initial_top_k"):
        load_settings(invalid, environ={})


def test_unknown_config_key_fails(tmp_path: Path) -> None:
    invalid = tmp_path / "unknown.yaml"
    invalid.write_text((ROOT / "configs" / "base.yaml").read_text() + "typo_key: 1\n")
    with pytest.raises(ValueError, match="typo_key"):
        load_settings(invalid, environ={})
