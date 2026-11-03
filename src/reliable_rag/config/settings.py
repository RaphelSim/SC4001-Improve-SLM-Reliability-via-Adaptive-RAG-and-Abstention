"""Validated hierarchical configuration with a small environment override surface."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from reliable_rag.contracts.enums import SystemVariant

DEFAULT_PROJECT_SEED = 88


class StrictSettings(BaseModel):
    """Disallow unknown keys to make misspelled configuration fail early."""

    model_config = ConfigDict(extra="forbid")


class DatasetSettings(StrictSettings):
    """Pinned HotpotQA source, deterministic sample and project partition."""

    dataset_name: str = "hotpotqa/hotpot_qa"
    dataset_config: str = "distractor"
    source_split: str = "train"
    revision: str | None = None
    subset_size: int | None = Field(default=500, ge=1)
    project_split: str = "development"
    input_path: Path | None = None

    @field_validator("project_split")
    @classmethod
    def known_split(cls, value: str) -> str:
        if value not in {"development", "validation", "test"}:
            raise ValueError("project_split must be development, validation or test")
        return value


class RetrievalSettings(StrictSettings):
    """Model selection and bounded retrieval settings."""

    embedding_model: str = Field(min_length=1)
    embedding_revision: str | None = None
    embedding_device: str = "cpu"
    initial_top_k: int = Field(ge=1)
    followup_top_k: int = Field(ge=1)
    max_total_retrieval_rounds: int = Field(default=2, ge=1, le=2)
    similarity_metric: str = "cosine"

    @field_validator("similarity_metric")
    @classmethod
    def metric_name_must_not_be_empty(cls, value: str) -> str:
        value = value.strip().lower()
        if not value:
            raise ValueError("similarity_metric must not be empty")
        return value


class SLMSettings(StrictSettings):
    """Model selection, structured generation, and prompt settings."""

    model_id: str = Field(min_length=1)
    revision: str | None = None
    runtime: str = Field(default="auto", min_length=1)
    quantization: str | None = None
    max_input_tokens: int = Field(ge=1)
    max_new_tokens: int = Field(ge=1)
    temperature: float = Field(ge=0.0)
    do_sample: bool = False
    enable_thinking: Literal[False] = False
    prompt_version: str = Field(default="v1", min_length=1)
    sufficiency_prompt_path: Path = Path("prompts/sufficiency_v1.txt")
    final_answer_prompt_path: Path = Path("prompts/final_answer_v1.txt")
    structured_retry_prompt_path: Path = Path("prompts/structured_retry_v1.txt")
    format_retry_limit: int = Field(default=1, ge=0, le=1)


class ExperimentSettings(StrictSettings):
    """Per-run labels; final conditions must be frozen before test evaluation."""

    system_variant: SystemVariant
    run_id: str = Field(min_length=1)


class Settings(StrictSettings):
    """Complete project configuration loaded from YAML and approved env overrides."""

    project_seed: int = Field(ge=0)
    data_root: Path = Path("data")
    cache_root: Path = Path("cache")
    output_root: Path = Path("outputs")
    dataset: DatasetSettings
    retrieval: RetrievalSettings
    slm: SLMSettings
    experiment: ExperimentSettings

    @property
    def dataset_dir(self) -> Path:
        size = self.dataset.subset_size if self.dataset.subset_size is not None else "all"
        return self.data_root / "processed" / "hotpotqa" / f"subset-{size}-seed{self.project_seed}"

    @property
    def questions_path(self) -> Path:
        return self.dataset_dir / f"questions-{self.dataset.project_split}.jsonl"

    @property
    def evidence_path(self) -> Path:
        return self.dataset_dir / f"evidence-{self.dataset.project_split}.jsonl"

    @property
    def selection_path(self) -> Path:
        return self.dataset_dir / f"subset-{self.dataset.project_split}.json"

    @property
    def index_path(self) -> Path:
        size = self.dataset.subset_size if self.dataset.subset_size is not None else "all"
        return self.cache_root / f"hotpotqa-subset-{size}-seed{self.project_seed}.npz"

    @model_validator(mode="after")
    def enforce_shared_round_limit(self) -> Settings:
        if self.retrieval.max_total_retrieval_rounds > 2:
            raise ValueError("the foundation supports at most two total retrieval rounds")
        return self


_ENV_PATH_FIELDS = {
    "RELIABLE_RAG_DATA_ROOT": "data_root",
    "RELIABLE_RAG_CACHE_ROOT": "cache_root",
    "RELIABLE_RAG_OUTPUT_ROOT": "output_root",
}
_ENV_SEED = "RELIABLE_RAG_PROJECT_SEED"


def _apply_environment_overrides(raw: dict[str, Any], environ: dict[str, str]) -> dict[str, Any]:
    """Apply the documented path and seed overrides without hidden config state."""
    updated = dict(raw)
    for variable, field_name in _ENV_PATH_FIELDS.items():
        if environ.get(variable):
            updated[field_name] = environ[variable]
    if environ.get(_ENV_SEED):
        try:
            updated["project_seed"] = int(environ[_ENV_SEED])
        except ValueError as exc:
            raise ValueError(f"{_ENV_SEED} must be an integer") from exc
    return updated


def load_settings(
    path: str | Path = "configs/base.yaml", *, environ: dict[str, str] | None = None
) -> Settings:
    """Load YAML, apply documented environment overrides, and validate with Pydantic.

    Args:
        path: YAML configuration path, interpreted relative to the current working directory.
        environ: Optional environment mapping, useful for deterministic callers and tests.

    Raises:
        FileNotFoundError: If the requested YAML file does not exist.
        ValueError: If YAML is malformed, has the wrong root type, or an override is invalid.
        pydantic.ValidationError: If the resulting settings violate the typed schema.
    """
    config_path = Path(path)
    try:
        raw_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw_data, dict):
        raise ValueError(f"Configuration root in {config_path} must be a YAML mapping")
    source_env = os.environ if environ is None else environ
    values = _apply_environment_overrides(raw_data, dict(source_env))
    try:
        return Settings.model_validate(values)
    except ValidationError as exc:
        errors = "; ".join(
            f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}" for item in exc.errors()
        )
        raise ValueError(f"Invalid configuration in {config_path}: {errors}") from exc
