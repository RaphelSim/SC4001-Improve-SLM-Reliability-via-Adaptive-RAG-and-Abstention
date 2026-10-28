# Configuration

`pyproject.toml` is the dependency source of truth. YAML config is parsed by `load_settings()` in `config/settings.py` and validated by nested Pydantic models. Unknown keys are errors. Paths are relative to the process working directory unless overridden; no user-specific absolute paths are committed.

## Standard keys

| Section | Keys | Notes |
|---|---|---|
| General | `project_seed`, `data_root`, `cache_root`, `output_root` | Seed defaults to 42. Every stochastic component must derive its seed from it. |
| Dataset | `dataset_name`, split names, optional `development_path`, `validation_path`, `test_path` | Names are labels; paths are optional and dataset-specific. |
| Retrieval | `embedding_model`, `initial_top_k`, `followup_top_k`, `max_total_retrieval_rounds`, `similarity_metric`, `deduplicate_evidence` | Current top-k 3, MiniLM, and cosine are development defaults, not validated final values. Maximum total rounds is 2. |
| SLM | `model_id`, `runtime`, `quantization`, `max_input_tokens`, `max_new_tokens`, `temperature`, `do_sample` | Runtime-neutral placeholders. Deterministic defaults are temperature 0 and sampling disabled. Final choices must be frozen before test. |
| Experiment | `system_variant`, `evidence_condition`, `run_id` | Enum strings must match `contracts/enums.py`. |

`configs/base.yaml` provides the shared provisional baseline. `configs/development.yaml` points only to a synthetic fixture and mock model label for local wiring.

## Loading and overrides

```python
from reliable_rag.config import load_settings

settings = load_settings("configs/base.yaml")
```

The only recognized environment overrides are `RELIABLE_RAG_DATA_ROOT`, `RELIABLE_RAG_CACHE_ROOT`, `RELIABLE_RAG_OUTPUT_ROOT`, and integer `RELIABLE_RAG_PROJECT_SEED`. Tokens such as `HF_TOKEN` are not loaded into shared settings. Keep secrets in an ignored local `.env` or the runtime's secret store; `.env.example` contains names only.

Validate with `python scripts/validate_config.py configs/base.yaml`. Record the fully resolved config for every final run.
