# Setup and run

This repository runs a single-round RAG + Abstention baseline: prepare HotpotQA, retrieve evidence with MiniLM, ask Qwen whether it is sufficient, then answer or abstain. Adaptive follow-up retrieval is not implemented. See [architecture](SYSTEM_ARCHITECTURE.md) for internals and [outputs and analysis](OUTPUTS_AND_ANALYSIS.md) for interpretation.

## First-time environment setup

Use **native ARM Python 3.11 on an Apple Silicon Mac** with Metal access. `pyproject.toml` requires Python `>=3.11,<3.12`. Dataset preparation, retrieval and automated tests can run elsewhere; the configured inference backend requires MLX on Apple Silicon. Installing a CUDA dependency group does not provide an alternative inference implementation.

From a new clone, enter the repository root and run:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,mac]'
python scripts/check_environment.py --check-mlx
python scripts/validate.py config
```

All subsequent commands assume this directory and activated environment. The editable install makes `reliable_rag` importable from scripts. No setup shell wrapper is required. To finish a session, run `deactivate`; results remain on disk.

MiniLM, Qwen3-1.7B MLX 4-bit, and HotpotQA are public. No Hugging Face API key is required. If authentication or higher rate limits are needed, run `hf auth login` or export `HF_TOKEN` in your shell. `.env.example` is illustrative; the application does not load `.env` automatically. Never commit tokens.

`configs/base.yaml` is the shared configuration: seed **88**, subset **500**, HotpotQA distractor/train, initial top-k **3**, MiniLM on CPU, Qwen on MLX, greedy non-thinking decoding, prompt v1 and one formatting retry. Dataset and model revisions are pinned. Paths are relative to the working directory. Use `--config PATH` for a complete alternative YAML; it is not merged into the base file. Unknown keys fail validation. `configs/local.yaml` is ignored for local experiments.

Environment overrides are `RELIABLE_RAG_DATA_ROOT`, `RELIABLE_RAG_CACHE_ROOT`, `RELIABLE_RAG_OUTPUT_ROOT`, and `RELIABLE_RAG_PROJECT_SEED`. An explicit CLI seed wins over the environment. For a larger or smaller sample, pass the same `--subset-size` to preparation, indexing, and execution; the flag does not edit YAML. Set `dataset.subset_size: null` in an alternative YAML to use all source records in source order.

Generated paths:

- `data/processed/hotpotqa/subset-N-seedS/`: normalized data and selection provenance.
- `cache/datasets/`: Hugging Face Arrow dataset cache.
- `cache/huggingface/hub/`: revision-specific model snapshots.
- `cache/hotpotqa-subset-N-seedS.npz` and `.npz.json`: evidence vectors and provenance.
- `outputs/RUN_ID/`: persisted experiment and evaluation artifacts.

These directories are created when needed and ignored by Git. `data/splits/` can hold small versioned selection/split manifests. Historical raw exports may remain in `data/raw/`; the normal workflow does not require a second raw-data copy.

## Five-question quick run

This uses the same pipeline as a larger experiment, with real HotpotQA questions:

```bash
python scripts/prepare_hotpotqa.py --subset-size 5
python scripts/prepare_models.py
python scripts/build_embedding_index.py --subset-size 5
python scripts/run_experiment.py --subset-size 5 --run-id quick5-seed88
python scripts/evaluate_results.py --run-id quick5-seed88
python scripts/validate.py jsonl outputs/quick5-seed88/results.jsonl --schema experiment-record
```

Preparation selects five questions, not five paragraphs. Each question contributes its supplied distractor-context paragraphs; the tested five-question sample contains 50 unique paragraphs. The first source download can be substantially larger than the selected sample, but embeddings are created only for selected evidence. Five examples check operation, not research quality.

## 500-question development experiment

```bash
python scripts/prepare_hotpotqa.py --subset-size 500
python scripts/prepare_models.py
python scripts/build_embedding_index.py --subset-size 500
python scripts/run_experiment.py --subset-size 500 --run-id dev500-seed88-v1
python scripts/evaluate_results.py --run-id dev500-seed88-v1
```

The defaults already specify 500 and seed 88; explicit sizes make examples easy to adapt. The existing `subset-500-seed88` run and its 4,935-paragraph corpus are preserved. Use a new run ID to compare against it. A 10,000-question sample is optional, not required to exercise the current baseline; start at 500 to measure failures and time before scaling. This sample is development data, not a held-out evaluation partition.

## Rerun using caches

Repeat the preparation commands above, then choose a new run ID:

```bash
python scripts/prepare_hotpotqa.py --subset-size 500
python scripts/prepare_models.py
python scripts/build_embedding_index.py --subset-size 500
python scripts/run_experiment.py --subset-size 500 --run-id dev500-seed88-v2
python scripts/evaluate_results.py --run-id dev500-seed88-v2
```

Compatible prepared data, model files, and evidence vectors are reused. Query embedding, retrieval and Qwen generation execute again. Model inference responses are not cached. Reusing an existing run ID raises an error before inference; there is no automatic resume. `--limit 20` on execution uses the first 20 prepared questions against the same corpus, rather than sampling a new 20-question dataset.

For an explicit repeat with identical saved retrieval:

```bash
python scripts/run_experiment.py --subset-size 500 --retrieval outputs/dev500-seed88-v1/retrieval.jsonl --run-id dev500-seed88-repeat
```

IDs, order, query text, initial stage and configured top-k must match. The pipeline also checks candidate evidence scope before assessment. Precomputed retrieval is recorded as a hashed input; its original embedding provenance must be retained with the source run.

## Force regeneration

```bash
python scripts/prepare_hotpotqa.py --subset-size 5 --force
python scripts/build_embedding_index.py --subset-size 5 --force
python scripts/run_experiment.py --subset-size 5 --run-id quick5-seed88 --force
python scripts/evaluate_results.py --run-id quick5-seed88
```

Dataset `--force` regenerates normalized artifacts and provenance. Index `--force` re-embeds evidence. Experiment `--force` **deletes and replaces the entire named run directory**, including its previous evaluation. Prefer a new run ID for scientific comparisons. Model preparation repairs missing assets and reuses complete snapshots; no redundant force-download flag is exposed. Evaluation recomputes and replaces its derived report, preserving raw results.

## Script reference

Every script supports `--help`. There are six normal workflow commands and two optional developer/research operations.

| Script | Inputs, output and cache behavior |
| --- | --- |
| `check_environment.py` | No positional arguments. Reports Python, architecture and installed packages. `--check-mlx` additionally executes a Metal operation; failure returns 1. Does not install packages or modify caches. |
| `prepare_hotpotqa.py` | No positionals. `--config` defaults to `configs/base.yaml`; `--subset-size`/`--seed` default to YAML (500/88). `--input PATH` overrides `dataset.input_path` for local raw HotpotQA JSON-list/JSONL. `--output-dir PATH` overrides the derived processed directory. `--manifest-dir PATH` additionally publishes an ID-only provenance JSON. `--force` bypasses processed-cache reuse. Outputs three JSONL files and `subset-SPLIT.json`. Source, split, revision, seed, size, hashes and selected IDs are checked before reuse. Missing/incompatible data is rebuilt. |
| `prepare_models.py` | No positionals. `--config` chooses models/cache root. Tries exact configured local snapshots first, verifies required files, downloads missing assets. `--offline` forbids downloads and fails for unavailable/incomplete models. Runtime execution is offline by default, so run this stage first. |
| `build_embedding_index.py` | No positionals. `--config`, `--subset-size`, `--seed` derive inputs; `--evidence PATH` and `--index PATH` override them. `--batch-size` defaults to 64 and must be positive. `--force` rebuilds even a compatible index. Reads prepared evidence; writes NPZ plus JSON metadata. Validates IDs/order, ordered text fingerprint, model/revision, dimensions, finite nonzero vectors and preprocessing before reuse. |
| `run_experiment.py` | No positionals. `--config`, `--subset-size`, `--seed`; `--run-id` defaults to `experiment.run_id` (`development`). `--questions PATH`, `--evidence PATH`, `--index PATH` override derived inputs. Optional `--retrieval PATH` skips fresh retrieval. Optional positive `--limit N`; `--force` replaces existing run. Writes retrieval/results/errors/events/metadata under the run ID. No automatic preparation/downloads or inference-response reuse. Only `rag_abstention` is executable. Exit 1 also indicates completed runs with invalid assessments; inspect saved status/errors before diagnosing a crash. |
| `evaluate_results.py` | Required `--run-id ID`; optional `--config`; `--k K [K ...]` defaults to 1 and 3. Cannot evaluate beyond saved retrieval depth. `--output PATH` defaults to the run's `evaluation.json`. Reads saved questions/retrieval/results/errors/metadata, checks question/retrieval hashes for modern runs, and writes metrics with denominators. Also reads the preserved historical `baseline.*` run layout. It does not generate predictions. |
| `freeze_hotpotqa_splits.py` | Optional partition operation. Positional `questions` JSONL; required `--output-dir`; optional `--config`, `--seed` (shared 88), `--source-split` (train), `--development-fraction` (0.8), `--validation-fraction` (0.1). Assigns unique question IDs to development/validation/test and writes questions/manifest JSONL for each. Uses the existing full evidence corpus/index, not independent re-sampling. Use a fresh directory; `--force` explicitly replaces existing split files. |
| `validate.py` | Subcommands: `config [--config PATH]`; `jsonl PATH --schema NAME`; `schemas [--output-dir PATH]` (default `schemas`). JSONL names: `question-record`, `evidence-chunk`, `evidence-set`, `retrieval-request`, `retrieval-result`, `sufficiency-decision`, `answer-result`, `experiment-record`, `adaptive-round-record`. Config/JSONL validation is read-only and fails on malformed input. `schemas` regenerates the eight checked-in schemas from Pydantic definitions. |

Example of a disjoint partition for a later frozen evaluation:

```bash
python scripts/freeze_hotpotqa_splits.py data/processed/hotpotqa/subset-500-seed88/questions-development.jsonl --output-dir data/processed/hotpotqa/split500-seed88
python scripts/run_experiment.py --questions data/processed/hotpotqa/split500-seed88/questions-validation.jsonl --run-id validation500-seed88
```

This uses the default 500-question evidence corpus/index. Do not call `prepare_hotpotqa` independently for three project splits with the same source/seed/size and assume they are disjoint: it would select the same source IDs. Freeze partitions, tune on development/validation, then freeze configuration/prompts before test. Five-question subsets are too small for all three nonempty partitions.

## Troubleshooting and development checks

- **`ModuleNotFoundError: reliable_rag`**: activate `.venv` and run the editable install from the root.
- **MLX device failure / process abort**: use native ARM Python and a session with Metal access. The environment report without `--check-mlx` checks availability only. Automated tests inject clients and do not require Metal.
- **Missing model**: run `prepare_models.py`; `--offline` is useful after the first successful fetch. Tokenizers, weight shards and SentenceTransformer pooling configuration must be present.
- **Stale index**: run index preparation again. A different seed/size changes paths; changed evidence/model/revision invalidates compatibility. Do not copy an arbitrary index into place.
- **Hub offline/network error while preparing a new sample**: a matching processed subset needs no network. Loading a different sample still requires access to a cached or downloadable source. After an initial download, offline use can set `HF_HOME="$PWD/cache/huggingface" HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1` before preparation. Keep the pinned source cache; the loader rejects an offline fallback to another revision. Unset offline variables when new assets are required.
- **Invalid model output**: generation can complete with malformed JSON, missing keys or invented evidence IDs. One formatting retry is allowed; inspect `errors.jsonl` and `events.jsonl`. This is distinct from a semantic insufficiency decision. Do not loosen validation merely to increase record count.
- **Interrupted or failed run**: each completed result/error is flushed, and metadata records progress/status. Use a new ID to rerun; automatic resume is not implemented. A hard process kill may leave `status: running`; compare counts and last saved row.
- **Input too long**: the adapter fails instead of silently truncating beyond 8,192 input tokens. Preserve the runtime error and adjust an explicitly recorded experimental configuration if needed.

Developer checks:

```bash
python -m pytest -q
ruff check src scripts tests
ruff format --check src scripts tests
```

Tests cover schemas, deterministic sampling, caches, candidate scopes, strict output parsing/retry, gating, failure persistence and evaluation denominators. They are automated tests, not an alternative user-facing pipeline.
