# Reproducibility

## Shared baseline

- Use Python 3.11; record the exact patch version for final runs.
- Use `project_seed` (development default 42) as the source for every stochastic component, including data construction, sampling, retrieval tie handling, and generation where supported. Derive component seeds deterministically and document the derivation.
- Install dependencies from `pyproject.toml` extras. Before final evaluation, capture the resolved environment; do not multiply hand-maintained version lists.
- Store a resolved configuration snapshot per run. Final generation settings should be deterministic where possible (`temperature: 0`, `do_sample: false`) and must be frozen before test evaluation.
- Record exact model identifiers and revisions, dataset revisions/split IDs when available, prompt versions, and evidence construction policy.
- Identify code with a Git commit hash. If a working tree is dirty, preserve a patch or mark the run as non-final.

## Cache policy

Caches are regenerable and ignored by Git. Keys should include dataset revision, source/preprocessing version, model ID/revision, relevant retrieval configuration, and prompt version. Cache hits must not silently cross these boundaries. Keep embeddings internal; do not serialize them in standard contract JSONL.

## Final run capture

For each final run, save at least:

- raw JSONL `ExperimentRecord` output;
- resolved YAML/JSON configuration snapshot;
- timestamp, seed, split/dataset revision, model ID/revision, prompt versions, and Git commit;
- environment snapshot generated from the active Python environment, for example `python -m pip freeze > outputs/<run-id>/requirements-lock.txt`;
- the output from `python scripts/check_environment.py` and any runtime-specific hardware details.

Prefer rerunning from a clean checkout of the frozen commit and captured environment. If an optional runtime or accelerator is unavailable, record that difference; do not imply bitwise determinism where the backend does not provide it.
