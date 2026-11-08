# Output guide

The current paragraph retrieval experiment, its three primary metrics, duplicate counting example, artifact schema and comparison interpretation are documented in [System architecture](SYSTEM_ARCHITECTURE.md). Copy-paste-ready commands and cache behaviour are in [Setup and run](SETUP_AND_RUN.md).

Historical `scripts/run_experiment.py` artifacts use the former answer/abstention `ExperimentRecord` and are retained for reproducibility. Their answer coverage, unsupported-answer and abstention metrics are not part of the new paragraph retrieval study. Do not mix historical answer results with `RetrievalStudyRecord` rows in a comparison.
