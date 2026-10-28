# Experiment protocol

## Dataset split conventions

Use the serialized `DatasetSplit` labels `development`, `validation`, and `test` consistently in every `QuestionRecord`. Preserve official source-dataset partitions when they fit the study design and record the source split mapping. If a deterministic sub-sample or new partition is required, create an ID-only manifest under `data/splits/` using the shared seed, document the source dataset revision and generation method, and keep question IDs mutually exclusive across partitions. Do not move examples between partitions in response to model results. Development is for debugging, validation is for selection, and test is held out until freeze.

1. Development data may be used for debugging, fixture checks, and implementation iteration.
2. Use validation data to select Top-k, prompt wording/version, controller rules, retrieval settings, and any thresholds.
3. Keep test data untouched until architecture and settings are frozen. Do not tune from, repeatedly inspect, or change prompts/configuration after seeing test results.
4. Freeze an experiment manifest before test runs: config snapshot, code commit, model identifier/revision, dataset and split/revision, seed, decoding settings, evidence-construction policy, and prompt versions.
5. Every final experiment stores the resolved config snapshot, Git commit hash if available, model identifier, dataset/split identifier, seed, timestamp, and raw output records. See the reproducibility procedure in `10_reproducibility.md`.
6. Comparisons use the same SLM, quantization, decoding settings, evidence-construction policy, and runtime where practical. Record deviations explicitly.
7. Do not silently change research assumptions. If the protocol changes, add a decision-log entry and state whether validation/test results have already been examined.

The initial experiment setup and support annotations define the controlled labels, not every potentially valid evidence path. Report those limitations when interpreting results.
