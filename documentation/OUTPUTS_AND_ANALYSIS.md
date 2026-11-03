# Outputs and analysis

Use [setup and run](SETUP_AND_RUN.md) for commands and [architecture](SYSTEM_ARCHITECTURE.md) for schema/data flow. This guide describes the artifacts actually produced and their interpretation. JSONL means one JSON object per line; do not interpret a file's missing rows as model-predicted insufficiency.

## Dataset artifacts

For size 500 and seed 88, preparation writes `data/processed/hotpotqa/subset-500-seed88/`:

| File | Contents and purpose |
| --- | --- |
| `questions-development.jsonl` | QuestionRecord rows in selected source order: source ID, text, gold answer, project partition, support/candidate evidence IDs and source metadata. Count is the requested question count. |
| `evidence-development.jsonl` | Unique EvidenceChunk paragraphs from selected questions' candidate contexts: ID, document ID, title, joined text. Count is not the question count; the existing 500-question subset has 4,935 evidence paragraphs. Scores are absent until retrieval. |
| `manifest-development.jsonl` | Per-question ID, source split and project split assignments. This is a partition manifest, not the full sampling record. |
| `subset-development.json` | Sampling/source provenance: source kind/location/config/split, requested/dataset revision, local input hash when applicable, source count/fingerprint, seed, requested/selected size, ordered source indices and selected question IDs, method/version information, artifact SHA-256 and row counts. This also validates processed-cache reuse. |

Names use the configured project split in place of `development` for other preparations. `--manifest-dir data/splits` additionally publishes `hotpotqa-subset-500-seed88-development.json`: the same provenance without raw question/context text, suitable for versioning. This extra published file is optional; the processed selection manifest is always written.

Selection uses Python's seeded shuffled prefix over the ordered pinned source. With the same source and seed 88, sizes 5 and 500 share the first five selected IDs. A changed source/revision/order is a changed experiment even if the seed is unchanged. `dataset.subset_size: null` selects all source rows in source order. Preparation checks requested size against source count rather than silently returning a smaller sample.

`freeze_hotpotqa_splits.py` writes separate `questions-development/validation/test.jsonl` and matching `manifest-*.jsonl` with disjoint question IDs. It does not duplicate the evidence corpus/index. The default 500-question preparation itself is **all development**, not an 80/10/10 split.

## Embedding artifacts

`cache/hotpotqa-subset-500-seed88.npz` holds ordered `evidence_ids` and a float32 `embeddings` matrix. `.npz.json` holds embedding model/revision, count, dimension and ordered corpus fingerprint; new indexes additionally record device, dataset revision and preprocessing identity. The fingerprint hashes IDs and text, so changed text under an unchanged ID invalidates reuse.

For the chosen MiniLM, expected shape is **4,935 × 384** for the existing 500 sample. A cosine score is a ranking similarity, not a calibrated probability or sufficiency score. NPZ is an internal cache format, not a contract JSONL schema.

## Per-run artifacts

```text
outputs/<run-id>/
  retrieval.jsonl
  results.jsonl
  errors.jsonl
  events.jsonl
  run_metadata.json
  evaluation.json       written by evaluate_results.py
```

| Artifact | What to inspect |
| --- | --- |
| `retrieval.jsonl` | One RetrievalResult per planned question: question/query, `stage: initial`, requested `top_k`, ranked evidence with exact IDs/text/titles/scores. Saved before Qwen inference. |
| `results.jsonl` | ExperimentRecord per **valid assessment**, containing question, initial evidence, round-0 decision, final answer, gold/model flags and model/prompt metadata. Invalid answer output can still produce a record with typed abstention. Invalid assessment produces no record. |
| `errors.jsonl` | Unrepaired model-contract failures and runtime errors. Empty is valid. Model errors contain task, question ID, attempts, `failure_reason`, validation `message`, gold sufficiency, `record_written`, and the typed final answer. Runtime errors have a separate category, exception type and reason. |
| `events.jsonl` | Timestamped model-component log events: assessment/answer outcomes, invalid-output messages, formatting retry and retry success. Preserves successful repair history even when no final error exists. Does not persist raw prompts or raw generations. |
| `run_metadata.json` | Run ID/time/status, complete effective configuration, Git commit and dirty state, package versions, input/prompt SHA-256, query limit, source question path, local model path and progress counters. Metadata and completed rows are updated during the run. |
| `evaluation.json` | Derived report with run status, input hashes, retrieval metrics, classification counts/rates, answer/abstention metrics and failure breakdowns. Re-evaluation replaces this report, not raw observations. |

`run_metadata.summary` separates planned/attempted questions, recorded results, unrecorded assessment failures, answered/abstained records, model errors and runtime errors. An invalid final answer counts as both a model error and a recorded abstention; these categories intentionally overlap. Do not sum them as independent question populations.

Statuses: `running`, `completed`, `completed_with_model_errors`, `failed`, `interrupted`. A model-contract failure is a completed inference with invalid structure, not necessarily a failed run. An unrepaired assessment makes the CLI return 1 even though it processes subsequent questions. Runtime exceptions stop the run and propagate after persistence. A hard kill cannot guarantee final status updates; previously flushed rows survive. There is no resume implementation.

Historical `outputs/subset-500-seed88/` keeps its original `initial-retrieval.jsonl`, `baseline.jsonl`, `baseline.errors.jsonl`, `baseline.summary.json`, `baseline.manifest.json` and `retrieval-metrics.json`. The evaluator recognizes this layout without migrating/relabeling raw results. To analyze it without adding any file there:

```bash
python scripts/evaluate_results.py --run-id subset-500-seed88 --output /tmp/subset500-analysis.json
```

## Decisions, evidence state and answering

The contract values are lowercase strings; the enum member names are uppercase:

- `SUFFICIENT` → `"sufficient"`: Qwen predicts supplied evidence is enough. At least one supplied supporting ID is required. The answer generator is then invoked.
- `NEED_MORE_EVIDENCE` → `"need_more_evidence"`: Qwen predicts a gap and provides missing information/search query. The baseline abstains immediately; it does not execute that query.

Compare these predictions with `gold_sufficient_initially/finally`. Gold sufficiency means all annotated supporting paragraph IDs are present. It is a controlled annotation-based proxy, not a guarantee that every possible question-answer interpretation has been resolved. Initial and final flags match in the single-round baseline.

`condition` is complete when all gold supports were retrieved, partially incomplete when some were retrieved, fully incomplete when none were retrieved. `distractor_heavy` exists for controlled evidence experiments but is not automatically assigned by this runner.

`final_answer.answer` is text when answering and null when abstaining. `abstained` specifies the action; `abstention_reason` categorizes why; `used_evidence_ids` records supplied evidence used by an answer. Citation validity does not establish answer correctness.

| Reason | Meaning and current usage |
| --- | --- |
| `insufficient_evidence` | Valid semantic insufficiency decision, or valid answer-generator evidence-based refusal. Ordinary intentional abstention. |
| `invalid_model_output` | JSON/schema/citation output remained invalid after the allowed formatting repair. This is a contract failure, not a model-insufficiency prediction. |
| `retrieval_failed` | Reserved shared label for retrieval failure handling. The current baseline does not emit it automatically; even empty evidence goes through assessment. |
| `max_retrieval_rounds_reached` | Reserved shared label for a future bounded adaptive controller. The current baseline does not emit it. |

If assessment is invalid, `errors.jsonl` retains the typed invalid-output abstention, but no SufficiencyDecision or ExperimentRecord is invented. If the assessment was valid and final-answer output is invalid, the record remains with its valid prediction and invalid-output abstention. Runtime failures are separately categorized and never relabeled as semantic insufficiency.

## Structured-output diagnosis

Common validation failures include malformed JSON, missing required keys, wrong question ID, duplicate IDs, nonexistent `supporting_evidence_ids` or `used_evidence_ids`, an answer without citations, or incompatible status/abstention fields. Exact supplied IDs matter: `bruce-willis` cannot substitute for `bruce-willis::0`.

The existing retry prompt requests a formatting/schema repair using the invalid output and validation error. It does not retrieve additional evidence. `events.jsonl` records the invalid attempt, retry and any success. Failure after the second total attempt becomes ModelOutputError, with the final message retained in `errors.jsonl`. A repaired case counts as a valid decision; compare events as well as final errors when measuring formatting fragility.

## Metrics and denominators

The evaluator reuses `src/reliable_rag/evaluation/metrics.py`. Every rate includes numerator, denominator and value; a zero denominator produces JSON `null`, not zero.

Supporting Evidence Recall@k is **micro-aggregated paragraph support recall**:

- Numerator: total annotated support IDs present among each question's first k results.
- Denominator: total annotated support IDs across evaluated questions.
- Higher is better. Recall@1 measures the first paragraph, Recall@3 the first three. This is not the percentage of questions with every required support found, nor answer accuracy.
- The default saved depth is 3, so Recall@5 cannot be inferred. Increase `retrieval.initial_top_k` in an explicitly recorded configuration and run retrieval again to measure it.

Binary classification uses **need-more-evidence as the positive class**, preserving the existing protocol:

| Count | Gold evidence | Model prediction |
| --- | --- | --- |
| TP | Insufficient | Need more evidence |
| FP | Sufficient | Need more evidence |
| TN | Sufficient | Sufficient |
| FN | Insufficient | Sufficient |

Classification metrics use only recorded cases with valid assessments; unrepaired assessments are reported as failures, not silently assigned a class.

| Metric | Numerator | Denominator | Interpretation |
| --- | --- | --- | --- |
| Sufficiency accuracy | TP + TN | TP + FP + TN + FN | Higher: correct decision fraction. |
| Precision | TP | TP + FP | Higher: predicted insufficiency is justified. |
| Recall | TP | TP + FN | Higher: detects gold insufficiency. |
| F1 | 2TP | 2TP + FP + FN | Higher: balances insufficiency precision/recall. |
| False sufficiency rate | FN | TP + FN | Lower: fewer unsafe sufficient predictions on incomplete evidence. |
| False insufficiency rate | FP | FP + TN | Lower: fewer needless insufficiency predictions on complete evidence. |
| Answer coverage, recorded | Non-abstained records | All records | Higher answers more often; interpret alongside unsupported answering. |
| Answer coverage, attempted | Non-abstained records | Attempted questions | Includes invalid-assessment/runtime impact in the denominator. |
| Correct abstention rate | Abstained records with gold-insufficient final evidence | All gold-insufficient records | Higher refusal on insufficient evidence; includes typed contract-failure abstentions, so also inspect reasons. |
| False abstention rate | Abstained records with gold-sufficient final evidence | All gold-sufficient records | Lower unnecessary refusal; includes typed failures. |
| Unsupported answer rate | Answered records with gold-insufficient final evidence | All gold-insufficient records | Lower answering despite missing annotated support. |
| Unsupported answer rate among answered | Answered records with gold-insufficient final evidence | All answered records | Lower unsupported fraction of substantive answers. Different denominator from the preceding metric. |

All answer metrics except `coverage_attempted` operate on recorded cases. Model-error sidecars and recorded abstentions overlap for final-answer failures. Missing assessment records can bias conditional metrics; always show record count, assessment-failure count and attempted coverage together. Unsupported answering is defined by missing gold evidence, not a separate factuality judge.

**Not implemented:** answer Exact Match, answer token F1, adaptive recovery rate and recovery curves. They are listed as unimplemented in reports. No adaptive results or answer-quality claims should be inferred from coverage or sufficiency accuracy. Exact answer evaluation would require an explicitly agreed normalization/scoring protocol.

## Existing 500-question observations

The preserved seed-88 run has 500 attempts, 459 records, 41 invalid assessments, 55 total unrepaired model-output failures (41 assessment, 14 final answer), 248 answered records and 211 recorded abstentions. Classification accuracy is 313/459 = 68.19%; attempted answer coverage is 248/500 = 49.6%. Recall@1 = 406/1,000 = 0.406; Recall@3 = 712/1,000 = 0.712. Unsupported answers among answered = 89/248 = 35.89%.

These are a development baseline with material schema/citation failures and false sufficiency, not a final reliability result. They motivate future controlled comparisons without changing prompts or reclassifying failures during cleanup. The five-question verification run `quick5-seed88-cleanup` produced five records, one answer, four insufficiency abstentions and no contract/runtime failures; it validates the workflow only.

## Inspecting and comparing runs

```bash
python scripts/evaluate_results.py --run-id dev500-seed88-v1
python scripts/validate.py jsonl outputs/dev500-seed88-v1/retrieval.jsonl --schema retrieval-result
python scripts/validate.py jsonl outputs/dev500-seed88-v1/results.jsonl --schema experiment-record
```

Read metadata first, then evaluation counts/rates, errors grouped by task/message, and representative result rows. Compare runs only after checking question IDs/order and input hashes, source revision, candidate scope/top-k, embedding and SLM revisions, prompt hashes/version, decoding settings and valid-record populations. Prefer paired comparisons on the same questions and report missing decisions explicitly. Preserve prepared inputs if regenerating into a different source/config at the same path: the evaluator refuses a changed question hash. Configuration snapshots, package versions and Git state document provenance, but editable installs and broad dependency ranges are not a frozen environment lockfile.
