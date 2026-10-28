# Data contracts and IDs

`src/reliable_rag/contracts/records.py` is the canonical source for shared Pydantic v2 models. Field names below are the project vocabulary; schema changes are API changes and require team agreement, updated docs, tests, and regenerated JSON Schemas. Embedding vectors are internal cache details and never belong in standard JSONL records.

## String enums

Values are stable serialized strings, not Python member names:

| Enum | Values | Meaning |
|---|---|---|
| `DatasetSplit` | `development`, `validation`, `test` | Data-use partition; test remains held out until freeze. |
| `EvidenceCondition` | `complete`, `partially_incomplete`, `fully_incomplete`, `distractor_heavy` | Controlled initial evidence condition; definitions and limitations are in `02_definitions.md`. |
| `SystemVariant` | `standard_rag`, `rag_abstention`, `adaptive_retrieval`, `adaptive_retrieval_abstention` | Named system comparison conditions. |
| `SufficiencyStatus` | `sufficient`, `need_more_evidence` | Model-predicted decision, not a gold label. |
| `AbstentionReason` | `insufficient_evidence`, `retrieval_failed`, `invalid_model_output`, `max_retrieval_rounds_reached` | Why an answer was withheld. |
| `RetrievalStage` | `initial`, `follow_up` | Round 0 and round 1 retrieval labels. |

## Record fields

- `EvidenceChunk`: required `evidence_id`, `document_id`, `title` (nullable), and `text`; optional `retrieval_score`, `retrieval_rank`, and `retrieval_stage`. It contains no embedding.
- `QuestionRecord`: `question_id`, `question`, `gold_answer`, `split`, `gold_support_ids`, `candidate_evidence_ids`, and optional `metadata`. Support IDs are nonempty and unique.
- `EvidenceSet`: `question_id`, `condition`, `evidence`, and `gold_support_ids`. Complete/partial/full conditions are validated against support-ID presence. `gold_sufficient` and `evidence_ids` are computed properties, not duplicated storage.
- `RetrievalRequest`: `question_id`, `query`, `stage`, `top_k`, `exclude_evidence_ids`.
- `RetrievalResult`: `question_id`, `query`, `stage`, `top_k`, `evidence`.
- `SufficiencyDecision`: `question_id`, `status`, nullable `missing_information`, nullable `search_query`, and `supporting_evidence_ids`. `sufficient` requires `search_query` to be null and normally leaves `missing_information` null; `need_more_evidence` requires both to be non-empty. No hidden chain-of-thought is stored.
- `AnswerResult`: `question_id`, nullable `answer`, `abstained`, nullable `abstention_reason`, and `used_evidence_ids`. Abstentions use `answer: null` and require a reason; substantive answers must be non-empty and have no reason.
- `AdaptiveRoundRecord`: `round_index`, `retrieval`, and `sufficiency_decision`. Round 0 requires `initial`; round 1 requires `follow_up`.
- `ExperimentRecord`: canonical per-case raw result with experiment ID, question, initial condition/evidence, system variant, round history, final evidence IDs, final answer, initial/final gold sufficiency, optional initial/final model-predicted sufficiency, and metadata. It stores raw observations, not aggregate EM/F1/recovery metrics.

Unknown fields are rejected, and model IDs/evidence IDs are checked for basic validity. The record-level validators enforce matching question IDs, contiguous rounds, evidence provenance for final IDs, and agreement between recorded gold-sufficiency booleans and controlled support coverage.

## Stable IDs

Use upstream dataset IDs where available. For generated chunks, call `make_evidence_id(document_id, chunk_index)`, which emits `<document_id>::<chunk_index>`; do not use random UUIDs. For experiments, `make_experiment_id(...)` emits `<question_id>::<condition>::<system_variant>::<run_id>`. `run_id` distinguishes repeated runs. Include dataset revision and source context in metadata as needed.

JSONL and JSON Schema snapshots are described in `schemas/README.md`; all stored values use the enum strings above.
