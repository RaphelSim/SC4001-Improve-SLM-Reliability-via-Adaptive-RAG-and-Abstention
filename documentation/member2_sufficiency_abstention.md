# Member 2: evidence sufficiency and abstention

## Responsibility

Member 2 receives a shared `QuestionRecord` and the current `EvidenceSet`, asks whether the answer is supported by those supplied records, and returns the shared `SufficiencyDecision`. When sufficient, a grounded answer is generated as an `AnswerResult`. For the `RAG + Abstention` baseline, `NEED_MORE_EVIDENCE` returns a typed abstention. This layer does not fetch or merge evidence.

The detector judges **support in the supplied evidence**, not whether a model knows the answer from pretraining. For example, knowing that Paris is the capital does not make the evidence sufficient when the only supplied text says France is in Western Europe. Gold support IDs and evidence-condition metadata are deliberately omitted from both model prompts to avoid leaking the target label.

## Implementation map

- `src/reliable_rag/model/slm.py` — backend-neutral `SLMClient` Protocol and optional lazy MLX-LM adapter.
- `src/reliable_rag/model/sufficiency.py` — `SLMSufficiencyAssessor`, strict parser, supported-ID checks, and one optional formatting retry.
- `src/reliable_rag/model/answer.py` — `SLMAnswerGenerator`, strict `AnswerResult` parser, and used-evidence checks.
- `src/reliable_rag/model/abstention.py` — `RAGAbstentionResponder`; maps insufficient evidence or malformed model output to a shared typed abstention.
- `src/reliable_rag/model/structured_output.py` and `errors.py` — strict JSON parsing, prompt rendering, repair prompt construction, and structured `ModelOutputError`.
- `prompts/sufficiency_v1.txt`, `prompts/final_answer_v1.txt`, `prompts/structured_retry_v1.txt` — versioned baseline task and formatting prompts.
- `tests/fixtures/member2_sufficiency_cases.json` — synthetic sufficiency cases; not experimental data.

The model logic depends only on the injectable `SLMClient` and the existing shared contracts. Tests use a queued fake client and never load MLX, Transformers, or model weights. The assessor stores only client/config/template references and does not retain per-question evidence or call a retriever, so Member 3 can reuse it for another `EvidenceSet`.

## Structured output and failure handling

The sufficiency prompt requests exactly the shared keys: `question_id`, `status`, `missing_information`, `search_query`, and `supporting_evidence_ids`. Serialized `SufficiencyStatus` values are lowercase: `sufficient` and `need_more_evidence`. A sufficient result needs at least one supporting ID from the input and a null search query. An insufficient result needs non-empty missing-information and search-query text. IDs not present in the current `EvidenceSet`, duplicate JSON keys, extra/missing fields, non-JSON text, wrong question IDs, and Pydantic-invalid content are rejected.

If parsing or validation fails, config permits at most one formatting-only retry. The retry sees the expected JSON Schema plus the invalid response encoded as data; it is told not to re-answer the task or invent values. A second failure raises `ModelOutputError`, which contains `question_id`, task, attempt count, the shared `INVALID_MODEL_OUTPUT` reason, and a safe `as_dict()` representation. Raw prompts/responses are not logged. `RAGAbstentionResponder` maps that error to `AnswerResult(answer=None, abstained=True, abstention_reason=INVALID_MODEL_OUTPUT)`.

## Abstention behavior

```python
decision = assessor.assess(question, evidence)

if decision.status == SufficiencyStatus.NEED_MORE_EVIDENCE:
    return AnswerResult(
        question_id=question.question_id,
        answer=None,
        abstained=True,
        abstention_reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
        used_evidence_ids=[],
    )

return answer_generator.generate(question, evidence)
```

`RAGAbstentionResponder.answer(...)` implements this baseline. It does not call the answer generator for `NEED_MORE_EVIDENCE`; it returns an internal null answer, not prose such as “I don't know.” If the structured output remains invalid, it uses `INVALID_MODEL_OUTPUT` instead.

## Member 3 handoff

For adaptive retrieval, Member 3 should call the assessor and consume its structured result directly:

```python
decision = assessor.assess(question, current_evidence)
if decision.status == SufficiencyStatus.NEED_MORE_EVIDENCE:
    query = decision.search_query
    missing = decision.missing_information
    # Member 3 retrieves `query`, merges evidence, then calls assess(...) again.
```

The query and missing-information summary are ordinary shared fields, not text to parse from prose. The assessor is independent of round number and retrieval backend. Member 3 owns follow-up invocation, evidence merge/deduplication, and controller behavior.

## Configuration and local SLM diagnostic

Member 2 reads `Settings.slm`: model ID, runtime, quantization label, input/output token limits, temperature, sampling flag, prompt version/path, and format retry limit. The prompt version defaults to `v1`; paths default to the three checked-in templates. `format_retry_limit` is validated to 0 or 1.

The optional `MLXLMClient` imports MLX-LM lazily and supports deterministic greedy generation when `do_sample: false`. Unit tests never call it. It enforces `max_input_tokens` without truncating evidence. Create a local config from the baseline, set `slm.runtime: mlx` and a suitable MLX model ID/path, and keep that machine-specific config ignored:

```bash
cp configs/base.yaml configs/local.yaml
# Edit configs/local.yaml: set slm.runtime and slm.model_id.
pip install -e ".[dev,mac]"
python scripts/test_sufficiency_model.py --config configs/local.yaml
```

A local directory model needs no download flag. If `model_id` is a remote model repository, append `--allow-model-download` only when you intend to fetch it. The script runs the five synthetic examples and prints gold sufficiency, predicted status, missing information, and the search query. It is manual diagnostics, not an experiment runner.

The shared `model_id` remains `TBD`; no model has been selected or downloaded as part of this implementation. MLX-LM's Python API loads a model/tokenizer and generates from a prompt; the adapter passes its configured token cap and sampler without exposing that backend to the assessor. [Official MLX-LM Python API](https://github.com/ml-explore/mlx-lm/blob/main/README.md#python-api)

## Member 2 metrics

`reliable_rag.evaluation.metrics` is the shared metrics module. It treats `NEED_MORE_EVIDENCE` as positive and reports TP/FP/TN/FN, accuracy, precision, recall, F1, false-sufficiency rate, and false-insufficiency rate. Existing answer coverage, supported-evidence recall, unsupported-answer, correct-abstention, and false-abstention rate functions are reused rather than duplicated.

- **False sufficiency:** gold evidence is insufficient, but the model predicted `SUFFICIENT` (FN). This is the risky error because it can lead to unsupported answering.
- **False insufficiency:** gold evidence is sufficient, but the model predicted `NEED_MORE_EVIDENCE` (FP). This can lower answer coverage or cause an unnecessary retrieval attempt.
- **Correct abstention:** final evidence is gold-insufficient and the final answer abstains.
- **False abstention:** final evidence is gold-sufficient and the final answer abstains.

No test-set tuning is permitted. Prompt and model choices must be selected on validation data and frozen before test evaluation.

## Tests and limitations

Run focused and full tests with:

```bash
pytest tests/model tests/evaluation
pytest
ruff check .
ruff format --check .
```

Tests cover strict sufficiency/answer parsing, question/evidence ID checks, one retry, fake-client behavior, abstention gating, metric labels, and the France parametric-knowledge trap. No real SLM result is asserted. Prompt v1 wording still needs validation-data tuning; free-form language models may misjudge support even with the evidence-only instruction. The parser verifies structure and provenance, not semantic entailment. Long evidence that exceeds the configured input length fails clearly rather than silently dropping context.
