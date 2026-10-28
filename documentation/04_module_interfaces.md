# Module interfaces

The structural interfaces live in `contracts/interfaces.py` as Python Protocols. They intentionally define inputs/outputs without binding a workstream to a concrete library or device.

## Ownership and signatures

| Workstream | Contract | Method | Expected output |
|---|---|---|---|
| Dataset / retrieval | `Retriever` | `retrieve(request: RetrievalRequest) -> RetrievalResult` | At most `top_k` chunks, preserving question ID, query, stage, and stable evidence IDs. |
| Sufficiency | `SufficiencyAssessor` | `assess(question: QuestionRecord, evidence: EvidenceSet) -> SufficiencyDecision` | Concise validated status; missing fields/query present only for insufficiency. |
| Answering | `AnswerGenerator` | `generate(question: QuestionRecord, evidence: EvidenceSet) -> AnswerResult` | Grounded answer with used evidence IDs or typed abstention. |
| Adaptive / evaluation | `AdaptiveController` | `run(question, initial_evidence, *, retriever, assessor, answer_generator) -> ExperimentRecord` | Raw, internally consistent record for one case. |

The controller receives implementations through interfaces. It must not know whether retrieval uses cosine similarity, FAISS, or another index, or whether a model uses MLX, Transformers, or a mock. The contract establishes no more than two total retrieval calls. This foundation does not choose when or how the follow-up query is generated beyond the structured decision fields.

## Integration rules

- Keep each member's internal types behind its adapter and exchange canonical records at module boundaries.
- Parse and validate external/model output before it becomes a contract object.
- Attach `retrieval_stage` and rank/score when known; never invent a score.
- Preserve stable IDs through merges and record the evidence used to answer.
- Use mocks implementing the same Protocols so work can proceed before other modules exist.
- A shared field or signature change needs team agreement and a matching docs/schema/test update.
