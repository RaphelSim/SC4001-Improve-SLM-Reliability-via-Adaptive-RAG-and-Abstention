# Operational definitions

These definitions apply to the controlled experiment and to the corresponding serialized fields. **Gold annotations are an experimental operational definition:** they provide a consistent label for this study and may not represent every possible alternative valid reasoning path.

## Evidence states

- **Complete evidence:** all gold supporting evidence required by the constructed evaluation instance is available to the system.
- **Partially incomplete evidence:** at least one required gold supporting item is deliberately absent while at least one other required item remains available.
- **Fully incomplete evidence:** all required gold supporting items are absent from the evidence initially supplied to the SLM.
- **Distractor-heavy evidence:** the evidence set includes irrelevant candidate material alongside the support material. Any numeric distractor ratio or cutoff is an experimental choice to be selected and documented using validation data; the enum alone does not encode a threshold.

## Sufficiency and support

- **Sufficient evidence:** evidence judged sufficient to support a final answer under the project's operational evaluation protocol.
- **`gold_sufficient`:** a controlled label derived from gold-support construction. In the foundation contract it is true when every `gold_support_id` is present in the relevant evidence state.
- **`model_predicted_sufficient`:** the model's own decision, derived from `SufficiencyDecision.status`. Keep it separate from `gold_sufficient`; never infer one from the other.
- **Unsupported answer:** a non-abstained final answer produced when the final controlled evidence state does not satisfy the operational sufficient-evidence condition. This measures evidence support, not whether the answer text happens to match the gold answer. A factually correct answer can still be unsupported by the provided evidence.

## Abstention and recovery

- **Abstention:** the system intentionally declines to provide a substantive answer because available evidence remains insufficient. Store `answer: null` internally and a typed `abstention_reason`; presentation code may render `INSUFFICIENT EVIDENCE` later.
- **Correct abstention:** the model abstains when the final controlled evidence state is insufficient.
- **False abstention / over-abstention:** the model abstains even though sufficient controlled evidence is available.
- **Evidence recovery:** initially missing required supporting evidence is retrieved during an adaptive follow-up retrieval round.
- **Recovery rate:** among initially incomplete cases eligible for recovery, count cases where all initially missing required support items are recovered, divided by all initially incomplete recoverable cases. Eligibility means the missing support items are present in the permitted retrieval corpus under the predeclared construction; retrieval failure after that point remains in the denominator.

## Retrieval rounds

One retrieval round is one retriever invocation using one query. Round 0 is initial retrieval. Round 1 is adaptive follow-up retrieval. The core project allows at most one adaptive follow-up, for a maximum of two total rounds.
