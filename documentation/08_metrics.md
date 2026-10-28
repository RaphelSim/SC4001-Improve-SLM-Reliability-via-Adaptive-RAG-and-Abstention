# Metric definitions

Metrics should be computed from raw `ExperimentRecord` observations after the per-run records are validated. Keep aggregate values out of each record. `supporting_evidence_recall_at_k`, `answer_coverage`, and the two unsupported-answer rates below have small implementations in `evaluation/metrics.py`; other metrics remain protocol definitions until their implementation and normalization choices are agreed. Freeze those choices using validation data before test evaluation.

Let (N) be the evaluated case set, (A_i) a model answer, (G_i) the gold answer, (S_i) the final controlled evidence sufficiency label, and (P_i) the model's predicted sufficiency. Define `gold_sufficient` as true/false and the insufficiency-positive target as `NEED_MORE_EVIDENCE`.

## Answer quality

- **Exact Match (EM):** for answered set (A), `EM_answered = (1 / |A|) Σᵢ∈A 1[normalize(Aᵢ) = normalize(Gᵢ)]`. Report abstentions separately. If using all cases (N), score abstentions as zero and label it coverage-adjusted EM: `EM_all = (1 / |N|) Σᵢ∈N 1[not abstainedᵢ and normalize(Aᵢ) = normalize(Gᵢ)]`. Do not hide the denominator.
- **Answer F1:** normalize prediction/gold into token sequences. For each answered case, `Pᵢ = overlapᵢ / predicted_tokensᵢ`, `Rᵢ = overlapᵢ / gold_tokensᵢ`, `F1ᵢ = 2PᵢRᵢ / (Pᵢ + Rᵢ)` (define zero-overlap/empty-token behavior as zero). Average `F1ᵢ` over answered cases. If reporting an all-case score, score abstention as zero and label it coverage-adjusted F1. Use a declared normalization procedure consistently.

## Retrieval and sufficiency

- **Supporting Evidence Recall@k:** for a retrieval result with ranked top-k items, `recallᵢ@k = |top_k_idsᵢ ∩ gold_support_idsᵢ| / |gold_support_idsᵢ|`. Report `mean_i(recallᵢ@k)` over evaluated questions and specify the retrieval stage (initial or follow-up); empty support sets are invalid in this project.
- **Sufficiency Detection Accuracy:** `number of correct SufficiencyDecision.status labels / number of decisions included`. Report initial and post-recovery decisions separately where applicable.
- **Sufficiency Detection Precision / Recall / F1:** treat `NEED_MORE_EVIDENCE` as the positive class. `precision = TP/(TP+FP)`, `recall = TP/(TP+FN)`, and `F1 = 2PR/(P+R)`, where TP means the model predicted insufficient and controlled gold evidence is insufficient. State the aggregation method if macro/weighted variants are added.

## Recovery and safety

- **Recovery Rate:** among initially incomplete cases eligible for recovery, `number where all initially missing required support IDs appear in follow-up results / number of initially incomplete cases constructed as recoverable`. Eligibility means missing support items are present in the allowed corpus. Failed follow-up retrieval stays in the denominator.
- **Unsupported Answer Rate (controlled insufficient denominator):** `count(not abstained and final evidence insufficient) / count(final evidence insufficient)`.
- **Unsupported Answer Rate among answered cases:** `count(not abstained and final evidence insufficient) / count(not abstained)`. Report alongside the controlled-insufficient-denominator rate; the denominator is different.
- **Correct Abstention Rate:** `count(abstained and final evidence insufficient) / count(final evidence insufficient)`.
- **False Abstention Rate / Over-abstention Rate:** `count(abstained and final evidence sufficient) / count(final evidence sufficient)`.
- **Answer Coverage:** `count(not abstained) / count(all evaluated cases)`.

For any zero denominator, report `N/A` with the numerator and denominator rather than silently substituting zero. These are definitions only; the final metric implementation and any answer normalization remain future work.
