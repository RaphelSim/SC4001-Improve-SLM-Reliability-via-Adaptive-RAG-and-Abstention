# Integration and handoffs

## Member boundaries

```text
Member 1: dataset / evidence construction / retrieval
                  ↓ RetrievalResult + EvidenceSet
Member 2: sufficiency assessment / abstention / answering
                  ↓ SufficiencyDecision + AnswerResult
Member 3: adaptive controller / experiment recording / evaluation
                  ↓ ExperimentRecord
```

These boundaries are logical ownership areas rather than restrictions on internal code. Each member can build mocks implementing `Retriever`, `SufficiencyAssessor`, and `AnswerGenerator` from `contracts/interfaces.py`, then integrate through the records.

## Small contract examples

The synthetic fixture is not experimental data. It contains these major records:

**EvidenceChunk**

```json
{"evidence_id":"mack-rides::0","document_id":"mack-rides","title":"Mack Rides","text":"Mack Rides is headquartered in Germany.","retrieval_score":0.88,"retrieval_rank":1,"retrieval_stage":"follow_up"}
```

**QuestionRecord**

```json
{"question_id":"lost-gravity-country","question":"What country is the manufacturer of Lost Gravity based in?","gold_answer":"Germany","split":"development","gold_support_ids":["lost-gravity::0","mack-rides::0"],"candidate_evidence_ids":["lost-gravity::0","mack-rides::0","park-distractor::0"],"metadata":{"fixture_only":true}}
```

**EvidenceSet (complete)**

```json
{"question_id":"lost-gravity-country","condition":"complete","gold_support_ids":["lost-gravity::0","mack-rides::0"],"evidence":[{"evidence_id":"lost-gravity::0","document_id":"lost-gravity","title":"Lost Gravity","text":"Lost Gravity was manufactured by Mack Rides."},{"evidence_id":"mack-rides::0","document_id":"mack-rides","title":"Mack Rides","text":"Mack Rides is headquartered in Germany."}]}
```

**RetrievalRequest / RetrievalResult**

```json
{"question_id":"lost-gravity-country","query":"Mack Rides headquarters country","stage":"follow_up","top_k":3,"exclude_evidence_ids":["lost-gravity::0"]}
```

```json
{"question_id":"lost-gravity-country","query":"Mack Rides headquarters country","stage":"follow_up","top_k":3,"evidence":[{"evidence_id":"mack-rides::0","document_id":"mack-rides","title":"Mack Rides","text":"Mack Rides is headquartered in Germany.","retrieval_score":0.88,"retrieval_rank":1,"retrieval_stage":"follow_up"}]}
```

**SufficiencyDecision**

```json
{"question_id":"lost-gravity-country","status":"need_more_evidence","missing_information":"The country where Mack Rides is based.","search_query":"Mack Rides headquarters country","supporting_evidence_ids":["lost-gravity::0"]}
```

**AnswerResult (abstention form)**

```json
{"question_id":"lost-gravity-country","answer":null,"abstained":true,"abstention_reason":"insufficient_evidence","used_evidence_ids":["lost-gravity::0"]}
```

**AdaptiveRoundRecord**

```json
{"round_index":1,"retrieval":{"question_id":"lost-gravity-country","query":"Mack Rides headquarters country","stage":"follow_up","top_k":3,"evidence":[{"evidence_id":"mack-rides::0","document_id":"mack-rides","title":"Mack Rides","text":"Mack Rides is headquartered in Germany."}]},"sufficiency_decision":{"question_id":"lost-gravity-country","status":"sufficient","missing_information":null,"search_query":null,"supporting_evidence_ids":["lost-gravity::0","mack-rides::0"]}}
```

## End-to-end ExperimentRecord

`tests/fixtures/example_experiment.jsonl` is a complete, machine-validated `ExperimentRecord`. It demonstrates partial evidence at round 0 (evidence A only), a targeted follow-up that recovers evidence B, reassessment as sufficient, and the final answer `Germany`. Its `fixture_only` metadata is deliberate: this is a contract/handoff example, not a dataset sample or experimental result. Validate it with:

```bash
python scripts/validate_jsonl.py --schema experiment-record tests/fixtures/example_experiment.jsonl
```

## Recovery walkthrough

`complete_evidence.json` contains evidence A+B. `partial_evidence.json` contains A only. The end-to-end JSONL record captures adding B during round 1 and then answering. All evidence, queries, decisions, and answer IDs retain the stable question/evidence IDs.
