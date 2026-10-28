# Decision log

Use this ADR-like format for new or revised shared choices:

```text
Decision ID:
Date:
Status:
Context:
Decision:
Alternatives:
Consequences:
```

## Initial decisions

### D-001 — Python 3.11 baseline

Decision ID: D-001  
Date: 2026-10-28  
Status: Accepted  
Context: Members need a common local and Colab runtime.  
Decision: Use Python 3.11 for shared development and experiments.  
Alternatives: Allow per-member Python versions.  
Consequences: Optional native runtimes and dependency snapshots must target this baseline.

### D-002 — Pydantic v2 contracts

Decision ID: D-002  
Date: 2026-10-28  
Status: Accepted  
Context: Modules need explicit, validated serialized handoffs.  
Decision: Use Pydantic v2 models and generated JSON Schema for shared records.  
Alternatives: Untyped dictionaries or dataclasses.  
Consequences: Schema changes are API changes and require coordinated updates.

### D-003 — JSONL handoff records

Decision ID: D-003  
Date: 2026-10-28  
Status: Accepted  
Context: Raw per-case outputs should be streamable and inspectable.  
Decision: Use one validated record per UTF-8 JSONL line for cache/handoff records.  
Alternatives: One large JSON array or a database.  
Consequences: Malformed lines fail with line-numbered errors; there is no server requirement.

### D-004 — Provisional Top-k defaults

Decision ID: D-004  
Date: 2026-10-28  
Status: Accepted  
Context: Development needs executable defaults before tuning.  
Decision: Set initial and follow-up Top-k to 3 as development defaults only. Select final values on validation data.  
Alternatives: Leave config unusable until tuning.  
Consequences: Defaults must not be described as experimentally validated.

### D-005 — Two maximum retrieval rounds

Decision ID: D-005  
Date: 2026-10-28  
Status: Accepted  
Context: The intended study permits one adaptive follow-up.  
Decision: Round 0 is initial retrieval and round 1 is the only follow-up.  
Alternatives: Unbounded iterative retrieval.  
Consequences: Records/config enforce a maximum of two total retrieval invocations.

### D-006 — No GraphRAG in core scope

Decision ID: D-006  
Date: 2026-10-28  
Status: Accepted  
Context: The research question targets adaptive evidence recovery and abstention.  
Decision: Do not include GraphRAG in the initial core experiment.  
Alternatives: Add graph indexing as another retrieval condition.  
Consequences: Keeps the initial comparison focused and integration smaller.

### D-007 — No initial SLM fine-tuning

Decision ID: D-007  
Date: 2026-10-28  
Status: Accepted  
Context: The initial study compares behavior under evidence conditions.  
Decision: Do not fine-tune the SLM in the initial core experiment.  
Alternatives: Add task-specific model training.  
Consequences: Model behavior is evaluated with frozen inference settings.

### D-008 — Reliability is primary

Decision ID: D-008  
Date: 2026-10-28  
Status: Accepted  
Context: Inference cost is not the principal research question.  
Decision: Prioritize evidence support, correct abstention, recovery, and answer quality; treat inference cost as secondary.  
Alternatives: Optimize runtime cost as the primary outcome.  
Consequences: Report cost as supplementary context if measured.

### D-009 — Hold out test until freeze

Decision ID: D-009  
Date: 2026-10-28  
Status: Accepted  
Context: Test results must remain independent of design/tuning.  
Decision: Keep test data untouched until architecture and settings are frozen; do not change settings after inspecting test results.  
Alternatives: Use test feedback for iterative tuning.  
Consequences: Final run settings and prompt versions need a recorded freeze point.
