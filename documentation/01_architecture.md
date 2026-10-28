# Architecture

## Package responsibilities

```text
src/reliable_rag/
├── contracts/   Enums, Pydantic records, IDs, and Protocol interfaces
├── config/      YAML parsing and typed shared settings
├── data/        Future dataset and split handling
├── retrieval/   Future retriever implementations
├── model/       Future SLM runtime, sufficiency, and answer adapters
├── adaptive/    Future controller implementation
├── evaluation/  Future metric implementation after protocol freeze
└── io/          Validated JSONL exchange
```

`scripts/` contains small command-line checks. `configs/` holds shared settings. `schemas/` holds generated representations of persisted Pydantic contracts. Tests use only a tiny synthetic fixture and make no research claim.

## Data flow

```text
QuestionRecord
     ↓
Retriever.retrieve(RetrievalRequest) → RetrievalResult
     ↓                                  ↓
EvidenceSet → SufficiencyAssessor.assess(QuestionRecord, EvidenceSet)
                                      ↓
               sufficient ────────────┴────────→ AnswerGenerator.generate(...)
               need more → targeted request → Retriever (one follow-up maximum)
                                      ↓
                       reassess merged EvidenceSet
                         ├─ sufficient → answer
                         └─ insufficient → abstain
                                      ↓
                              ExperimentRecord
```

`round 0` is initial retrieval; `round 1` is the only permitted adaptive follow-up. A `RetrievalRequest` is one invocation. `EvidenceSet` represents exactly what is visible to the SLM at that point, and its `gold_sufficient` property is derived from controlled support-ID coverage. The model's assessment is stored separately in `SufficiencyDecision`.

## Dependency direction

```text
Retriever implementation ───── implements ───→ Retriever Protocol
SLM sufficiency implementation implements ───→ SufficiencyAssessor Protocol
Answer implementation ──────── implements ───→ AnswerGenerator Protocol

Adaptive controller ────────── depends only on the three Protocols
```

Concrete implementations depend on shared contracts. The controller depends on abstractions, not a particular embedding library, index, model runtime, or device. This dependency inversion allows members to work independently and use mocks.

## Adaptation boundary

An adaptive controller may inspect a structured insufficiency decision and pass its search query to `Retriever`. It may merge and deduplicate returned evidence according to the agreed data policy. The contract establishes the maximum of two retrieval rounds, but the research algorithm and prompt wording are intentionally unimplemented.
