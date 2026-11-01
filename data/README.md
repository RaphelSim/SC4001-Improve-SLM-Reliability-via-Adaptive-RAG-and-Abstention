# Data directory policy

- `raw/` contains downloaded/original data. It is Git-ignored.
- `processed/` contains derived datasets. It is Git-ignored except deliberately tiny test fixtures kept under `tests/fixtures/`.
- `splits/` contains split manifests. Lightweight deterministic question-ID manifests may be version controlled when dataset licensing permits; include dataset revision and generation details. Large or licensed data must not be committed.

Do not place model weights, credentials, or experimental output here. Keep paths relative and machine independent in shared configuration.

Member 1 normalizes HotpotQA paragraphs into `QuestionRecord` and `EvidenceChunk`
objects through `reliable_rag.data.hotpotqa`. The complete paragraph collection is
the retrieval corpus; incomplete `EvidenceSet` conditions only control what is
initially shown to the model. `reliable_rag.data.splits` can create deterministic
ID-only manifests using the shared project seed.

## HotpotQA preparation

The preparation command downloads one Hugging Face split or accepts a local
HotpotQA JSON/JSONL export. It writes canonical question and evidence JSONL plus
an ID-only manifest:

```bash
PYTHONPATH=src python scripts/prepare_hotpotqa.py \
  --dataset-name hotpotqa/hotpot_qa \
  --dataset-config distractor \
  --source-split train \
  --project-split development \
  --dataset-revision hotpotqa-hf \
  --output-dir data/processed/hotpotqa \
  --raw-output data/raw/hotpotqa-train.json
```

Run the command separately for the official validation split and, after the
experiment freeze, the held-out test split. For a bounded smoke test, use
`--source-split 'train[:100]'`. The raw data and processed outputs are ignored
by Git. The resulting `questions-*.jsonl` and `evidence-*.jsonl` files can be
checked with `scripts/validate_jsonl.py`.
