# MuSiQue evidence-complexity experiment

This experiment uses the existing MiniLM retriever, fixed one-shot baseline runner, Adaptive V3 Top-2-unseen controller, evaluator, and result schema. It changes the dataset preparation only. The 100 selected questions come from MuSiQue-Answerable train with seed 88; the original `is_supporting` paragraph flags define gold evidence. No gold IDs or gold counts enter the V3 model prompt.

## Source and frozen sample

The requested `StonyBrookNLP/musique` Hugging Face repository returned HTTP 401. The original-format `musique_ans_v1.0_train.jsonl` was downloaded from the public `voidful/MuSiQue` mirror at commit `a7d9f9adf6191604fc67cde318ee1a86fcf7babc`. The local source SHA-256 is recorded in `data/processed/musique/subset-100-seed88/subset-development.json`, together with the selected source indices, question IDs, gold IDs, distractor IDs, original paragraph indices, and final context order. The raw download stays in `data/raw/`; the frozen processed sample is versionable.

The sample contains 34 questions with two gold paragraphs, 33 with three, and 33 with four. Distractor counts are randomly drawn within low (6–9), medium (10–13), and high (14–18) bands, with 33/33/34 questions respectively. Counts are constrained by the original candidate pool: a 20-paragraph question with four gold paragraphs can have at most 16 original distractors. No support labels are changed or fabricated. Each question keeps every original gold paragraph, has unique candidate IDs, and has a deterministically shuffled context.

The observed distractor-count distribution is:

| Distractors | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Questions | 8 | 7 | 9 | 9 | 8 | 10 | 7 | 8 | 10 | 9 | 4 | 7 | 4 |

## Reproduce preparation and runs

From the repository root with `.venv` activated, obtain the pinned MuSiQue answerable train JSONL and save it as `data/raw/musique_ans_v1.0_train.jsonl`. The following uses the public mirror and records its exact commit:

```bash
python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="voidful/MuSiQue",
    repo_type="dataset",
    filename="musique_ans_v1.0_train.jsonl",
    revision="a7d9f9adf6191604fc67cde318ee1a86fcf7babc",
    local_dir="data/raw",
)
PY
```

Then run:

```bash
python scripts/prepare_musique.py --input data/raw/musique_ans_v1.0_train.jsonl --source-repo voidful/MuSiQue --revision a7d9f9adf6191604fc67cde318ee1a86fcf7babc
python scripts/build_embedding_index.py --config configs/musique_retrieval_study_v3.yaml --subset-size 100 --seed 88
```

Preparation refuses to replace a mismatched frozen sample unless `--force` is explicit. The index is separate from HotpotQA and validates the ordered MuSiQue corpus and MiniLM revision. Each run below refuses to overwrite an existing output directory; use a new run ID for repetitions.

```bash
python scripts/run_retrieval_study.py --config configs/musique_retrieval_study_v3.yaml --mode baseline --sample-size 100 --pool-size 100 --seed 88 --baseline-top-k 2 --run-id baseline_top2_musique_n100_seed88

for k in 3 4 5 6 7 8; do
  python scripts/run_retrieval_study.py --config configs/musique_retrieval_study_v3.yaml --mode baseline --sample-size 100 --pool-size 100 --seed 88 --baseline-top-k "$k" --reuse-question-ids-from outputs/baseline_top2_musique_n100_seed88/run_metadata.json --run-id "baseline_top${k}_musique_n100_seed88"
done

python scripts/run_retrieval_study.py --config configs/musique_retrieval_study_v3.yaml --mode adaptive --sample-size 100 --pool-size 100 --seed 88 --adaptive-top-k 2 --max-rounds 5 --reuse-question-ids-from outputs/baseline_top2_musique_n100_seed88/run_metadata.json --run-id adaptive_v3_musique_n100_seed88

python scripts/compare_musique_variation.py
```

All methods read the one frozen `questions-development.jsonl` and `evidence-development.jsonl`. Fixed runs call MiniLM once using the original question and vary only `initial_top_k`. V3 uses the unchanged Qwen3-4B MLX 4-bit model, prompt, Top-2 unseen policy, cumulative unique evidence, and five-round cap. The comparison script verifies matching ordered IDs, gold and candidate IDs, source/index hashes, baseline one-round behavior, and Top-k ranking prefixes before writing metrics.

Top-2 selects the complete frozen 100-question pool. Top-3 through Top-8 and V3 explicitly reuse the Top-2 run's ordered ID list.

## Outputs and interpretation

Each run writes the existing `results.jsonl`, `run_metadata.json`, `summary.json`, `summary.csv`, `errors.jsonl`, and `events.jsonl` files under `outputs/<run-id>/`. The original per-question and summary field names are unchanged. `scripts/compare_musique_variation.py` writes a separate `outputs/musique_n100_seed88_comparison/comparison.json`, `comparison.csv`, `breakdowns.json`, and `breakdowns.csv`. The breakdown groups results by original gold count, distractor band, and total candidate count without changing a run's summary schema.

Gold evidence percentage remains the per-question fraction of **that question's** gold paragraphs retrieved. Full recovery requires all two, three, or four original supporting paragraphs, as applicable. Unique noise is the fraction of distinct retrieved IDs outside that question's gold set. Raw paragraphs count returned instances; V3 ranking hits skipped because they were seen earlier do not count as returns.

## Measured results: 100 questions per system

All eight runs completed with 100 result rows and zero experiment errors. The table below summarizes the saved `comparison.csv`; percentages are means of per-question percentages except the full/zero rates. Fixed systems made no reasoning-model calls.

| Metric | Top-2 | Top-3 | Top-4 | Top-5 | Top-6 | Top-7 | Top-8 | V3 Top-2 unseen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mean gold evidence | 40.25% | 50.83% | 57.00% | 63.17% | 68.08% | 74.08% | 78.08% | 55.75% |
| Median gold evidence | 33.33% | 50.00% | 50.00% | 66.67% | 66.67% | 75.00% | 75.00% | 50.00% |
| Full gold recovery | 7% | 15% | 19% | 25% | 30% | 39% | 46% | 25% |
| Zero gold recovery | 15% | 10% | 9% | 5% | 3% | 2% | 2% | 12% |
| Mean unique noise | 43.00% | 51.67% | 58.75% | 63.40% | 67.17% | 69.14% | 71.25% | 53.83% |
| Mean raw paragraphs | 2.00 | 3.00 | 4.00 | 5.00 | 6.00 | 7.00 | 8.00 | 3.60 |
| Mean unique paragraphs | 2.00 | 3.00 | 4.00 | 5.00 | 6.00 | 7.00 | 8.00 | 3.60 |
| Mean retrieval rounds | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.80 |
| Mean duplicate retrievals | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Duplicate share of raw | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| Final need-more status | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 96 |
| Experiment errors | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

V3 returned 3.60 paragraphs on average, so Top-4 is its closest fixed paragraph budget. V3 had 1.25 percentage points **less** mean gold evidence than Top-4, but six points more full-gold recovery and 4.92 points less mean noise. It also had three points more zero-gold questions and incurred 2.83 reasoning-model calls per question on average (1.80 normal controller and 1.03 query-regeneration calls). Its stop reasons were 96 `repeated_query` and four `sufficient`; 72 repeated-query stops still lacked complete annotated gold evidence. The low gold-based false-sufficiency rate, 3/151 incomplete-evidence decisions, must be read alongside this high repeated-query rate: the controller rarely chose a sufficient stop.

The separate `breakdowns.csv` and `.json` give all eight systems by gold count, distractor band, and total candidate count. For V3 alone:

| Group | Questions | Mean gold evidence | Full recovery | Mean noise |
|---|---:|---:|---:|---:|
| 2 gold | 34 | 67.65% | 52.9% | 55.15% |
| 3 gold | 33 | 51.52% | 15.2% | 58.84% |
| 4 gold | 33 | 47.73% | 6.1% | 47.47% |
| Low distractors (6–9) | 33 | 55.81% | 27.3% | 58.59% |
| Medium distractors (10–13) | 33 | 63.38% | 24.2% | 42.93% |
| High distractors (14–18) | 34 | 48.28% | 23.5% | 59.80% |

These groups describe this one frozen sample; gold count and distractor density are not independently randomized interventions.

The fixed Top-2 through Top-8 and V3 results use a different dataset and should not be combined with the canonical HotpotQA 100-question comparison. See [the HotpotQA experiment guide](EXPERIMENT_ARCHITECTURE_AND_RESULTS.md) for that separate study.
