# Setup, frozen data, and experiments

The research question is whether adaptive multi-round retrieval obtains more required HotpotQA paragraphs while adding less irrelevant unique evidence than single-shot Top-k retrieval. The main experiment does not generate answers or score abstention. The baseline retrieves Top-5 once without a reasoning model. Adaptive retrieval asks a Qwen3-4B MLX controller to inspect unique accumulated paragraphs, stop if sufficient, or request another targeted Top-2 retrieval, for at most five rounds.

## Environment

From the repository root on an Apple Silicon Mac with native ARM Python 3.11 and Metal access:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,mac]'
python scripts/check_environment.py --check-mlx
python scripts/prepare_models.py --config configs/retrieval_study.yaml
```

MiniLM and Qwen3-4B MLX 4-bit are public. A Hugging Face token is normally unnecessary. Model snapshots are pinned in `configs/retrieval_study.yaml`; the model ID and revision can be overridden for another compatible MLX checkpoint. `scripts/prepare_models.py --offline` checks cached snapshots. Data preparation and index creation do not need MLX.

## Phase 1: freeze the 1,000-question pool

```bash
python scripts/prepare_hotpotqa.py --config configs/retrieval_study.yaml --subset-size 1000 --seed 88
python scripts/build_embedding_index.py --config configs/retrieval_study.yaml --subset-size 1000 --seed 88
```

The pinned HotpotQA distractor/train revision is sampled with Python's seed-88 shuffled-prefix method. `data/processed/hotpotqa/subset-1000-seed88/` stores question JSONL, evidence JSONL, an ID manifest, and `subset-development.json` with source revision/fingerprint, selected source indices and IDs, preprocessing version, sampling settings, and artifact SHA-256 hashes. Every selected question retains its candidate context, original gold answer/support references, and frozen `gold_paragraph_ids`. A paragraph is gold when it contains at least one annotated supporting sentence. The prepared 1,000 questions are all unique; the first 100 in saved order are the Phase 3 cohort. No second sampling occurs for Phase 3.

`cache/hotpotqa-subset-1000-seed88.npz` holds MiniLM paragraph embeddings; its JSON sidecar records model/revision, device, preprocessing representation, corpus fingerprint, dimension, dataset revision, and cosine similarity. The corpus contains all 1,000 questions' candidate paragraphs; retrieval for each question is restricted to its own candidate IDs. Repeated preparation validates hashes and reuses matching data/index. Use `--force` on preparation or index creation only when intentionally rebuilding them. Experiment run IDs are never silently overwritten.

## Phase 2 and Phase 3: 100-question comparison

Run the paired comparison using one implementation for both modes:

```bash
python scripts/run_retrieval_study.py --mode combined --sample-size 100 --pool-size 1000 --seed 88 --baseline-top-k 5 --adaptive-top-k 2 --max-rounds 5 --run-id phase3-n100-seed88-top5-adaptive2x5-v2
```

The command creates `outputs/phase3-n100-seed88-top5-adaptive2x5-v2/` with `baseline/` and `adaptive/` subdirectories, plus aggregate `summary.json`, `comparison.csv`, `paired.jsonl`, and `metadata.json`. Each system directory has `run_metadata.json`, `results.jsonl`, `errors.jsonl`, `events.jsonl`, `summary.json`, and `summary.csv`. All 100 attempted questions get a result row, including typed controller errors with their partial retrieval history. Quality means exclude those error rows and report their count separately.

The baseline and adaptive modes can also run independently:

```bash
python scripts/run_retrieval_study.py --mode baseline --sample-size 100 --run-id baseline-n100-seed88-top5
python scripts/run_retrieval_study.py --mode adaptive --sample-size 100 --run-id adaptive-n100-seed88-top2x5
```

The baseline does not load Qwen. Adaptive uses the configured Qwen3-4B MLX 4-bit model, greedy decoding, thinking disabled, at most 512 output tokens and one structured-output repair. `prompts/retrieval_controller_v1.txt` asks only for evidence sufficiency and a focused follow-up query; `prompts/retrieval_retry_v1.txt` repairs malformed responses. The model prompt gets the question and unique accumulated paragraph ID/title/text, without gold labels, gold answer, or evaluation scores. Every raw model response is saved in the corresponding round record.

## Adaptive V2 controller ablation

V2 preserves the V1 run and baseline. It uses the same pinned Qwen3-4B MLX 4-bit revision, MiniLM index, candidate scopes, Top-2 per round, max five rounds, greedy decoding, and 100 question IDs. The V2 controller prompt requires a list of answer requirements and explicitly grounded facts, sends prior retrieval queries/IDs/statuses as history, prefers concrete bridge-entity queries, and flags stagnation. Gold labels are added only after each controller decision. A normalized repeated-query guard asks the same model for a new query and refuses to spend another retrieval call if regeneration repeats a failed query.

The clean V2.1 prompt uses generic placeholders rather than a sample-specific example. Its exact-sample command was:

```bash
python scripts/run_retrieval_study.py --config configs/retrieval_study_v2_1.yaml --mode adaptive --sample-size 100 --pool-size 1000 --seed 88 --adaptive-top-k 2 --max-rounds 5 --reuse-question-ids-from outputs/phase3-n100-seed88-top5-adaptive2x5-v2/metadata.json --run-id phase3-n100-seed88-top5-adaptive2x5-v4
```

It writes `outputs/phase3-n100-seed88-top5-adaptive2x5-v4/`. The run attempted all 100 frozen IDs; 63 had valid final controller records and 37 were typed controller errors (26 repeated-query regeneration failures and 11 output/requirement validation failures). Retrieval metrics are reported across all 100 attempts because evidence and retrieval cost remain observable when the controller errors; sufficiency metrics use valid controller decisions and show their denominators. The prior `v3` prompt pilot is preserved but excluded from analysis because its prompt included a sample-specific entity phrase from the first question.

Recreate the three-way comparison without rerunning the baseline or V1:

```bash
python scripts/compare_retrieval_studies.py --baseline outputs/phase3-n100-seed88-top5-adaptive2x5-v2/baseline/results.jsonl --adaptive-v1 outputs/phase3-n100-seed88-top5-adaptive2x5-v2/adaptive/results.jsonl --adaptive-v2 outputs/phase3-n100-seed88-top5-adaptive2x5-v4/results.jsonl --output-dir outputs/phase3-n100-seed88-top5-adaptive2x5-v4
```

This creates `three_way_comparison.json`, `three_way_comparison.csv`, and `paired_adaptive_v1_v2.jsonl`. The comparison asserts identical IDs and order across the three result files. The JSON separates all-attempt retrieval metrics, controller metrics for valid V2.1 decisions, and the common 63-question V2.1-valid cohort.

Inspect a completed run:

```bash
python scripts/validate.py jsonl outputs/phase3-n100-seed88-top5-adaptive2x5-v2/baseline/results.jsonl --schema retrieval-study-record
python scripts/validate.py jsonl outputs/phase3-n100-seed88-top5-adaptive2x5-v2/adaptive/results.jsonl --schema retrieval-study-record
cat outputs/phase3-n100-seed88-top5-adaptive2x5-v2/comparison.csv
```

`run_metadata.json` records the resolved configuration, selected question IDs, pool and index hashes, prompt hashes, model ID/revision/quantization/runtime, package versions, Git state, timestamps, runtime, status, and error counts. A completed run ID cannot be reused; choose a new ID to rerun. The main `run_retrieval_study.py` does not offer `--force` because replacing raw experiment observations would undermine provenance. Old `scripts/run_experiment.py` and `scripts/evaluate_results.py` remain available for the historical answer/abstention study only.

## Adaptive V3: same 100 IDs, Top-2 unseen

V3 uses a single-field controller output: `{"search_query": null}` stops and `{"search_query": "targeted query"}` continues. The model receives the original question, all accumulated unique paragraphs, and previous retrieval queries. Python ranks the unchanged question-specific MiniLM candidate corpus and fills each round's two slots with unseen IDs, skipping previously returned IDs without counting them as returned paragraphs. It stops after at most five rounds. Repeated queries receive one regeneration attempt and then end cleanly with `repeated_query` if still repeated; a depleted candidate pool ends with `retrieval_pool_exhausted`. Controller, repair and regeneration calls are logged separately. Reliable token counts are unavailable from the current MLX text-generation interface.

The frozen same-100 V3 experiment command is:

```bash
python scripts/run_retrieval_study.py --config configs/retrieval_study_v3.yaml --mode adaptive --sample-size 100 --pool-size 1000 --seed 88 --adaptive-top-k 2 --max-rounds 5 --reuse-question-ids-from outputs/phase3-n100-seed88-top5-adaptive2x5-v2/metadata.json --run-id phase3-n100-seed88-top2unseen-v3
```

This explicitly reuses the saved ordered IDs. It writes `outputs/phase3-n100-seed88-top2unseen-v3/` with `results.jsonl`, `errors.jsonl`, `events.jsonl`, `summary.json`, `summary.csv`, and `run_metadata.json`. Prompt hashes, model revision, sample manifest hash, index hash, and resolved settings are in metadata. The earlier Top-5, V1 and V2.1 runs are reused, not rerun. Recreate the four-way and paired V2.1/V3 analysis with:

```bash
python scripts/compare_retrieval_studies_v3.py --top5 outputs/phase3-n100-seed88-top5-adaptive2x5-v2/baseline/results.jsonl --v1 outputs/phase3-n100-seed88-top5-adaptive2x5-v2/adaptive/results.jsonl --v2 outputs/phase3-n100-seed88-top5-adaptive2x5-v4/results.jsonl --v3 outputs/phase3-n100-seed88-top2unseen-v3/results.jsonl --output-dir outputs/phase3-n100-seed88-top2unseen-v3
```

This writes `four_way_comparison.json`, `four_way_comparison.csv`, and `paired_v2_v3.jsonl`. The script asserts exactly 100 identical ordered IDs and equal frozen candidate/gold IDs. Retrieval metrics include all 100 attempts, even if a later controller call errors. Gold-based controller diagnostics use completed controller decisions; HotpotQA gold marks one annotated supporting path, so a gold-incomplete stop is an annotation-relative diagnostic. V3 differs from V2.1 in both output interface and unseen retrieval policy, so the comparison does not isolate either change.

The saved V3 run completed with 100 records and no experiment-level errors. Its all-attempt mean gold recovery is 82.0%, full gold recovery 71%, mean unique noise 54.8%, and mean raw/unique returned paragraphs 4.16 each. It has 71 clean `repeated_query` stops, including 20 with incomplete annotated gold evidence. See [System architecture](SYSTEM_ARCHITECTURE.md) for the four-way table and conditional round trajectory. A focused query-regeneration revision should be evaluated on the same 100 IDs before running the command in Phase 4.

## Changing settings

The new runner accepts `--model-id`, `--model-revision`, `--sample-size`, `--pool-size`, `--seed`, `--baseline-top-k`, `--adaptive-top-k`, `--max-rounds`, `--config`, and `--run-id`. For example:

```bash
python scripts/run_retrieval_study.py --mode combined --sample-size 100 --model-id Qwen/Qwen3-1.7B-MLX-4bit --model-revision 21457c6f51ed54a7c16e988c0844db973815c137 --run-id alternate-model-n100
```

Changing `--sample-size` up to 1,000 uses a longer prefix of the same frozen pool and index. To change seed or pool size, run both preparation commands with the new values first, then pass the same values to the study runner and use a new run ID. To change embedding model, edit a copied YAML config, prepare that model snapshot, build its matching index, and run with `--config`. The runtime validates the index against corpus/model/similarity provenance. Changing Top-k or max rounds needs no re-embedding. Future Top-3/5/7 and adaptive Top-1/2/3 sensitivity runs use these flags; that grid is not part of Phase 3.

## Phase 4: frozen 1,000-question experiment

Run this command yourself after reviewing the same-100 V3 results. It uses the same frozen pool and seed, keeps the Top-5 baseline, and selects the V3 Qwen3-4B controller with Top-2 unseen retrieval. **The 1,000-question experiment has not been executed.**

```bash
python scripts/run_retrieval_study.py --config configs/retrieval_study_v3.yaml --mode combined --sample-size 1000 --pool-size 1000 --seed 88 --baseline-top-k 5 --adaptive-top-k 2 --max-rounds 5 --run-id final1000-seed88-top5-adaptive2unseenx5-v3
```
