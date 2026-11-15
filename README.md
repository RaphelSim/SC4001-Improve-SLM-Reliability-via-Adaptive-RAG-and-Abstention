# Adaptive paragraph retrieval study

This project measures how much annotated evidence a fixed Top-k retriever or an adaptive Qwen3 controller recovers. HotpotQA and MuSiQue are prepared as question and paragraph records. MiniLM embeds paragraphs; the adaptive system asks Qwen3 whether more evidence is needed and, if so, retrieves another batch. The study records evidence recovery, noise, retrieval rounds, and model errors. The retrieval experiments do not generate final answers.

## 0. Experiment Clarification

The results in our assignment report were generated with a quantized model through Apple MLX. Rerunning the experiments on Windows may produce different results.

## 1. Set up

Use Python 3.11 from the repository root. The default Mac setup uses Apple Silicon and MLX; Windows uses PyTorch and Transformers with the corresponding full-weight Qwen3 checkpoint.

**macOS (Apple Silicon):**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[mac,notebook]'
```

**Windows (PowerShell):**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[windows,notebook]"
```

The Mac install command includes `mlx` and `mlx-lm`; use native Apple Silicon Python and run in a terminal with Metal access. The Windows install command includes PyTorch and Transformers. Public Hugging Face models normally need no token. Internet access is needed on first download. For a CUDA Windows machine, install the PyTorch build appropriate for its CUDA version before installing the project; CPU inference also works but Qwen3-4B is slow and needs substantial memory.

## Reproduce our experiment

The submission has two archives. `Experiment_Data.zip` contains the original local inputs, frozen HotpotQA 500/1000-question data, frozen MuSiQue 100-question data, and the saved split. `outputs/Experiments_Output.zip` contains the saved experiment folders. Extract both from the repository root using Python on either OS:

```bash
python -m zipfile -e Experiment_Data.zip .
python -m zipfile -e outputs/Experiments_Output.zip outputs
```

The archives do not contain downloaded model weights or MiniLM indexes. To rerun the default HotpotQA 500-question pool after extraction, download the configured models and build or verify its index:

```bash
python scripts/prepare_models.py --config configs/retrieval_study_v3.yaml
python scripts/prepare_data.py
```

Then edit `configs/experiment.yaml`, choose a **new** `run_id`, and run `python scripts/run_study.py`. The default data and experiment configs use HotpotQA pool size 500 and seed 88. For the archived HotpotQA 1000-question pool or MuSiQue sample, set the corresponding `base_config`, `subset_size`, and `pool_size` before building its index and running an experiment. Extracting the output archive is sufficient to inspect recorded results; rerunning inference is optional.

## 2. Check dependencies and run one inference case

Run one command for your OS. The script checks packages and the inference device, prepares one **HotpotQA** question with seed 88, downloads or verifies the default MiniLM and Qwen3 models, builds the embedding index, and runs one adaptive V3 case. Its prepared data and experiment output live in a temporary directory and are deleted when the check finishes, including on failure. Downloaded model files remain in `cache/` for real experiments; a newly created smoke index is removed.

```bash
python scripts/check_and_smoke.py --os mac
```

```powershell
python scripts/check_and_smoke.py --os windows
```

## 3. Prepare data

Edit `configs/data.yaml`. Set its `base_config` to `configs/retrieval_study_v3.yaml` for HotpotQA or `configs/musique_retrieval_study_v3.yaml` for MuSiQue. Set `subset_size` and `seed`, then run:

```bash
python scripts/prepare_data.py
```

This calls the existing dataset adapter and builds or reuses the matching MiniLM index. HotpotQA can download from the pinned Hugging Face source or read a local export via `input_path`. The current MuSiQue adapter is intentionally fixed to the original 100-question, seed-88 variation study. For MuSiQue, obtain the answerable train JSONL, set `input_path: data/raw/musique_ans_v1.0_train.jsonl`, `source_repo: voidful/MuSiQue`, and `revision: a7d9f9adf6191604fc67cde318ee1a86fcf7babc` in `configs/data.yaml`. The raw file can be downloaded with:

```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='voidful/MuSiQue', repo_type='dataset', filename='musique_ans_v1.0_train.jsonl', revision='a7d9f9adf6191604fc67cde318ee1a86fcf7babc', local_dir='data/raw')"
```

Prepared questions, evidence, and the selection manifest are under `data/processed/<dataset>/subset-<size>-seed<seed>/`. The index is under `cache/`. Existing frozen data are reused when compatible; do not force regeneration when reproducing a recorded result.

## 4. Run an experiment

Edit `configs/experiment.yaml` and run:

```bash
python scripts/run_study.py
```

Choose `base_config` for V1 (`retrieval_study.yaml`), V2.1 (`retrieval_study_v2_1.yaml`), V3 (`retrieval_study_v3.yaml`), or MuSiQue V3 (`musique_retrieval_study_v3.yaml`). Set `mode` to `baseline` or `adaptive`, `os` to `mac` or `windows`, and choose an unused `run_id` (the output folder name). `pool_size` must match prepared data; `sample_size` selects how many questions to run. `baseline_top_k` controls the single retrieval round; `adaptive_top_k` controls each adaptive round; `max_rounds` caps adaptive retrieval. `model_id` and `model_revision` optionally override the preset. On Windows, the default MLX Qwen3-4B or 1.7B ID is mapped to the matching Transformers checkpoint. `reuse_question_ids_from` can point to an earlier `run_metadata.json` to reuse its ordered sample.

For a baseline series, run this command once per k value with a new `run_id`. For an adaptive comparison, set the same pool, seed, and `reuse_question_ids_from` to the baseline run metadata. Existing run folders are never overwritten.

## 5. Inspect results and notebooks

Each `outputs/<run_id>/` contains `results.jsonl` (one record per question), `summary.json` and `summary.csv` (aggregate metrics), `run_metadata.json` (source, settings, hashes and timing), plus `errors.jsonl` and `events.jsonl`. For a rerun, edit the data and experiment configs, rerun steps 3–5, and use a new run ID.


## Architecture and directory layout

```text
configs/        User-edited data and experiment configs; versioned study presets
data/           Raw downloads (local) and prepared question/evidence records
cache/          Downloaded models and MiniLM embedding indexes (local)
prompts/        Versioned controller and retry prompts
schemas/        JSON schemas for the study records
scripts/        Preparation, environment check, and experiment entry points
src/reliable_rag/  Dataset adapters, retrieval, model clients, records, metrics
outputs/        Saved experiment observations and summaries
notebooks/      Presentation notebook and figures
```

The flow is `raw dataset → prepared records → MiniLM index → fixed or adaptive retrieval → per-question records → summaries → notebook`. Gold paragraph IDs are held for evaluation and are not supplied to the adaptive model.
