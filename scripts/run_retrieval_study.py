#!/usr/bin/env python3
"""Run single-shot baseline, adaptive retrieval, or their paired comparison."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

from reliable_rag.config import load_settings
from reliable_rag.contracts.enums import SystemVariant
from reliable_rag.contracts.records import EvidenceChunk, QuestionRecord
from reliable_rag.evaluation.retrieval_study import paired_differences, summarize
from reliable_rag.io.artifacts import file_sha256, write_json
from reliable_rag.io.jsonl import read_jsonl
from reliable_rag.retrieval.study import run_retrieval_case
from reliable_rag.run_artifacts import RunWriter, run_directory
from reliable_rag.runtime import create_model_components, create_retriever


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary_rows(system: str, summary: dict) -> list[dict]:
    rows = []
    for name in (
        "gold_evidence_percentage",
        "noise_percentage",
        "raw_paragraphs_retrieved",
        "unique_paragraphs_retrieved",
        "retrieval_rounds",
        "duplicate_retrievals",
    ):
        rows.append({"system": system, "metric": name, **summary[name]})
    for name in (
        "full_gold_recovery_rate",
        "zero_gold_recovery_rate",
        "duplicate_percentage_raw",
        "attempted",
        "valid",
        "errors",
    ):
        rows.append(
            {
                "system": system,
                "metric": name,
                "mean": summary[name],
                "median": None,
                "std": None,
                "min": None,
                "max": None,
            }
        )
    rows.append(
        {
            "system": system,
            "metric": "final_insufficient_rate",
            "mean": (
                summary["ending_model_status"].get("need_more_evidence", 0) / summary["valid"]
                if system == "adaptive" and summary["valid"]
                else None
            ),
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }
    )
    return rows


def _run_one(system, settings, questions, retriever, *, run_id, inputs, allow_model_download=False):
    local = settings.model_copy(deep=True)
    local.experiment.run_id = run_id
    local.experiment.system_variant = (
        SystemVariant.STANDARD_RAG if system == "baseline" else SystemVariant.ADAPTIVE_RETRIEVAL
    )
    prompts = (
        []
        if system == "baseline"
        else [
            local.slm.sufficiency_prompt_path,
            local.slm.structured_retry_prompt_path,
        ]
    )
    writer = RunWriter(
        local, run_id, inputs=inputs, prompt_paths=prompts, scope="paragraph_retrieval_study"
    )
    logger = logging.getLogger("reliable_rag")
    logger.addHandler(writer)
    start = time.monotonic()
    writer.metadata["summary"] = {
        "planned": len(questions),
        "attempted": 0,
        "records_written": 0,
        "model_errors": 0,
        "runtime_errors": 0,
    }
    writer.metadata["selected_question_ids"] = [q.question_id for q in questions]
    writer.metadata["source_questions_path"] = str(local.questions_path)
    writer.metadata["dataset_pool_size"] = local.dataset.subset_size
    pool_manifest = json.loads(local.selection_path.read_text())
    writer.metadata["dataset_source"] = pool_manifest["source"]
    writer.metadata["preprocessing"] = pool_manifest["preprocessing"]
    writer.metadata["pool_manifest_sha256"] = file_sha256(local.selection_path)
    writer.metadata["sample_size"] = len(questions)
    writer.metadata["model_used"] = system == "adaptive"
    writer.metadata["prompt_version"] = local.slm.prompt_version if system == "adaptive" else None
    writer.save()
    records = []
    current_question = None
    try:
        assessor = None
        if system == "adaptive":
            model = create_model_components(local, allow_model_download=allow_model_download)
            assessor = model.assessor
            writer.metadata["local_model_path"] = str(model.client.model_path)
        top_k = (
            local.retrieval.initial_top_k
            if system == "baseline"
            else local.retrieval.followup_top_k
        )
        max_rounds = 1 if system == "baseline" else local.retrieval.max_total_retrieval_rounds
        for question in questions:
            current_question = question.question_id
            writer.metadata["summary"]["attempted"] += 1
            writer.save()
            record = run_retrieval_case(
                question,
                retriever=retriever,
                assessor=assessor,
                system=system,
                top_k=top_k,
                max_rounds=max_rounds,
            )
            writer.append("results", record.model_dump(mode="json"))
            records.append(record)
            writer.metadata["summary"]["records_written"] += 1
            if record.final_model_status == "ERROR":
                writer.metadata["summary"]["model_errors"] += 1
                writer.append(
                    "errors",
                    {
                        "question_id": question.question_id,
                        "category": "controller_error",
                        "error_type": record.error_type,
                        "message": record.error_message,
                        "retrieval_rounds_completed": record.retrieval_rounds,
                    },
                )
            writer.save()
            logging.info("%s: completed %s/%s", system, len(records), len(questions))
        summary = summarize(records)
        write_json(writer.directory / "summary.json", summary)
        _write_csv(writer.directory / "summary.csv", _summary_rows(system, summary))
        writer.metadata["status"] = (
            "completed_with_model_errors" if summary["errors"] else "completed"
        )
        writer.metadata["result_sha256"] = file_sha256(writer.directory / "results.jsonl")
        writer.metadata["summary_sha256"] = file_sha256(writer.directory / "summary.json")
    except BaseException as exc:
        writer.metadata["status"] = (
            "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        )
        writer.metadata["summary"]["runtime_errors"] += 1
        writer.append(
            "errors",
            {
                "category": "runtime_error",
                "question_id": current_question,
                "exception_type": type(exc).__name__,
                "message": str(exc),
            },
        )
        raise
    finally:
        writer.metadata["finished_at"] = datetime.now(UTC).isoformat()
        writer.metadata["elapsed_seconds"] = time.monotonic() - start
        writer.save()
        logger.removeHandler(writer)
        writer.close()
    return records, writer.directory, summary


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/retrieval_study.yaml"))
    parser.add_argument("--mode", choices=("baseline", "adaptive", "combined"), default="combined")
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--pool-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--baseline-top-k", type=int)
    parser.add_argument("--adaptive-top-k", type=int)
    parser.add_argument("--max-rounds", type=int)
    parser.add_argument("--model-id")
    parser.add_argument("--model-revision")
    parser.add_argument("--run-id", help="Defaults to a timestamped unique ID")
    args = parser.parse_args()
    settings = load_settings(args.config)
    sample_size = (
        args.sample_size if args.sample_size is not None else settings.experiment.sample_size or 100
    )
    pool_size = args.pool_size if args.pool_size is not None else settings.dataset.subset_size
    seed = args.seed if args.seed is not None else settings.project_seed
    baseline_top_k = (
        args.baseline_top_k if args.baseline_top_k is not None else settings.retrieval.initial_top_k
    )
    adaptive_top_k = (
        args.adaptive_top_k
        if args.adaptive_top_k is not None
        else settings.retrieval.followup_top_k
    )
    max_rounds = (
        args.max_rounds
        if args.max_rounds is not None
        else settings.retrieval.max_total_retrieval_rounds
    )
    if pool_size is None:
        parser.error("a finite frozen pool size is required")
    if not 1 <= sample_size <= pool_size:
        parser.error("sample size must be between one and pool size")
    if pool_size < 1 or seed < 0 or min(baseline_top_k, adaptive_top_k, max_rounds) < 1:
        parser.error("pool size/top-k/rounds must be positive and seed non-negative")
    settings.project_seed = seed
    settings.dataset.subset_size = pool_size
    settings.retrieval.initial_top_k = baseline_top_k
    settings.retrieval.followup_top_k = adaptive_top_k
    settings.retrieval.max_total_retrieval_rounds = max_rounds
    settings.experiment.sample_size = sample_size
    if args.model_id:
        settings.slm.model_id = args.model_id
        if args.model_revision is None:
            settings.slm.revision = None
    if args.model_revision:
        settings.slm.revision = args.model_revision
    settings = type(settings).model_validate(settings.model_dump())
    if not settings.selection_path.exists() or not settings.index_path.exists():
        raise FileNotFoundError("Prepare the frozen pool and embedding index first")
    questions = read_jsonl(settings.questions_path, QuestionRecord)
    if len(questions) != pool_size or len({q.question_id for q in questions}) != len(questions):
        raise ValueError("frozen pool size or unique question IDs do not match")
    selection = json.loads(settings.selection_path.read_text())
    if [q.question_id for q in questions] != selection["sampling"]["selected_question_ids"]:
        raise ValueError("prepared question order differs from frozen selection manifest")
    if (
        selection["sampling"]["seed"] != seed
        or selection["sampling"]["selected_count"] != pool_size
    ):
        raise ValueError("selection manifest does not match requested seed/pool")
    if (
        file_sha256(settings.questions_path)
        != selection["artifacts"][settings.questions_path.name]["sha256"]
    ):
        raise ValueError("prepared questions changed after pool freeze")
    selected = questions[:sample_size]
    if any(not q.gold_paragraph_ids for q in selected):
        raise ValueError("selected question lacks frozen gold paragraph IDs")
    evidence = read_jsonl(settings.evidence_path, EvidenceChunk)
    corpus_ids = {item.evidence_id for item in evidence}
    for question in selected:
        if not set(question.gold_paragraph_ids).issubset(question.candidate_evidence_ids):
            raise ValueError("gold paragraphs missing from candidate context")
        if not set(question.candidate_evidence_ids).issubset(corpus_ids):
            raise ValueError("candidate paragraphs missing from prepared corpus")
    retriever = create_retriever(settings, evidence, settings.index_path, questions=selected)
    run_id = args.run_id or (
        f"{datetime.now(UTC):%Y%m%d-%H%M%S}_{args.mode}_n{sample_size}_seed{seed}"
    )
    root = run_directory(settings.output_root, run_id)
    if args.mode == "combined" and root.exists():
        raise FileExistsError(f"Comparison run already exists: {root}")
    inputs = [
        settings.questions_path,
        settings.evidence_path,
        settings.selection_path,
        settings.index_path,
        settings.index_path.with_suffix(settings.index_path.suffix + ".json"),
    ]
    if args.mode == "combined":
        settings.output_root = root
    outputs = {}
    for system in ("baseline", "adaptive") if args.mode == "combined" else (args.mode,):
        records, directory, summary = _run_one(
            system,
            settings,
            selected,
            retriever,
            run_id=system if args.mode == "combined" else run_id,
            inputs=inputs,
        )
        outputs[system] = (records, directory, summary)
    if args.mode == "combined":
        paired = paired_differences(outputs["baseline"][0], outputs["adaptive"][0])
        with (root / "paired.jsonl").open("w", encoding="utf-8") as handle:
            for row in paired:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        comparison = {
            "run_id": run_id,
            "sample_size": sample_size,
            "seed": seed,
            "pool_size": pool_size,
            "question_ids": [q.question_id for q in selected],
            "question_pool_sha256": file_sha256(settings.questions_path),
            "candidate_corpus_sha256": file_sha256(settings.evidence_path),
            "baseline": outputs["baseline"][2],
            "adaptive": outputs["adaptive"][2],
            "valid_pairs": sum(row["valid_pair"] for row in paired),
        }
        write_json(root / "summary.json", comparison)
        _write_csv(
            root / "comparison.csv",
            _summary_rows("baseline", comparison["baseline"])
            + _summary_rows("adaptive", comparison["adaptive"]),
        )
        baseline_metadata = json.loads((outputs["baseline"][1] / "run_metadata.json").read_text())
        adaptive_metadata = json.loads((outputs["adaptive"][1] / "run_metadata.json").read_text())
        write_json(
            root / "metadata.json",
            {
                "run_id": run_id,
                "status": "completed_with_model_errors"
                if comparison["adaptive"]["errors"]
                else "completed",
                "created_at": baseline_metadata["created_at"],
                "finished_at": adaptive_metadata["finished_at"],
                "elapsed_seconds": baseline_metadata["elapsed_seconds"]
                + adaptive_metadata["elapsed_seconds"],
                "baseline_run": str(outputs["baseline"][1]),
                "adaptive_run": str(outputs["adaptive"][1]),
                "git_commit": adaptive_metadata["git_commit"],
                "git_dirty": adaptive_metadata["git_dirty"],
                "packages": adaptive_metadata["packages"],
                "resolved_configs": {
                    "baseline": baseline_metadata["config"],
                    "adaptive": adaptive_metadata["config"],
                },
                "dataset_source": adaptive_metadata["dataset_source"],
                "preprocessing": adaptive_metadata["preprocessing"],
                "selected_question_ids": comparison["question_ids"],
                "pool_manifest": str(settings.selection_path),
                "pool_manifest_sha256": file_sha256(settings.selection_path),
                "index_sha256": adaptive_metadata["inputs"][str(settings.index_path)],
                "prompts": adaptive_metadata["prompts"],
                "model_path": adaptive_metadata.get("local_model_path"),
                "errors": {
                    "baseline": comparison["baseline"]["errors"],
                    "adaptive": comparison["adaptive"]["errors"],
                },
                "summary_sha256": file_sha256(root / "summary.json"),
                "paired_sha256": file_sha256(root / "paired.jsonl"),
            },
        )
        logging.info("Comparison saved to %s", root)
    else:
        logging.info("%s saved to %s", args.mode, outputs[args.mode][1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
