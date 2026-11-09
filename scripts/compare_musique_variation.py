#!/usr/bin/env python3
"""Validate and summarize the eight frozen MuSiQue retrieval runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from compare_retrieval_studies_v3 import load, system_metrics
from reliable_rag.evaluation.retrieval_study import summarize
from reliable_rag.io.artifacts import file_sha256, write_json

NAMES = [
    *(f"baseline_top{k}_musique_n100_seed88" for k in range(2, 9)),
    "adaptive_v3_musique_n100_seed88",
]


def _write_csv(path: Path, metrics: dict[str, dict]) -> None:
    columns = ["metric", *NAMES]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for metric in metrics[NAMES[-1]]:
            if isinstance(metrics[NAMES[-1]][metric], dict):
                continue
            writer.writerow(
                {"metric": metric, **{name: metrics[name].get(metric) for name in NAMES}}
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/musique/subset-100-seed88/subset-development.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/musique_n100_seed88_comparison")
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    contexts = manifest["sampling"]["contexts"]
    expected_ids = manifest["sampling"]["selected_question_ids"]
    if len(expected_ids) != 100 or len(set(expected_ids)) != 100:
        raise ValueError("MuSiQue manifest must have exactly 100 unique questions")
    by_id = {item["question_id"]: item for item in contexts}
    if set(by_id) != set(expected_ids):
        raise ValueError("MuSiQue context manifest IDs differ from selected questions")
    rows_by_system = {}
    input_hashes = None
    for name in NAMES:
        directory = args.config_root / name
        metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
        rows = load(directory / "results.jsonl")
        if metadata["status"] not in {"completed", "completed_with_model_errors"}:
            raise ValueError(f"unfinished MuSiQue run: {name}")
        if (
            metadata["selected_question_ids"] != expected_ids
            or [r.question_id for r in rows] != expected_ids
        ):
            raise ValueError(f"question sample/order differs: {name}")
        if file_sha256(directory / "results.jsonl") != metadata["result_sha256"]:
            raise ValueError(f"result hash differs: {name}")
        if metadata["pool_manifest_sha256"] != file_sha256(args.manifest):
            raise ValueError(f"pool manifest differs: {name}")
        common_inputs = {
            path: digest
            for path, digest in metadata["inputs"].items()
            if path.endswith(
                ("questions-development.jsonl", "evidence-development.jsonl", ".npz", ".npz.json")
            )
        }
        if input_hashes is None:
            input_hashes = common_inputs
        elif common_inputs != input_hashes:
            raise ValueError(f"question, evidence or index inputs differ: {name}")
        if metadata["config"]["dataset"]["dataset_name"] != "StonyBrookNLP/musique":
            raise ValueError(f"dataset config differs: {name}")
        if name.startswith("baseline_"):
            k = int(name.split("top", 1)[1].split("_", 1)[0])
            if metadata["model_used"] or any(r.retrieval_rounds != 1 for r in rows):
                raise ValueError(f"baseline used a model or multiple rounds: {name}")
            if metadata["config"]["retrieval"]["initial_top_k"] != k:
                raise ValueError(f"baseline k differs: {name}")
        else:
            config = metadata["config"]
            if (
                not metadata["model_used"]
                or config["retrieval"]["followup_top_k"] != 2
                or config["retrieval"]["max_total_retrieval_rounds"] != 5
                or config["slm"]["prompt_version"] != "retrieval-controller-v3"
            ):
                raise ValueError("adaptive run is not the configured V3 Top-2 × 5 experiment")
        for row in rows:
            context = by_id[row.question_id]
            if row.candidate_paragraph_ids != context["candidate_paragraph_ids_in_order"] or set(
                row.gold_paragraph_ids
            ) != set(context["gold_paragraph_ids"]):
                raise ValueError(f"candidate/gold context differs: {name}, {row.question_id}")
        rows_by_system[name] = rows
    for observations in zip(*(rows_by_system[name] for name in NAMES), strict=True):
        top8 = observations[6].raw_retrieved_ids
        for k, row in zip(range(2, 8), observations[:6], strict=True):
            if row.raw_retrieved_ids != top8[:k]:
                raise ValueError(f"Top-{k} differs from the Top-8 prefix for {row.question_id}")
    metrics = {}
    for name, rows in rows_by_system.items():
        values = system_metrics("v3" if name == NAMES[-1] else "top5", rows)
        original_summary = summarize(rows)
        values["mean_duplicate_retrievals"] = original_summary["duplicate_retrievals"]["mean"]
        values["final_need_more_evidence"] = (
            original_summary["ending_model_status"].get("need_more_evidence", 0)
            if name == NAMES[-1]
            else None
        )
        metrics[name] = values
    groups: dict[str, dict[str, dict[str, list]]] = defaultdict(lambda: defaultdict(dict))
    for name, rows in rows_by_system.items():
        for field in ("gold_count", "density_band", "total_context_count"):
            buckets = defaultdict(list)
            for row in rows:
                context = by_id[row.question_id]
                value = (
                    context["gold_count"] + context["distractor_count"]
                    if field == "total_context_count"
                    else context[field]
                )
                buckets[str(value)].append(row)
            groups[field][name] = {
                value: system_metrics("v3" if name == NAMES[-1] else "top5", bucket)
                for value, bucket in sorted(buckets.items())
            }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "comparison.json",
        {"sample_manifest_sha256": file_sha256(args.manifest), "metrics": metrics},
    )
    _write_csv(args.output_dir / "comparison.csv", metrics)
    write_json(args.output_dir / "breakdowns.json", groups)
    with (args.output_dir / "breakdowns.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "group_type",
                "group_value",
                "system",
                "questions",
                "mean_gold_evidence_percentage",
                "full_gold_recovery_rate",
                "mean_unique_noise_percentage",
                "mean_raw_paragraphs",
                "mean_unique_paragraphs",
                "mean_retrieval_rounds",
                "total_experiment_errors",
            ],
        )
        writer.writeheader()
        for group_type, systems in groups.items():
            for name, buckets in systems.items():
                for group_value, values in buckets.items():
                    writer.writerow(
                        {
                            "group_type": group_type,
                            "group_value": group_value,
                            "system": name,
                            "questions": values["attempted"],
                            **{key: values[key] for key in writer.fieldnames[4:]},
                        }
                    )
    print("Validated 8 MuSiQue runs on 100 identical ordered questions")
    print(f"Comparison: {args.output_dir / 'comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
