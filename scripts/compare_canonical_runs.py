#!/usr/bin/env python3
"""Validate the seven-run registry and regenerate the canonical comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from compare_retrieval_studies_v3 import load, system_metrics
from reliable_rag.io.artifacts import file_sha256, write_json

ORDER = (
    "baseline_top2",
    "baseline_top3",
    "baseline_top4",
    "baseline_top5",
    "adaptive_v1",
    "adaptive_v2",
    "adaptive_v3",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("documentation/canonical_runs.json"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    if set(registry["runs"]) != set(ORDER):
        raise ValueError("canonical registry must contain exactly the seven named runs")
    question_ids = registry["selected_question_ids"]
    if len(question_ids) != 100 or len(set(question_ids)) != 100:
        raise ValueError("canonical question manifest must contain 100 unique IDs")
    ids_hash = hashlib.sha256(json.dumps(question_ids, separators=(",", ":")).encode()).hexdigest()
    if ids_hash != registry["selected_question_ids_sha256"]:
        raise ValueError("canonical question manifest hash differs")
    if file_sha256(Path(registry["sample_source_file"])) != registry["sample_source_sha256"]:
        raise ValueError("archived original sample manifest changed")

    systems = {}
    for name in ORDER:
        spec = registry["runs"][name]
        path = Path(spec["results_file"])
        if not path.is_file() or file_sha256(path) != spec["results_sha256"]:
            raise ValueError(f"missing or modified canonical results: {name}")
        meta = json.loads(Path(spec["metadata_file"]).read_text(encoding="utf-8"))
        if meta["result_sha256"] != spec["results_sha256"]:
            raise ValueError(f"run metadata result hash differs: {name}")
        if file_sha256(Path(spec["summary_file"])) != spec["summary_sha256"]:
            raise ValueError(f"canonical summary changed: {name}")
        rows = load(path)
        if [row.question_id for row in rows] != question_ids:
            raise ValueError(f"canonical question IDs/order differ: {name}")
        if len(rows) != spec["sample_size"] or spec["question_ids_sha256"] != ids_hash:
            raise ValueError(f"canonical sample size/hash differs: {name}")
        if (
            meta["status"] != spec["status"]
            or meta["summary"]["model_errors"] != spec["error_count"]
        ):
            raise ValueError(f"canonical run status/errors differ: {name}")
        if name.startswith("baseline_"):
            if meta["model_used"] or any(row.retrieval_rounds != 1 for row in rows):
                raise ValueError(f"fixed baseline invoked a model or multiple rounds: {name}")
        systems[name] = rows

    for observations in zip(*(systems[name] for name in ORDER), strict=True):
        if len({tuple(row.gold_paragraph_ids) for row in observations}) != 1:
            raise ValueError(f"gold paragraph IDs differ for {observations[0].question_id}")
        if len({tuple(row.candidate_paragraph_ids) for row in observations}) != 1:
            raise ValueError(f"candidate IDs differ for {observations[0].question_id}")
        top5_ids = observations[3].raw_retrieved_ids
        for index, k in enumerate((2, 3, 4)):
            if observations[index].raw_retrieved_ids != top5_ids[:k]:
                raise ValueError(f"Top-{k} results differ from Top-5 ranking prefix")

    metrics = {
        name: system_metrics(
            "top5" if name.startswith("baseline_") else name.removeprefix("adaptive_"), rows
        )
        for name, rows in systems.items()
    }
    output_dir = args.output_dir or Path(registry["runs"]["adaptive_v3"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "registry_file": str(args.registry),
        "registry_sha256": file_sha256(args.registry),
        "selected_question_ids_sha256": ids_hash,
        "metrics": metrics,
        "source_result_sha256": {name: registry["runs"][name]["results_sha256"] for name in ORDER},
    }
    write_json(output_dir / "canonical_comparison.json", result)
    with (output_dir / "canonical_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", *ORDER])
        writer.writeheader()
        for metric in metrics["adaptive_v3"]:
            if isinstance(metrics["adaptive_v3"][metric], dict):
                continue
            writer.writerow(
                {"metric": metric, **{name: metrics[name].get(metric) for name in ORDER}}
            )
    print(f"Validated {len(ORDER)} canonical runs on {len(question_ids)} identical ordered IDs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
