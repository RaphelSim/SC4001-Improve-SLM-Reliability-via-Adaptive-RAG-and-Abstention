#!/usr/bin/env python3
"""Compare frozen Top-5, V1, V2.1, and V3 results without rerunning them."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from reliable_rag.contracts.retrieval_study import RetrievalStudyRecord
from reliable_rag.evaluation.retrieval_study import summarize
from reliable_rag.io.artifacts import file_sha256, write_json
from reliable_rag.io.jsonl import read_jsonl


def load(path: Path) -> list[RetrievalStudyRecord]:
    rows = read_jsonl(path, RetrievalStudyRecord)
    if len(rows) != len({row.question_id for row in rows}):
        raise ValueError(f"duplicate question IDs in {path}")
    return rows


def retrieval_metrics(rows: list[RetrievalStudyRecord]) -> dict:
    n = len(rows)
    gold = sorted(row.gold_evidence_percentage for row in rows)
    raw = sum(row.raw_paragraphs_retrieved for row in rows)
    return {
        "attempted": n,
        "mean_gold_evidence_percentage": mean(gold),
        "median_gold_evidence_percentage": (gold[(n - 1) // 2] + gold[n // 2]) / 2,
        "full_gold_recovery_rate": sum(x == 100 for x in gold) / n,
        "zero_gold_recovery_rate": sum(x == 0 for x in gold) / n,
        "mean_unique_noise_percentage": mean(row.noise_percentage for row in rows),
        "mean_raw_paragraphs": raw / n,
        "mean_unique_paragraphs": mean(row.unique_paragraphs_retrieved for row in rows),
        "mean_retrieval_rounds": mean(row.retrieval_rounds for row in rows),
        "duplicate_share_raw": sum(row.duplicate_retrieval_count for row in rows) / raw
        if raw
        else 0,
    }


def system_metrics(name: str, rows: list[RetrievalStudyRecord]) -> dict:
    result = retrieval_metrics(rows)
    summary = summarize(rows)
    diagnostics = summary["controller_diagnostics"]
    result.update(
        gold_based_false_sufficiency=diagnostics["gold_based_false_sufficiency"],
        gold_based_false_sufficiency_count=diagnostics["gold_based_false_sufficiency_count"],
        gold_incomplete_decisions=diagnostics["gold_incomplete_decisions"],
        round_1_gold_based_false_sufficiency=diagnostics["round_1_gold_based_false_sufficiency"],
        round_1_false_sufficiency_count=diagnostics["round_1_false_sufficiency_count"],
        round_1_gold_incomplete_decisions=diagnostics["round_1_gold_incomplete_decisions"],
        mean_controller_calls=mean(r.controller_model_calls for r in rows)
        if name == "v3"
        else None,
        mean_repair_calls=mean(r.repair_model_calls for r in rows) if name == "v3" else None,
        mean_query_regeneration_calls=mean(r.query_regeneration_model_calls for r in rows)
        if name == "v3"
        else None,
        mean_total_reasoning_model_calls=mean(r.total_reasoning_model_calls for r in rows)
        if name == "v3"
        else None,
        repeated_query_stops=sum(r.stop_reason == "repeated_query" for r in rows)
        if name == "v3"
        else None,
        retrieval_pool_exhausted_stops=sum(
            r.stop_reason == "retrieval_pool_exhausted" for r in rows
        )
        if name == "v3"
        else None,
        max_rounds_stops=sum(r.stop_reason == "max_rounds_reached" for r in rows)
        if name == "v3"
        else None,
        model_output_errors=sum(r.stop_reason == "model_output_error" for r in rows)
        if name == "v3"
        else None,
        total_experiment_errors=sum(r.final_model_status == "ERROR" for r in rows),
        stop_reason_distribution=dict(Counter(r.stop_reason for r in rows if r.stop_reason))
        if name == "v3"
        else None,
    )
    if name == "top5":
        result["gold_based_false_sufficiency"] = None
        result["round_1_gold_based_false_sufficiency"] = None
        result.update(
            mean_controller_calls=0,
            mean_repair_calls=0,
            mean_query_regeneration_calls=0,
            mean_total_reasoning_model_calls=0,
        )
    return result


def paired_v2_v3(
    v2: list[RetrievalStudyRecord], v3: list[RetrievalStudyRecord]
) -> tuple[list[dict], dict]:
    paired = []
    for before, after in zip(v2, v3, strict=True):
        paired.append(
            {
                "question_id": before.question_id,
                "v2_error": before.final_model_status == "ERROR",
                "v3_error": after.final_model_status == "ERROR",
                "v2_gold_percentage": before.gold_evidence_percentage,
                "v3_gold_percentage": after.gold_evidence_percentage,
                "gold_delta": after.gold_evidence_percentage - before.gold_evidence_percentage,
                "noise_delta": after.noise_percentage - before.noise_percentage,
                "raw_delta": after.raw_paragraphs_retrieved - before.raw_paragraphs_retrieved,
                "unique_delta": after.unique_paragraphs_retrieved
                - before.unique_paragraphs_retrieved,
                "rounds_delta": after.retrieval_rounds - before.retrieval_rounds,
                "v3_stop_reason": after.stop_reason,
            }
        )
    summary = {
        "questions": len(paired),
        "gold_improved": sum(r["gold_delta"] > 0 for r in paired),
        "gold_reduced": sum(r["gold_delta"] < 0 for r in paired),
        "gold_unchanged": sum(r["gold_delta"] == 0 for r in paired),
        "noise_reduced": sum(r["noise_delta"] < 0 for r in paired),
        "noise_increased": sum(r["noise_delta"] > 0 for r in paired),
        "noise_unchanged": sum(r["noise_delta"] == 0 for r in paired),
        "rounds_fewer": sum(r["rounds_delta"] < 0 for r in paired),
        "rounds_more": sum(r["rounds_delta"] > 0 for r in paired),
        "rounds_unchanged": sum(r["rounds_delta"] == 0 for r in paired),
        "v2_error_to_v3_clean": sum(r["v2_error"] and not r["v3_error"] for r in paired),
        "v2_clean_to_v3_error": sum(not r["v2_error"] and r["v3_error"] for r in paired),
        "became_full_gold": sum(
            r["v2_gold_percentage"] < 100 and r["v3_gold_percentage"] == 100 for r in paired
        ),
        "lost_full_gold": sum(
            r["v2_gold_percentage"] == 100 and r["v3_gold_percentage"] < 100 for r in paired
        ),
        "mean_gold_delta": mean(r["gold_delta"] for r in paired),
        "mean_noise_delta": mean(r["noise_delta"] for r in paired),
        "mean_raw_delta": mean(r["raw_delta"] for r in paired),
        "mean_unique_delta": mean(r["unique_delta"] for r in paired),
        "mean_rounds_delta": mean(r["rounds_delta"] for r in paired),
    }
    return paired, summary


def v3_rounds(rows: list[RetrievalStudyRecord]) -> dict:
    by_round = defaultdict(list)
    for row in rows:
        for rd in row.rounds:
            by_round[rd.round_number].append(rd)
    return {
        str(number): {
            "active_questions": len(items),
            "mean_cumulative_gold_percentage": mean(
                r.cumulative_gold_evidence_percentage for r in items
            ),
            "full_gold_count": sum(r.gold_complete_after_round for r in items),
            "full_gold_rate": sum(r.gold_complete_after_round for r in items) / len(items),
            "mean_cumulative_noise_percentage": mean(
                r.cumulative_unique_noise_percentage for r in items
            ),
            "mean_cumulative_unique_paragraphs": mean(
                len(r.cumulative_unique_paragraph_ids) for r in items
            ),
            "seen_candidate_ids_skipped": sum(len(r.ranked_seen_ids_skipped) for r in items),
            "stops": sum(r.stop_reason is not None for r in items),
            "continues": sum(r.stop_reason is None for r in items),
            "new_unique_paragraphs": sum(len(r.new_unique_ids) for r in items),
        }
        for number, items in sorted(by_round.items())
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ("top5", "v1", "v2", "v3"):
        parser.add_argument(f"--{label}", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    paths = {label: getattr(args, label) for label in ("top5", "v1", "v2", "v3")}
    systems = {label: load(path) for label, path in paths.items()}
    ids = [[r.question_id for r in systems[label]] for label in systems]
    if len({tuple(x) for x in ids}) != 1 or len(ids[0]) != 100:
        raise ValueError("all four runs must contain exactly the same 100 ordered question IDs")
    for rows in zip(*systems.values(), strict=True):
        if (
            len({tuple(r.gold_paragraph_ids) for r in rows}) != 1
            or len({tuple(r.candidate_paragraph_ids) for r in rows}) != 1
        ):
            raise ValueError(f"gold labels or candidate scope differ for {rows[0].question_id}")
    paired, pair_summary = paired_v2_v3(systems["v2"], systems["v3"])
    metrics = {label: system_metrics(label, rows) for label, rows in systems.items()}
    result = {
        "question_ids": ids[0],
        "metrics": metrics,
        "paired_v2_v3_summary": pair_summary,
        "v3_rounds": v3_rounds(systems["v3"]),
        "sources": {
            label: {"path": str(path), "sha256": file_sha256(path)} for label, path in paths.items()
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "four_way_comparison.json", result)
    with (args.output_dir / "four_way_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", *metrics])
        writer.writeheader()
        for metric in metrics["v3"]:
            if isinstance(metrics["v3"][metric], dict):
                continue
            writer.writerow(
                {"metric": metric, **{label: data.get(metric) for label, data in metrics.items()}}
            )
    with (args.output_dir / "paired_v2_v3.jsonl").open("w", encoding="utf-8") as handle:
        for row in paired:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"Compared four runs on {len(ids[0])} ordered question IDs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
