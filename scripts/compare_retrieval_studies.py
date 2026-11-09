#!/usr/bin/env python3
"""Compare an existing Top-k baseline, Adaptive V1 and Adaptive V2 study."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from reliable_rag.contracts.enums import SufficiencyStatus
from reliable_rag.contracts.retrieval_study import RetrievalStudyRecord
from reliable_rag.evaluation.retrieval_study import summarize
from reliable_rag.io.artifacts import file_sha256, write_json
from reliable_rag.io.jsonl import read_jsonl


def _load(path: Path):
    records = read_jsonl(path, RetrievalStudyRecord)
    if len({row.question_id for row in records}) != len(records):
        raise ValueError(f"duplicate question IDs in {path}")
    return records


def _means_summary(records):
    summary = summarize(records)
    diagnostics = summary["controller_diagnostics"]
    return {
        "attempted": summary["attempted"],
        "valid": summary["valid"],
        "errors": summary["errors"],
        "mean_gold_evidence_percentage": summary["gold_evidence_percentage"]["mean"],
        "median_gold_evidence_percentage": summary["gold_evidence_percentage"]["median"],
        "full_gold_recovery_rate": summary["full_gold_recovery_rate"],
        "mean_unique_noise_percentage": summary["noise_percentage"]["mean"],
        "average_raw_paragraphs": summary["raw_paragraphs_retrieved"]["mean"],
        "average_unique_paragraphs": summary["unique_paragraphs_retrieved"]["mean"],
        "average_retrieval_rounds": summary["retrieval_rounds"]["mean"],
        "average_duplicate_retrievals": summary["duplicate_retrievals"]["mean"],
        "duplicate_share_raw": summary["duplicate_percentage_raw"] / 100,
        "gold_based_false_sufficiency": diagnostics["gold_based_false_sufficiency"],
        "round_1_gold_based_false_sufficiency": diagnostics["round_1_gold_based_false_sufficiency"],
        "gold_based_false_insufficiency": diagnostics["gold_based_false_insufficiency"],
        "premature_annotated_gold_stop_questions": diagnostics[
            "premature_annotated_gold_stop_questions"
        ],
        "zero_new_evidence_round_rate": diagnostics["zero_new_evidence_round_rate"],
        "query_repetition_rate": diagnostics["query_repetition_rate"],
        "final_need_more_evidence": summary["ending_model_status"].get(
            SufficiencyStatus.NEED_MORE_EVIDENCE.value, 0
        ),
        "final_sufficient": summary["ending_model_status"].get(
            SufficiencyStatus.SUFFICIENT.value, 0
        ),
        "retrieval_stagnation_count": diagnostics["retrieval_stagnation_count"],
        "loop_guard_activation_count": diagnostics["loop_guard_activation_count"],
        "stopping_round_distribution": diagnostics["stopping_round_distribution"],
        "gold_recovery_by_round": diagnostics["gold_recovery_by_round"],
        "false_sufficiency_by_round": diagnostics["false_sufficiency_by_round"],
        "model_status_distribution": summary["ending_model_status"],
        "summary": summary,
    }


def _retrieval_metrics_all_attempts(records):
    """Retrieval is fully observed even when a later controller call errors."""
    n = len(records)
    raw_total = sum(row.raw_paragraphs_retrieved for row in records)
    duplicates = sum(row.duplicate_retrieval_count for row in records)
    gold = [row.gold_evidence_percentage for row in records]
    ordered = sorted(gold)
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    return {
        "attempted": n,
        "mean_gold_evidence_percentage": sum(gold) / n,
        "median_gold_evidence_percentage": median,
        "full_gold_recovery_rate": sum(value == 100 for value in gold) / n,
        "mean_unique_noise_percentage": sum(row.noise_percentage for row in records) / n,
        "average_raw_paragraphs": raw_total / n,
        "average_unique_paragraphs": sum(row.unique_paragraphs_retrieved for row in records) / n,
        "average_retrieval_rounds": sum(row.retrieval_rounds for row in records) / n,
        "average_duplicate_retrievals": duplicates / n,
        "duplicate_share_raw": duplicates / raw_total if raw_total else 0,
    }


def _paired_rows(left, right):
    left_by_id = {row.question_id: row for row in left}
    right_by_id = {row.question_id: row for row in right}
    if list(left_by_id) != list(right_by_id):
        raise ValueError("paired system question IDs/order differ")
    fields = {
        "gold_evidence_percentage": "gold_evidence_percentage",
        "unique_noise_percentage": "noise_percentage",
        "raw_paragraphs": "raw_paragraphs_retrieved",
        "unique_paragraphs": "unique_paragraphs_retrieved",
        "retrieval_rounds": "retrieval_rounds",
        "duplicate_retrievals": "duplicate_retrieval_count",
    }
    result = []
    for question_id, a in left_by_id.items():
        b = right_by_id[question_id]
        if (
            a.gold_paragraph_ids != b.gold_paragraph_ids
            or a.candidate_paragraph_ids != b.candidate_paragraph_ids
        ):
            raise ValueError(f"paired record inputs differ for {question_id}")
        result.append(
            {
                "question_id": question_id,
                "retrieval_pair_valid": True,
                "v2_controller_valid": b.final_model_status != "ERROR",
                **{key: getattr(b, attr) - getattr(a, attr) for key, attr in fields.items()},
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--adaptive-v1", type=Path, required=True)
    parser.add_argument("--adaptive-v2", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    systems = {
        "top5": _load(args.baseline),
        "adaptive_v1": _load(args.adaptive_v1),
        "adaptive_v2": _load(args.adaptive_v2),
    }
    ids = {name: [row.question_id for row in rows] for name, rows in systems.items()}
    if len({tuple(value) for value in ids.values()}) != 1:
        raise ValueError(
            "all three run files must contain identical question IDs in the same order"
        )
    valid_v2_ids = {
        row.question_id for row in systems["adaptive_v2"] if row.final_model_status != "ERROR"
    }
    if not valid_v2_ids:
        raise ValueError("Adaptive V2 contains no valid records")
    common = {
        name: [row for row in rows if row.question_id in valid_v2_ids]
        for name, rows in systems.items()
    }
    all_summaries = {name: _means_summary(rows) for name, rows in systems.items()}
    common_summaries = {name: _means_summary(rows) for name, rows in common.items()}
    paired = _paired_rows(systems["adaptive_v1"], systems["adaptive_v2"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "cohort_ids_sha256": __import__("hashlib")
        .sha256(json.dumps(ids["top5"], separators=(",", ":")).encode())
        .hexdigest(),
        "all_attempted_cohort": all_summaries,
        "retrieval_metrics_all_attempts": {
            name: _retrieval_metrics_all_attempts(rows) for name, rows in systems.items()
        },
        "v2_valid_common_cohort": {
            "n": len(valid_v2_ids),
            "systems": common_summaries,
        },
        "retrieval_pairs": len(paired),
        "v2_controller_valid_pairs": sum(row["v2_controller_valid"] for row in paired),
        "paired_adaptive_v1_to_v2": paired,
        "sources": {
            "top5": {"path": str(args.baseline), "sha256": file_sha256(args.baseline)},
            "adaptive_v1": {"path": str(args.adaptive_v1), "sha256": file_sha256(args.adaptive_v1)},
            "adaptive_v2": {"path": str(args.adaptive_v2), "sha256": file_sha256(args.adaptive_v2)},
        },
    }
    write_json(args.output_dir / "three_way_comparison.json", result)
    with (args.output_dir / "paired_adaptive_v1_v2.jsonl").open("w", encoding="utf-8") as f:
        for row in paired:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    csv_rows = []
    for cohort, summaries in (
        ("all_100_attempts", all_summaries),
        (f"v2_valid_common_n{len(valid_v2_ids)}", common_summaries),
    ):
        for system, summary in summaries.items():
            for metric, value in summary.items():
                if metric == "summary" or isinstance(value, (dict, list)):
                    continue
                csv_rows.append(
                    {"cohort": cohort, "system": system, "metric": metric, "value": value}
                )
    for system, summary in result["retrieval_metrics_all_attempts"].items():
        for metric, value in summary.items():
            csv_rows.append(
                {
                    "cohort": "retrieval_all_100_attempts",
                    "system": system,
                    "metric": metric,
                    "value": value,
                }
            )
    with (args.output_dir / "three_way_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=["cohort", "system", "metric", "value"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(
        f"Wrote comparison for {len(ids['top5'])} matched IDs; {len(valid_v2_ids)} V2-valid pairs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
