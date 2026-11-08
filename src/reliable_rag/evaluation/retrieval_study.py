"""Paragraph-level evidence and retrieval-cost summaries."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, median, pstdev

from reliable_rag.contracts.enums import SufficiencyStatus


def _stats(values):
    if not values:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None}
    return {
        "mean": mean(values),
        "median": median(values),
        "std": pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def summarize(records):
    """Exclude explicit controller errors from quality means while counting them."""
    valid = [r for r in records if r.final_model_status != "ERROR"]
    n = len(valid)
    by_round = defaultdict(list)
    for record in valid:
        by_round[record.retrieval_rounds].append(record)
    raw_total = sum(r.raw_paragraphs_retrieved for r in valid)
    duplicates = sum(r.duplicate_retrieval_count for r in valid)
    round_numbers = range(1, max(5, max(by_round, default=0)) + 1)
    status_counts = Counter(
        str(r.final_model_status.value) for r in valid if r.final_model_status is not None
    )
    return {
        "attempted": len(records),
        "valid": n,
        "errors": len(records) - n,
        "gold_evidence_percentage": _stats([r.gold_evidence_percentage for r in valid]),
        "noise_percentage": _stats([r.noise_percentage for r in valid]),
        "raw_paragraphs_retrieved": _stats([r.raw_paragraphs_retrieved for r in valid]),
        "unique_paragraphs_retrieved": _stats([r.unique_paragraphs_retrieved for r in valid]),
        "retrieval_rounds": _stats([r.retrieval_rounds for r in valid]),
        "duplicate_retrievals": _stats([r.duplicate_retrieval_count for r in valid]),
        "duplicate_percentage_raw": 100 * duplicates / raw_total if raw_total else 0,
        "full_gold_recovery_rate": sum(r.gold_evidence_percentage == 100 for r in valid) / n
        if n
        else None,
        "zero_gold_recovery_rate": sum(r.gold_evidence_percentage == 0 for r in valid) / n
        if n
        else None,
        "ending_model_status": dict(status_counts),
        "ending_model_status_percentages": {
            status: 100 * status_counts.get(status, 0) / n if n else 0
            for status in ("sufficient", "need_more_evidence")
        }
        if status_counts
        else {},
        "stopping_round_counts": {str(k): len(by_round[k]) for k in round_numbers},
        "stopping_round_percentages": {
            str(k): 100 * len(by_round[k]) / n if n else 0 for k in round_numbers
        },
        "by_stopping_round": {
            str(k): {
                "gold_evidence_percentage": _stats([r.gold_evidence_percentage for r in group]),
                "noise_percentage": _stats([r.noise_percentage for r in group]),
            }
            for k, group in sorted(by_round.items())
        },
        "model_stopped_early_without_gold": sum(
            r.final_model_status == SufficiencyStatus.SUFFICIENT
            and r.gold_evidence_percentage < 100
            for r in valid
        ),
        "model_continued_after_gold_complete": sum(
            any(item.gold_complete_after_round for item in r.rounds[:-1]) for r in valid
        ),
    }


def paired_differences(baseline, adaptive):
    base = {r.question_id: r for r in baseline}
    adap = {r.question_id: r for r in adaptive}
    if list(base) != list(adap) or len(base) != len(baseline) or len(adap) != len(adaptive):
        raise ValueError("baseline and adaptive question IDs/order must match")
    rows = []
    for question_id, b in base.items():
        a = adap[question_id]
        if (
            b.gold_paragraph_ids != a.gold_paragraph_ids
            or b.candidate_paragraph_ids != a.candidate_paragraph_ids
        ):
            raise ValueError("paired questions use different gold labels or candidate paragraphs")
        valid = a.final_model_status != "ERROR"
        rows.append(
            {
                "question_id": question_id,
                "valid_pair": valid,
                "adaptive_minus_baseline_gold_percentage": a.gold_evidence_percentage
                - b.gold_evidence_percentage
                if valid
                else None,
                "adaptive_minus_baseline_noise_percentage": a.noise_percentage - b.noise_percentage
                if valid
                else None,
                "adaptive_minus_baseline_raw_paragraphs": a.raw_paragraphs_retrieved
                - b.raw_paragraphs_retrieved
                if valid
                else None,
            }
        )
    return rows
