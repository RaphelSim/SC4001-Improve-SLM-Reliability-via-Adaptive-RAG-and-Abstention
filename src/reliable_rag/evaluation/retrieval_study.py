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
    adaptive_records = [record for record in records if record.system == "adaptive"]
    decisions = [
        (record, item)
        for record in adaptive_records
        for item in record.rounds
        if item.model_status_after_round is not None
    ]
    incomplete = [(r, rd) for r, rd in decisions if not rd.gold_complete_after_round]
    complete = [(r, rd) for r, rd in decisions if rd.gold_complete_after_round]
    false_sufficient = sum(
        rd.model_status_after_round == SufficiencyStatus.SUFFICIENT for _, rd in incomplete
    )
    false_insufficient = sum(
        rd.model_status_after_round == SufficiencyStatus.NEED_MORE_EVIDENCE for _, rd in complete
    )
    rounds_reached = defaultdict(list)
    for record in adaptive_records:
        cumulative_ids = []
        cumulative_raw = 0
        gold_ids = set(record.gold_paragraph_ids)
        for item in record.rounds:
            round_ids = [chunk.evidence_id for chunk in item.retrieval.evidence]
            cumulative_raw += len(round_ids)
            cumulative_ids = list(dict.fromkeys([*cumulative_ids, *round_ids]))
            cumulative_gold_count = len(gold_ids.intersection(cumulative_ids))
            rounds_reached[item.round_number].append(
                {
                    "round": item,
                    "gold_recall": 100 * cumulative_gold_count / len(gold_ids),
                    "unique_noise": 100
                    * (len(cumulative_ids) - cumulative_gold_count)
                    / len(cumulative_ids)
                    if cumulative_ids
                    else 0,
                    "cumulative_raw": cumulative_raw,
                }
            )
    trajectory = {}
    for round_number, observations in sorted(rounds_reached.items()):
        trajectory[str(round_number)] = {
            "questions_reaching_round": len(observations),
            "mean_cumulative_raw_paragraphs": mean(x["cumulative_raw"] for x in observations),
            "mean_cumulative_gold_recall_percentage": mean(x["gold_recall"] for x in observations),
            "mean_cumulative_unique_noise_percentage": mean(
                x["unique_noise"] for x in observations
            ),
            "new_unique_paragraphs": sum(len(x["round"].new_unique_ids) for x in observations),
            "duplicate_paragraphs": sum(len(x["round"].duplicate_ids) for x in observations),
            "zero_new_evidence_rounds": sum(not x["round"].new_unique_ids for x in observations),
        }
    attempted_query_count = sum(len(r.rounds) for r in adaptive_records)
    repeated_queries = 0
    false_sufficiency_by_round = defaultdict(lambda: [0, 0])
    for record in adaptive_records:
        seen_queries = set()
        for item in record.rounds:
            query_key = _query_key(item.retrieval.query)
            repeated_queries += query_key in seen_queries
            seen_queries.add(query_key)
            if not item.gold_complete_after_round:
                false_sufficiency_by_round[item.round_number][1] += 1
                false_sufficiency_by_round[item.round_number][0] += (
                    item.model_status_after_round == SufficiencyStatus.SUFFICIENT
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
        "stop_reason_distribution": dict(Counter(r.stop_reason for r in records if r.stop_reason)),
        "reasoning_model_calls": {
            "controller": _stats([r.controller_model_calls for r in records]),
            "repair": _stats([r.repair_model_calls for r in records]),
            "query_regeneration": _stats([r.query_regeneration_model_calls for r in records]),
            "total": _stats([r.total_reasoning_model_calls for r in records]),
        },
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
        "controller_diagnostics": {
            "gold_based_false_sufficiency": false_sufficient / len(incomplete)
            if incomplete
            else None,
            "gold_based_false_sufficiency_count": false_sufficient,
            "gold_incomplete_decisions": len(incomplete),
            "round_1_gold_based_false_sufficiency": (
                sum(
                    rd.round_number == 1
                    and rd.model_status_after_round == SufficiencyStatus.SUFFICIENT
                    for _, rd in incomplete
                )
                / sum(rd.round_number == 1 for _, rd in incomplete)
                if any(rd.round_number == 1 for _, rd in incomplete)
                else None
            ),
            "round_1_false_sufficiency_count": sum(
                rd.round_number == 1 and rd.model_status_after_round == SufficiencyStatus.SUFFICIENT
                for _, rd in incomplete
            ),
            "round_1_gold_incomplete_decisions": sum(rd.round_number == 1 for _, rd in incomplete),
            "gold_based_false_insufficiency": false_insufficient / len(complete)
            if complete
            else None,
            "gold_based_false_insufficiency_count": false_insufficient,
            "gold_complete_decisions": len(complete),
            "premature_annotated_gold_stop_questions": sum(
                r.final_model_status == SufficiencyStatus.SUFFICIENT
                and r.gold_evidence_percentage < 100
                for r in valid
                if r.system == "adaptive"
            ),
            "zero_new_evidence_round_rate": sum(
                not rd.new_unique_ids for r in adaptive_records for rd in r.rounds
            )
            / attempted_query_count
            if attempted_query_count
            else None,
            "duplicate_retrieval_rate": duplicates / raw_total if raw_total else 0,
            "query_repetition_rate": repeated_queries / attempted_query_count
            if attempted_query_count
            else None,
            "query_repetition_count": repeated_queries,
            "average_new_unique_paragraphs_per_round": mean(
                len(rd.new_unique_ids) for r in adaptive_records for rd in r.rounds
            )
            if attempted_query_count
            else None,
            "stopping_round_distribution": {str(k): len(v) for k, v in sorted(by_round.items())},
            "gold_recovery_by_round": trajectory,
            "retrieval_stagnation_count": sum(
                not rd.new_unique_ids for r in adaptive_records for rd in r.rounds
            ),
            "loop_guard_activation_count": sum(
                rd.loop_guard_activated for r in adaptive_records for rd in r.rounds
            ),
            "false_sufficiency_by_round": {
                str(k): {
                    "false_sufficient": values[0],
                    "gold_incomplete_decisions": values[1],
                    "rate": values[0] / values[1] if values[1] else None,
                }
                for k, values in sorted(false_sufficiency_by_round.items())
            },
            "final_status_distribution": dict(status_counts),
        },
    }


def _query_key(query):
    return " ".join("".join(c.lower() if c.isalnum() else " " for c in query).split())


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
