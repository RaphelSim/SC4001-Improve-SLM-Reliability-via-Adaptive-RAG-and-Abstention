#!/usr/bin/env python3
"""Evaluate persisted retrieval and baseline decisions with explicit denominators."""

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from reliable_rag.config import load_settings
from reliable_rag.contracts.records import ExperimentRecord, QuestionRecord, RetrievalResult
from reliable_rag.evaluation import metrics
from reliable_rag.io.artifacts import file_sha256, write_json
from reliable_rag.io.jsonl import read_jsonl
from reliable_rag.run_artifacts import run_directory


def rate(numerator, denominator):
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
    }


def mean(total, count):
    return {
        "total": total,
        "count": count,
        "value": total / count if count else None,
    }


def evaluate(questions, retrieval, records, errors, *, attempted, ks):
    if not questions or [q.question_id for q in questions] != [r.question_id for r in retrieval]:
        raise ValueError("Question and retrieval IDs/counts must align and be nonempty")
    if len({q.question_id for q in questions}) != len(questions):
        raise ValueError("Question IDs must be unique")
    report = {"retrieval": {}}
    for k in ks:
        if k < 1 or any(result.top_k < k for result in retrieval):
            raise ValueError(
                f"Cannot evaluate k={k}: exceeds saved retrieval depth or is nonpositive"
            )
        values = [
            metrics.supporting_evidence_recall_at_k(
                q.gold_support_ids, [chunk.evidence_id for chunk in r.evidence], k
            )
            for q, r in zip(questions, retrieval, strict=True)
        ]
        report["retrieval"][f"recall@{k}"] = rate(
            sum(m.numerator for m in values), sum(m.denominator for m in values)
        )
    known = {q.question_id for q in questions}
    record_ids = [r.question.question_id for r in records]
    if len(set(record_ids)) != len(record_ids) or not set(record_ids).issubset(known):
        raise ValueError("Experiment records must have unique IDs from the evaluated questions")
    if not len(records) <= attempted <= len(questions):
        raise ValueError("Attempted/recorded counts are inconsistent")
    classification = metrics.sufficiency_classification_metrics(
        [record.gold_sufficient_finally for record in records],
        [record.rounds[-1].sufficiency_decision.status for record in records],
    )
    tp, fp, tn, fn = (
        classification.true_positive,
        classification.false_positive,
        classification.true_negative,
        classification.false_negative,
    )
    report["sufficiency"] = {
        "scope": "records_with_valid_assessments",
        "positive_class": "need_more_evidence",
        "counts": asdict(classification),
        "accuracy": rate(tp + tn, len(records)),
        "precision": rate(tp, tp + fp),
        "recall": rate(tp, tp + fn),
        "f1": rate(2 * tp, 2 * tp + fp + fn),
        "false_sufficiency_rate": rate(fn, tp + fn),
        "false_insufficiency_rate": rate(fp, fp + tn),
    }
    report["answers"] = {
        name: {**asdict(value := function(records)), "value": value.value}
        for name, function in (
            ("coverage_recorded", metrics.answer_coverage),
            ("correct_abstention_rate", metrics.correct_abstention_rate),
            ("false_abstention_rate", metrics.false_abstention_rate),
            ("unsupported_answer_rate", metrics.unsupported_answer_rate),
            (
                "unsupported_answer_rate_among_answered",
                metrics.unsupported_answer_rate_among_answered,
            ),
        )
    }
    report["answers"]["coverage_attempted"] = rate(
        sum(not r.final_answer.abstained for r in records), attempted
    )

    exact = metrics.answer_exact_match(records)
    f1 = metrics.answer_token_f1(records)

    report["answer_correctness"] = {
        "normalization": "hotpotqa_official",
        "exact_match_answered": rate(exact.numerator, exact.denominator),
        "exact_match_recorded": rate(exact.numerator, len(records)),
        "exact_match_attempted": rate(exact.numerator, attempted),
        "token_f1_answered": mean(f1.total, f1.count),
        "token_f1_recorded": mean(f1.total, len(records)),
        "token_f1_attempted": mean(f1.total, attempted),
    }

    recovery = metrics.recovery_rate(records)
    followup = metrics.followup_support_recall(records)

    report["adaptive"] = {
        "recovery_rate": rate(recovery.numerator, recovery.denominator),
        "followup_support_recall": rate(followup.numerator, followup.denominator),
        "retrieval_rounds": {
            str(rounds): count
            for rounds, count in sorted(Counter(len(r.rounds) for r in records).items())
        },
    }

    report["abstention_reasons"] = dict(
        Counter(r.final_answer.abstention_reason.value for r in records if r.final_answer.abstained)
    )
    report["failures"] = {
        "model_contract_failures": sum(e.get("category") != "runtime_error" for e in errors),
        "unrecorded_assessment_failures": sum(
            e.get("category") != "runtime_error" and not e.get("record_written", False)
            for e in errors
        ),
        "runtime_errors": sum(e.get("category") == "runtime_error" for e in errors),
        "by_task": dict(Counter(e.get("task", "runtime") for e in errors)),
        "by_reason": dict(
            Counter(e.get("message", e.get("reason", "unspecified")) for e in errors)
        ),
    }
    report["counts"] = {
        "planned": len(questions),
        "attempted": attempted,
        "records": len(records),
        "not_started": len(questions) - attempted,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--output", type=Path, help="Default: outputs/<run-id>/evaluation.json")
    args = parser.parse_args()
    settings = load_settings(args.config)
    directory = run_directory(settings.output_root, args.run_id)
    modern = (directory / "run_metadata.json").exists()
    metadata_path = directory / ("run_metadata.json" if modern else "baseline.manifest.json")
    metadata = json.loads(metadata_path.read_text())
    # Existing baseline runs remain readable without rewriting historical artifacts.
    questions_path = Path(metadata["questions_path"] if modern else next(iter(metadata["inputs"])))
    if file_sha256(questions_path) != metadata["inputs"][str(questions_path)]:
        raise ValueError("Prepared questions changed since this run; restore the recorded input")
    retrieval_path = directory / ("retrieval.jsonl" if modern else "initial-retrieval.jsonl")
    results_path = directory / ("results.jsonl" if modern else "baseline.jsonl")
    errors_path = directory / ("errors.jsonl" if modern else "baseline.errors.jsonl")
    if modern and metadata.get("retrieval_sha256") != file_sha256(retrieval_path):
        raise ValueError("Saved retrieval changed since this run")
    questions = read_jsonl(questions_path, QuestionRecord)
    retrieval = read_jsonl(retrieval_path, RetrievalResult)
    questions = questions[: len(retrieval)]
    records = read_jsonl(results_path, ExperimentRecord)
    errors = [json.loads(line) for line in errors_path.read_text().splitlines()]
    report = evaluate(
        questions, retrieval, records, errors, attempted=metadata["summary"]["attempted"], ks=args.k
    )
    report.update(
        {
            "run_id": args.run_id,
            "run_status": metadata.get("status", "legacy_completed"),
            "source_sha256": {
                str(path): file_sha256(path)
                for path in (metadata_path, retrieval_path, results_path, errors_path)
            },
        }
    )
    output = args.output or directory / "evaluation.json"
    protected = [metadata_path, questions_path, retrieval_path, results_path, errors_path]
    if output.resolve() in {p.resolve() for p in protected}:
        raise ValueError("Evaluation output must not overwrite scientific inputs")
    write_json(output, report)
    print(json.dumps(report, indent=2))
    print(f"Evaluation saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
