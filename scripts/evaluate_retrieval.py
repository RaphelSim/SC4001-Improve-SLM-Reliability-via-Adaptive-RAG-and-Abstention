#!/usr/bin/env python3
"""Evaluate supporting-evidence Recall@k for retrieval JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reliable_rag.contracts.records import QuestionRecord, RetrievalResult
from reliable_rag.evaluation.metrics import supporting_evidence_recall_at_k
from reliable_rag.io.jsonl import read_jsonl


def main() -> int:
    """Report micro-aggregated support hits and denominators for each k."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", type=Path)
    parser.add_argument("retrieval", type=Path)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    questions = read_jsonl(args.questions, QuestionRecord)
    retrieval = read_jsonl(args.retrieval, RetrievalResult)
    if len(questions) != len(retrieval):
        raise ValueError("question and retrieval counts do not match")
    if [question.question_id for question in questions] != [
        result.question_id for result in retrieval
    ]:
        raise ValueError("question and retrieval IDs are not aligned")

    report = []
    for k in args.k:
        numerator = 0
        denominator = 0
        for question, result in zip(questions, retrieval):
            metric = supporting_evidence_recall_at_k(
                question.gold_support_ids,
                [chunk.evidence_id for chunk in result.evidence],
                k,
            )
            numerator += metric.numerator
            denominator += metric.denominator
        report.append(
            {
                "k": k,
                "numerator": numerator,
                "denominator": denominator,
                "value": numerator / denominator if denominator else None,
            }
        )
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
