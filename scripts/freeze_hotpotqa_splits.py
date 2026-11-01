#!/usr/bin/env python3
"""Freeze deterministic project splits from a prepared HotpotQA question file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reliable_rag.contracts.enums import DatasetSplit
from reliable_rag.contracts.records import QuestionRecord
from reliable_rag.data.splits import build_split_manifest, write_split_manifest
from reliable_rag.io.jsonl import read_jsonl, write_jsonl


def main() -> int:
    """Write split-specific question JSONL files and one manifest per split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--development-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    args = parser.parse_args()

    questions = read_jsonl(args.questions, QuestionRecord)
    manifest = build_split_manifest(
        [question.question_id for question in questions],
        source_split=args.source_split,
        seed=args.seed,
        development_fraction=args.development_fraction,
        validation_fraction=args.validation_fraction,
    )
    assignments = {row["question_id"]: row["project_split"] for row in manifest}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in DatasetSplit:
        split_questions = [
            question.model_copy(update={"split": split})
            for question in questions
            if assignments[question.question_id] == split.value
        ]
        write_jsonl(args.output_dir / f"questions-{split.value}.jsonl", split_questions)
        split_manifest = [
            row for row in manifest if row["project_split"] == split.value
        ]
        write_split_manifest(args.output_dir / f"manifest-{split.value}.jsonl", split_manifest)
    counts = {
        split.value: sum(row["project_split"] == split.value for row in manifest)
        for split in DatasetSplit
    }
    print(json.dumps({"seed": args.seed, "counts": counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
