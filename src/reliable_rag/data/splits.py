"""Deterministic question-ID split manifests."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TypedDict


class SplitManifestRow(TypedDict):
    """One reproducible project split assignment."""

    question_id: str
    source_split: str
    project_split: str


def build_split_manifest(
    question_ids: list[str],
    *,
    source_split: str,
    seed: int = 42,
    development_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> list[SplitManifestRow]:
    """Assign unique question IDs to development, validation, and test deterministically."""
    if not question_ids or len(question_ids) != len(set(question_ids)):
        raise ValueError("question_ids must be non-empty and unique")
    if not 0 < development_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be between zero and one")
    if development_fraction + validation_fraction >= 1:
        raise ValueError("development and validation fractions must leave test examples")

    shuffled = list(question_ids)
    random.Random(seed).shuffle(shuffled)
    development_count = int(len(shuffled) * development_fraction)
    validation_count = int(len(shuffled) * validation_fraction)
    if (
        development_count == 0
        or validation_count == 0
        or development_count + validation_count >= len(shuffled)
    ):
        raise ValueError(
            "input is too small for non-empty development, validation, and test splits"
        )

    rows: list[SplitManifestRow] = []
    for index, question_id in enumerate(shuffled):
        project_split = (
            "development"
            if index < development_count
            else "validation" if index < development_count + validation_count else "test"
        )
        rows.append(
            {
                "question_id": question_id,
                "source_split": source_split,
                "project_split": project_split,
            }
        )
    return sorted(rows, key=lambda row: row["question_id"])


def write_split_manifest(path: str | Path, rows: list[SplitManifestRow]) -> None:
    """Write an ID-only split manifest as deterministic JSONL."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
