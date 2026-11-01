#!/usr/bin/env python3
"""Download or normalize one HotpotQA split into canonical project JSONL."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from reliable_rag.contracts.enums import DatasetSplit
from reliable_rag.data import prepare_split

LOGGER = logging.getLogger("reliable_rag.prepare_hotpotqa")


def _load_remote_rows(
    dataset_name: str,
    dataset_config: str,
    source_split: str,
    revision: str | None,
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The datasets dependency could not be imported. Install the project "
            "dependencies with pip install -e '.[dev]' and check the active "
            f"environment. Original error: {exc}"
        ) from exc
    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=source_split,
        revision=revision,
    )
    return [dict(row) for row in dataset]


def main() -> int:
    """Prepare a local file or Hugging Face HotpotQA split."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Local HotpotQA JSON or JSONL export")
    source.add_argument("--dataset-name", help="Hugging Face dataset name")
    parser.add_argument("--dataset-config", default="distractor")
    parser.add_argument("--source-split", default="train")
    parser.add_argument(
        "--project-split",
        choices=[split.value for split in DatasetSplit],
        required=True,
    )
    parser.add_argument("--dataset-revision", default="unspecified")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/hotpotqa"))
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--revision")
    args = parser.parse_args()

    if args.input is not None:
        rows = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("the local HotpotQA JSON root must be a list")
    else:
        rows = _load_remote_rows(
            args.dataset_name,
            args.dataset_config,
            args.source_split,
            args.revision,
        )

    if not rows:
        raise ValueError("HotpotQA source contains no rows")
    result_paths = prepare_split(
        rows,
        source_split=args.source_split,
        project_split=DatasetSplit(args.project_split),
        dataset_revision=args.dataset_revision,
        output_dir=args.output_dir,
        raw_path=args.raw_output,
    )
    for path in result_paths:
        LOGGER.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
