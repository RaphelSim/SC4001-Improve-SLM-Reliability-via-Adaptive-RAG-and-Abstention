#!/usr/bin/env python3
"""Prepare or reuse a deterministic, provenance-checked HotpotQA subset."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import platform
from pathlib import Path
from typing import Any

from reliable_rag.config import load_settings
from reliable_rag.contracts.enums import DatasetSplit
from reliable_rag.contracts.records import QuestionRecord
from reliable_rag.data import prepare_split, select_subset_indices
from reliable_rag.io.artifacts import file_sha256, write_json
from reliable_rag.io.jsonl import read_jsonl

LOGGER = logging.getLogger(__name__)


def _load_local_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                raise ValueError(f"Blank line in {path}:{number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object in {path}:{number}")
            rows.append(row)
        return rows
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("HotpotQA input must be a JSON list of objects or JSONL")
    return rows


def _load_remote_source(dataset_name, dataset_config, source_split, revision, *, cache_dir):
    # Arrow-backed source access avoids materializing all contexts before sampling.
    from datasets import load_dataset

    source = load_dataset(
        dataset_name,
        dataset_config,
        split=source_split,
        revision=revision,
        cache_dir=str(cache_dir),
    )
    # Datasets may fall back to its latest cached Arrow source when offline.
    # A pinned HotpotQA request must never silently use another revision.
    if revision and len(revision) == 40:
        files = source.cache_files
        if not files or any(revision not in Path(entry["filename"]).parts for entry in files):
            raise ValueError("Cached dataset does not match the requested pinned revision")
    return source


def _cache_matches(directory: Path, request: dict) -> bool:
    try:
        metadata = json.loads((directory / f"subset-{request['project_split']}.json").read_text())
        source, sampling = metadata["source"], metadata["sampling"]
        method = (
            "python_random_shuffle_prefix_v1"
            if request["subset_size"] is not None
            else "all_source_order_v1"
        )
        if sampling.get("method") != method:
            return False
        expected = request["source"]
        if any(source.get(key) != value for key, value in expected.items()):
            return False
        if metadata["project_split"] != request["project_split"]:
            return False
        if (
            sampling["seed"] != request["seed"]
            or sampling["requested_subset_size"] != request["subset_size"]
        ):
            return False
        for name, info in metadata["artifacts"].items():
            if Path(name).name != name or file_sha256(directory / name) != info["sha256"]:
                return False
        for prefix in ["questions", "evidence", "manifest"]:
            if f"{prefix}-{request['project_split']}.jsonl" not in metadata["artifacts"]:
                return False
        questions = read_jsonl(
            directory / f"questions-{request['project_split']}.jsonl", QuestionRecord
        )
        ids = [question.question_id for question in questions]
        return (
            ids == sampling["selected_question_ids"]
            and len(ids) == sampling["selected_count"]
            and len(ids) == len(set(ids))
            and (request["subset_size"] is None or len(ids) == request["subset_size"])
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--subset-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--input", type=Path, help="Override dataset.input_path with a local export"
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Override the derived prepared-data directory"
    )
    parser.add_argument(
        "--manifest-dir", type=Path, help="Also publish an ID-only selection manifest"
    )
    parser.add_argument(
        "--force", action="store_true", help="Regenerate even if the prepared cache matches"
    )
    args = parser.parse_args()
    settings = load_settings(args.config)
    if args.subset_size is not None:
        if args.subset_size < 1:
            parser.error("--subset-size must be positive")
        settings.dataset.subset_size = args.subset_size
    if args.seed is not None:
        if args.seed < 0:
            parser.error("--seed must be non-negative")
        settings.project_seed = args.seed
    dataset = settings.dataset
    input_path = args.input or dataset.input_path
    directory = args.output_dir or settings.dataset_dir
    request = {
        "source": {
            "kind": "local" if input_path else "huggingface",
            "location": str(input_path) if input_path else dataset.dataset_name,
            "config": dataset.dataset_config,
            "split": dataset.source_split,
            "requested_revision": dataset.revision,
            "dataset_revision": dataset.revision or "unspecified",
            "input_sha256": file_sha256(input_path) if input_path else None,
        },
        "project_split": dataset.project_split,
        "seed": settings.project_seed,
        "subset_size": dataset.subset_size,
    }
    selection = directory / f"subset-{dataset.project_split}.json"
    if not args.force and _cache_matches(directory, request):
        LOGGER.info("Using cached prepared dataset: %s", directory)
    else:
        if directory.exists():
            LOGGER.info(
                "Prepared cache does not match requested source/config, or --force supplied. Regenerating %s",
                directory,
            )
        source = (
            _load_local_rows(input_path)
            if input_path
            else _load_remote_source(
                dataset.dataset_name,
                dataset.dataset_config,
                dataset.source_split,
                dataset.revision,
                cache_dir=settings.cache_root / "datasets",
            )
        )
        indices = select_subset_indices(
            len(source), subset_size=dataset.subset_size, seed=settings.project_seed
        )
        rows = [dict(source[i]) for i in indices]
        ids = [str(row.get("_id", row.get("id", ""))).strip() for row in rows]
        if not all(ids) or len(ids) != len(set(ids)):
            raise ValueError("Selected question IDs must be nonempty and unique")
        if input_path and input_path.resolve() in {
            selection.resolve(),
            *(
                (directory / f"{prefix}-{dataset.project_split}.jsonl").resolve()
                for prefix in ["questions", "evidence", "manifest"]
            ),
        }:
            raise ValueError("Prepared outputs must not overwrite the input file")
        paths = prepare_split(
            rows,
            source_split=dataset.source_split,
            project_split=DatasetSplit(dataset.project_split),
            dataset_revision=dataset.revision or "unspecified",
            output_dir=directory,
        )
        metadata = {
            "schema_version": 1,
            "source": {
                **request["source"],
                "question_count": len(source),
                "fingerprint": getattr(source, "_fingerprint", None),
            },
            "sampling": {
                "method": "python_random_shuffle_prefix_v1"
                if dataset.subset_size is not None
                else "all_source_order_v1",
                "seed": settings.project_seed,
                "requested_subset_size": dataset.subset_size,
                "selected_count": len(rows),
                "selected_source_indices": indices,
                "selected_question_ids": ids,
                "python_version": platform.python_version(),
                "datasets_version": importlib.metadata.version("datasets"),
            },
            "project_split": dataset.project_split,
            "artifacts": {
                path.name: {
                    "sha256": file_sha256(path),
                    "rows": sum(1 for _ in path.open(encoding="utf-8")),
                }
                for path in paths
            },
        }
        write_json(selection, metadata)
        LOGGER.info(
            "Prepared %s/%s questions (seed %s): %s",
            len(rows),
            len(source),
            settings.project_seed,
            directory,
        )
    if args.manifest_dir:
        metadata = json.loads(selection.read_text())
        published = (
            args.manifest_dir
            / f"hotpotqa-subset-{metadata['sampling']['selected_count']}-seed{settings.project_seed}-{dataset.project_split}.json"
        )
        if input_path and published.resolve() == input_path.resolve():
            raise ValueError("Selection manifest must not overwrite the input")
        write_json(published, metadata)
        LOGGER.info("Selection manifest: %s", published)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
