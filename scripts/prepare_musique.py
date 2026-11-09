#!/usr/bin/env python3
"""Freeze 100 MuSiQue questions with original gold and varied distractors."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from reliable_rag.config import load_settings
from reliable_rag.data.musique import GOLD_TARGETS, select_musique
from reliable_rag.data.splits import write_split_manifest
from reliable_rag.io.artifacts import file_sha256, write_json
from reliable_rag.io.jsonl import write_jsonl


def _local_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("MuSiQue local input must be a JSON array or JSONL")
    return rows


def _existing_matches(directory: Path, source: dict, seed: int) -> bool:
    try:
        manifest = json.loads((directory / "subset-development.json").read_text())
        if any(manifest["source"].get(key) != value for key, value in source.items()):
            return False
        if manifest["sampling"]["seed"] != seed:
            return False
        if manifest["sampling"]["selected_count"] != 100:
            return False
        return all(
            file_sha256(directory / filename) == info["sha256"]
            for filename, info in manifest["artifacts"].items()
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/musique_retrieval_study_v3.yaml")
    )
    parser.add_argument("--input", type=Path, help="Official MuSiQue train JSONL/JSON export")
    parser.add_argument("--revision", help="Exact Hugging Face dataset commit; resolved if omitted")
    parser.add_argument("--source-repo", help="Repository that supplied --input, for provenance")
    parser.add_argument(
        "--force", action="store_true", help="Replace a previous MuSiQue preparation"
    )
    args = parser.parse_args()
    settings = load_settings(args.config)
    if settings.dataset.dataset_name != "StonyBrookNLP/musique":
        parser.error("this adapter requires dataset_name: StonyBrookNLP/musique")
    if settings.dataset.subset_size != 100 or settings.project_seed != 88:
        parser.error("this study requires subset_size: 100 and project_seed: 88")
    if settings.dataset.source_split == "test":
        parser.error("the MuSiQue test split must not be used without support labels")
    input_path = args.input or settings.dataset.input_path
    if input_path:
        input_path = input_path.resolve()
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        source = {
            "kind": "local",
            "location": str(input_path),
            "split": settings.dataset.source_split,
            "source_sha256": file_sha256(input_path),
            "resolved_revision": args.revision or settings.dataset.revision,
            "origin_repository": args.source_repo,
        }
        if (
            settings.dataset_dir.exists()
            and not args.force
            and _existing_matches(settings.dataset_dir, source, settings.project_seed)
        ):
            print(f"Using frozen MuSiQue sample: {settings.dataset_dir}")
            return 0
        rows = _local_rows(input_path)
    else:
        from datasets import load_dataset
        from huggingface_hub import HfApi

        revision = (
            args.revision
            or settings.dataset.revision
            or HfApi().dataset_info(settings.dataset.dataset_name).sha
        )
        if not revision or len(revision) != 40:
            raise ValueError("MuSiQue remote source requires an exact 40-character commit revision")
        rows = load_dataset(
            settings.dataset.dataset_name,
            split=settings.dataset.source_split,
            revision=revision,
            cache_dir=str(settings.cache_root / "datasets"),
        )
        source = {
            "kind": "huggingface",
            "location": settings.dataset.dataset_name,
            "split": settings.dataset.source_split,
            "source_sha256": None,
            "resolved_revision": revision,
            "fingerprint": rows._fingerprint,
        }
    directory = settings.dataset_dir
    if directory.exists() and not args.force:
        if _existing_matches(directory, source, settings.project_seed):
            print(f"Using frozen MuSiQue sample: {directory}")
            return 0
        raise FileExistsError(
            f"{directory} exists with different provenance; use --force intentionally"
        )
    questions, evidence, contexts, rejected = select_musique(
        rows, seed=settings.project_seed, source_split=settings.dataset.source_split
    )
    if (
        len(questions) != 100
        or Counter(len(q.gold_paragraph_ids) for q in questions) != GOLD_TARGETS
    ):
        raise AssertionError("MuSiQue selection did not meet 34/33/33 target")
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    question_path = settings.questions_path
    evidence_path = settings.evidence_path
    split_path = directory / "manifest-development.jsonl"
    write_jsonl(question_path, questions)
    write_jsonl(evidence_path, evidence)
    write_split_manifest(
        split_path,
        [
            {
                "question_id": question.question_id,
                "source_split": settings.dataset.source_split,
                "project_split": "development",
            }
            for question in questions
        ],
    )
    artifacts = {
        path.name: {"sha256": file_sha256(path), "rows": sum(1 for _ in path.open())}
        for path in (question_path, evidence_path, split_path)
    }
    write_json(
        settings.selection_path,
        {
            "schema_version": 1,
            "source": {**source, "question_count": len(rows)},
            "project_split": "development",
            "preprocessing": {
                "version": "musique-original-support-variable-distractors-v1",
                "gold_definition": "original is_supporting paragraph annotation",
            },
            "sampling": {
                "method": "stratified_gold_count_balanced_density_v1",
                "seed": settings.project_seed,
                "requested_subset_size": 100,
                "selected_count": 100,
                "selected_question_ids": [q.question_id for q in questions],
                "gold_distribution": dict(Counter(len(q.gold_paragraph_ids) for q in questions)),
                "density_distribution": dict(Counter(x["density_band"] for x in contexts)),
                "distractor_count_distribution": dict(
                    Counter(x["distractor_count"] for x in contexts)
                ),
                "contexts": contexts,
                "rejected_source_rows": rejected,
            },
            "artifacts": artifacts,
        },
    )
    print(f"Prepared 100 MuSiQue questions in {directory}")
    print("Gold counts:", dict(Counter(len(q.gold_paragraph_ids) for q in questions)))
    print("Distractor density:", dict(Counter(x["density_band"] for x in contexts)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
