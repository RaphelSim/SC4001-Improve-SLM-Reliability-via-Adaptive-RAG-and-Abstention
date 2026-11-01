"""Persistence helpers for normalized HotpotQA split artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from reliable_rag.contracts.enums import DatasetSplit
from reliable_rag.contracts.records import EvidenceChunk
from reliable_rag.data.hotpotqa import normalize_hotpotqa_example
from reliable_rag.data.splits import SplitManifestRow, write_split_manifest
from reliable_rag.io.jsonl import write_jsonl


def prepare_split(
    rows: list[dict[str, Any]],
    *,
    source_split: str,
    project_split: DatasetSplit,
    dataset_revision: str,
    output_dir: Path,
    raw_path: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Normalize rows and write questions, evidence, and an ID-only manifest."""
    if raw_path is not None:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        import json

        raw_path.write_text(json.dumps(rows, ensure_ascii=False) + "\n", encoding="utf-8")
    examples = [
        normalize_hotpotqa_example(
            row,
            split=project_split,
            source_split=source_split,
            dataset_revision=dataset_revision,
        )
        for row in rows
    ]
    by_id: dict[str, EvidenceChunk] = {}
    normalized_examples = []
    for example in examples:
        replacements: dict[str, str] = {}
        normalized_evidence: list[EvidenceChunk] = []
        for chunk in example.evidence:
            previous = by_id.get(chunk.evidence_id)
            if previous is None:
                by_id[chunk.evidence_id] = chunk
                normalized_evidence.append(chunk)
                continue
            if previous == chunk:
                normalized_evidence.append(chunk)
                continue
            suffix = hashlib.sha1(chunk.text.encode("utf-8")).hexdigest()[:10]
            replacement_id = f"{chunk.document_id}--{suffix}::{chunk.evidence_id.rsplit('::', 1)[1]}"
            replacement = chunk.model_copy(
                update={
                    "document_id": f"{chunk.document_id}--{suffix}",
                    "evidence_id": replacement_id,
                }
            )
            existing_replacement = by_id.get(replacement_id)
            if existing_replacement is not None and existing_replacement != replacement:
                raise ValueError(f"Conflicting evidence content for ID {replacement_id}")
            by_id[replacement_id] = replacement
            replacements[chunk.evidence_id] = replacement_id
            normalized_evidence.append(replacement)
        if replacements:
            question = example.question.model_copy(
                update={
                    "gold_support_ids": [
                        replacements.get(evidence_id, evidence_id)
                        for evidence_id in example.question.gold_support_ids
                    ],
                    "candidate_evidence_ids": [
                        replacements.get(evidence_id, evidence_id)
                        for evidence_id in example.question.candidate_evidence_ids
                    ],
                }
            )
            normalized_examples.append(
                type(example)(question=question, evidence=tuple(normalized_evidence))
            )
        else:
            normalized_examples.append(example)

    output_dir.mkdir(parents=True, exist_ok=True)
    questions_path = output_dir / f"questions-{project_split.value}.jsonl"
    evidence_path = output_dir / f"evidence-{project_split.value}.jsonl"
    manifest_path = output_dir / f"manifest-{project_split.value}.jsonl"
    write_jsonl(questions_path, [example.question for example in normalized_examples])
    write_jsonl(evidence_path, sorted(by_id.values(), key=lambda chunk: chunk.evidence_id))
    manifest: list[SplitManifestRow] = [
        {
            "question_id": example.question.question_id,
            "source_split": source_split,
            "project_split": project_split.value,
        }
        for example in normalized_examples
    ]
    write_split_manifest(manifest_path, manifest)
    return questions_path, evidence_path, manifest_path
