"""HotpotQA loading and normalization into canonical project records."""

from __future__ import annotations

import json
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reliable_rag.contracts.enums import DatasetSplit
from reliable_rag.contracts.ids import make_evidence_id
from reliable_rag.contracts.records import EvidenceChunk, QuestionRecord


@dataclass(frozen=True)
class HotpotQAExample:
    """A normalized question together with its complete candidate evidence."""

    question: QuestionRecord
    evidence: tuple[EvidenceChunk, ...]


def _document_id(title: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", title.strip().lower()).strip("-")
    if not normalized:
        normalized = f"document-{hashlib.sha1(title.encode('utf-8')).hexdigest()[:12]}"
    return normalized


def _unique_document_ids(titles: list[str]) -> list[str]:
    """Create stable IDs when distinct titles share the same normalized slug."""
    counts: dict[str, int] = {}
    for title in titles:
        base = _document_id(title)
        counts[base] = counts.get(base, 0) + 1
    used: set[str] = set()
    document_ids: list[str] = []
    for title in titles:
        base = _document_id(title)
        if counts[base] == 1:
            candidate = base
        else:
            suffix = hashlib.sha1(title.encode("utf-8")).hexdigest()[:10]
            candidate = f"{base}--{suffix}"
        if candidate in used:
            occurrence = 2
            while f"{candidate}--{occurrence}" in used:
                occurrence += 1
            candidate = f"{candidate}--{occurrence}"
        used.add(candidate)
        document_ids.append(candidate)
    return document_ids


def _read_examples(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix == ".jsonl":
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    else:
        value = json.loads(source.read_text(encoding="utf-8"))
        rows = value if isinstance(value, list) else [value]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"HotpotQA file {source} must contain object records")
    return rows


def _paragraph_text(sentences: list[str]) -> str:
    if not sentences or not all(isinstance(sentence, str) for sentence in sentences):
        raise ValueError("HotpotQA context paragraphs must contain sentence strings")
    text = " ".join(sentence.strip() for sentence in sentences).strip()
    if not text:
        raise ValueError("HotpotQA context paragraphs must not be empty")
    return text


def normalize_hotpotqa_example(
    raw: dict[str, Any],
    *,
    split: DatasetSplit,
    source_split: str | None = None,
    dataset_revision: str | None = None,
) -> HotpotQAExample:
    """Normalize one HotpotQA JSON record and retain all candidate paragraphs."""
    question_id = str(raw.get("_id", raw.get("id", ""))).strip()
    question_text = str(raw.get("question", "")).strip()
    gold_answer = str(raw.get("answer", "")).strip()
    context = raw.get("context")
    supporting_facts = raw.get("supporting_facts")
    if not question_id or not question_text or not gold_answer:
        raise ValueError("HotpotQA records require _id, question, and answer")
    if isinstance(context, dict) and {"title", "sentences"} <= context.keys():
        context = list(zip(context["title"], context["sentences"]))
    if isinstance(supporting_facts, dict) and {"title", "sent_id"} <= supporting_facts.keys():
        supporting_facts = list(zip(supporting_facts["title"], supporting_facts["sent_id"]))
    if not isinstance(context, list) or not isinstance(supporting_facts, list):
        raise ValueError("HotpotQA records require context and supporting_facts collections")

    support_locations = {
        (str(item[0]), int(item[1]))
        for item in supporting_facts
        if isinstance(item, (list, tuple)) and len(item) == 2
    }
    if not support_locations:
        raise ValueError(f"HotpotQA record {question_id} has no valid supporting facts")

    titles = [paragraph[0] for paragraph in context if isinstance(paragraph, (list, tuple))]
    if len(titles) != len(context) or not all(isinstance(title, str) for title in titles):
        raise ValueError(f"HotpotQA record {question_id} has invalid context titles")
    document_ids = _unique_document_ids(titles)
    evidence: list[EvidenceChunk] = []
    support_ids: list[str] = []
    for paragraph, document_id in zip(context, document_ids):
        if not isinstance(paragraph, (list, tuple)) or len(paragraph) != 2:
            raise ValueError(f"HotpotQA record {question_id} has an invalid context paragraph")
        title, sentences = paragraph
        if not isinstance(title, str) or not isinstance(sentences, list):
            raise ValueError(f"HotpotQA record {question_id} has invalid context fields")
        evidence_id = make_evidence_id(document_id, 0)
        chunk = EvidenceChunk(
            evidence_id=evidence_id,
            document_id=document_id,
            title=title.strip(),
            text=_paragraph_text(sentences),
        )
        evidence.append(chunk)
        has_support = any(
            location_title == title
            and 0 <= location_index < len(sentences)
            for location_title, location_index in support_locations
        )
        if has_support:
            support_ids.append(evidence_id)

    if len(set(support_ids)) != len({title for title, _ in support_locations}):
        raise ValueError(f"HotpotQA record {question_id} support facts do not map to context")

    metadata = {
        "source_dataset": "hotpotqa",
        "fixture_only": False,
    }
    if source_split is not None:
        metadata["source_split"] = source_split
    if dataset_revision is not None:
        metadata["dataset_revision"] = dataset_revision
    question = QuestionRecord(
        question_id=question_id,
        question=question_text,
        gold_answer=gold_answer,
        split=split,
        gold_support_ids=support_ids,
        candidate_evidence_ids=[chunk.evidence_id for chunk in evidence],
        metadata=metadata,
    )
    return HotpotQAExample(question=question, evidence=tuple(evidence))


def load_hotpotqa(
    path: str | Path,
    *,
    split: DatasetSplit,
    source_split: str | None = None,
    dataset_revision: str | None = None,
) -> list[HotpotQAExample]:
    """Load and normalize a JSON or JSONL HotpotQA export."""
    return [
        normalize_hotpotqa_example(
            raw,
            split=split,
            source_split=source_split,
            dataset_revision=dataset_revision,
        )
        for raw in _read_examples(path)
    ]
