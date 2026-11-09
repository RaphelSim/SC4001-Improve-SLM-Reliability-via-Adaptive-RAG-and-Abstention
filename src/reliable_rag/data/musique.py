"""Convert a fixed MuSiQue sample into the existing paragraph retrieval records."""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from collections.abc import Iterable
from typing import Any

from reliable_rag.contracts.enums import DatasetSplit
from reliable_rag.contracts.ids import make_evidence_id
from reliable_rag.contracts.records import EvidenceChunk, QuestionRecord

GOLD_TARGETS = {2: 34, 3: 33, 4: 33}
DENSITY_BANDS = {"low": (6, 9), "medium": (10, 13), "high": (14, 18)}


def _paragraphs(value: Any) -> list[dict[str, Any]]:
    """Accept both ordinary JSON rows and Hugging Face struct-of-lists rows."""
    if isinstance(value, dict):
        keys = ("idx", "title", "paragraph_text", "is_supporting")
        if not all(isinstance(value.get(key), list) for key in keys):
            return []
        try:
            value = [
                dict(zip(keys, parts, strict=True))
                for parts in zip(*(value[k] for k in keys), strict=True)
            ]
        except ValueError:
            return []
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        return []
    return value


def select_musique(
    rows: Iterable[dict[str, Any]], *, seed: int = 88, source_split: str = "train"
) -> tuple[list[QuestionRecord], list[EvidenceChunk], list[dict[str, Any]], dict[str, int]]:
    """Select 34/33/33 original-support questions and vary distractor density.

    All random choices use one seed-controlled generator. The returned context
    manifest records the source paragraph indices and final order exactly.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")
    rng = random.Random(seed)
    eligible: dict[int, list[dict[str, Any]]] = {count: [] for count in GOLD_TARGETS}
    rejected: Counter[str] = Counter()
    seen_questions: set[str] = set()
    seen_question_texts: set[str] = set()
    for source_index, row in enumerate(rows):
        if not isinstance(row, dict):
            rejected["malformed_row"] += 1
            continue
        question_id = str(row.get("id", row.get("_id", ""))).strip()
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        if not question_id or not question or not answer or row.get("answerable") is False:
            rejected["missing_answer_or_unanswerable"] += 1
            continue
        if question_id in seen_questions or question.casefold() in seen_question_texts:
            rejected["duplicate_question"] += 1
            continue
        seen_questions.add(question_id)
        seen_question_texts.add(question.casefold())
        paragraphs = _paragraphs(row.get("paragraphs"))
        if not paragraphs:
            rejected["malformed_paragraphs"] += 1
            continue
        indices = [paragraph.get("idx") for paragraph in paragraphs]
        if any(not isinstance(index, int) for index in indices) or len(indices) != len(
            set(indices)
        ):
            rejected["invalid_paragraph_indices"] += 1
            continue
        if any(
            not isinstance(paragraph.get("paragraph_text"), str)
            or not paragraph["paragraph_text"].strip()
            or not isinstance(paragraph.get("is_supporting"), bool)
            for paragraph in paragraphs
        ):
            rejected["malformed_paragraphs"] += 1
            continue
        gold = [paragraph for paragraph in paragraphs if paragraph["is_supporting"]]
        distractors = [paragraph for paragraph in paragraphs if not paragraph["is_supporting"]]
        if len(gold) not in GOLD_TARGETS:
            rejected["gold_count_outside_2_to_4"] += 1
            continue
        if len(distractors) < 6:
            rejected["fewer_than_six_distractors"] += 1
            continue
        eligible[len(gold)].append(
            {
                "source_index": source_index,
                "question_id": question_id,
                "question": question,
                "answer": answer,
                "gold": gold,
                "distractors": distractors,
            }
        )

    selected: list[tuple[dict[str, Any], str, int]] = []
    for gold_count, target in GOLD_TARGETS.items():
        available = eligible[gold_count]
        rng.shuffle(available)
        band_counts = {"low": target // 3, "medium": target // 3, "high": target // 3}
        for band in ("high", "medium", "low")[: target % 3]:
            band_counts[band] += 1
        for band in ("high", "medium", "low"):
            lower, upper = DENSITY_BANDS[band]
            for _ in range(band_counts[band]):
                match = next(
                    (
                        index
                        for index, item in enumerate(available)
                        if len(item["distractors"]) >= lower
                    ),
                    None,
                )
                if match is None:
                    raise ValueError(
                        f"MuSiQue has too few valid {gold_count}-gold questions for {band} density"
                    )
                item = available.pop(match)
                distractor_count = rng.randint(lower, min(upper, len(item["distractors"])))
                selected.append((item, band, distractor_count))
    rng.shuffle(selected)

    questions: list[QuestionRecord] = []
    evidence: list[EvidenceChunk] = []
    contexts: list[dict[str, Any]] = []
    for item, band, distractor_count in selected:
        chosen_distractors = rng.sample(item["distractors"], distractor_count)
        context = [*item["gold"], *chosen_distractors]
        rng.shuffle(context)
        digest = hashlib.sha256(item["question_id"].encode()).hexdigest()[:20]
        candidate_ids = []
        gold_ids = []
        distractor_ids = []
        for paragraph in context:
            index = paragraph["idx"]
            document_id = f"musique-{digest}-paragraph-{index}"
            evidence_id = make_evidence_id(document_id, 0)
            candidate_ids.append(evidence_id)
            (gold_ids if paragraph["is_supporting"] else distractor_ids).append(evidence_id)
            evidence.append(
                EvidenceChunk(
                    evidence_id=evidence_id,
                    document_id=document_id,
                    title=str(paragraph.get("title") or f"Paragraph {index}"),
                    text=paragraph["paragraph_text"].strip(),
                )
            )
        question = QuestionRecord(
            question_id=item["question_id"],
            question=item["question"],
            gold_answer=item["answer"],
            split=DatasetSplit.DEVELOPMENT,
            gold_support_ids=gold_ids,
            gold_paragraph_ids=gold_ids,
            candidate_evidence_ids=candidate_ids,
            metadata={
                "source_dataset": "musique",
                "source_split": source_split,
                "source_index": item["source_index"],
                "gold_count": len(gold_ids),
                "distractor_count": len(distractor_ids),
                "distractor_density_band": band,
                "original_gold_paragraph_indices": [p["idx"] for p in item["gold"]],
                "selected_distractor_indices": [p["idx"] for p in chosen_distractors],
                "original_candidate_count": len(item["gold"]) + len(item["distractors"]),
            },
        )
        questions.append(question)
        contexts.append(
            {
                "question_id": item["question_id"],
                "source_index": item["source_index"],
                "gold_count": len(gold_ids),
                "distractor_count": len(distractor_ids),
                "density_band": band,
                "gold_paragraph_ids": gold_ids,
                "distractor_paragraph_ids": distractor_ids,
                "candidate_paragraph_ids_in_order": candidate_ids,
                "original_paragraph_indices_in_order": [p["idx"] for p in context],
            }
        )
    return questions, evidence, contexts, dict(rejected)
