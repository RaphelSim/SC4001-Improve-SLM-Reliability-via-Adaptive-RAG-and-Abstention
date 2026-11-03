"""Tests for data and retrieval data normalization and evidence construction."""

import pytest

from reliable_rag.contracts.enums import DatasetSplit, EvidenceCondition
from reliable_rag.data import (
    build_split_manifest,
    construct_evidence_set,
    normalize_hotpotqa_example,
    select_subset_indices,
)


def _raw_example() -> dict:
    return {
        "_id": "q1",
        "question": "Where is the manufacturer based?",
        "answer": "Germany",
        "supporting_facts": [["Lost Gravity", 0], ["Mack Rides", 0]],
        "context": [
            ["Lost Gravity", ["Lost Gravity was manufactured by Mack Rides."]],
            ["Mack Rides", ["Mack Rides is headquartered in Germany."]],
            ["Distractor", ["This fact is unrelated."]],
        ],
    }


def test_hotpotqa_normalization_preserves_support_and_candidates() -> None:
    example = normalize_hotpotqa_example(_raw_example(), split=DatasetSplit.DEVELOPMENT)
    assert example.question.gold_support_ids == ["lost-gravity::0", "mack-rides::0"]
    assert example.question.candidate_evidence_ids == [
        "lost-gravity::0",
        "mack-rides::0",
        "distractor::0",
    ]
    assert example.evidence[0].text.endswith("Mack Rides.")


def test_hotpotqa_normalization_accepts_huggingface_nested_shape() -> None:
    raw = _raw_example()
    raw["id"] = raw.pop("_id")
    raw["context"] = {
        "title": [title for title, _ in raw["context"]],
        "sentences": [sentences for _, sentences in raw["context"]],
    }
    raw["supporting_facts"] = {
        "title": ["Lost Gravity", "Mack Rides"],
        "sent_id": [0, 0],
    }

    example = normalize_hotpotqa_example(raw, split=DatasetSplit.DEVELOPMENT)

    assert example.question.question_id == "q1"
    assert example.question.gold_support_ids == ["lost-gravity::0", "mack-rides::0"]


def test_hotpotqa_normalization_disambiguates_colliding_title_slugs() -> None:
    raw = _raw_example()
    raw["context"].append(["Cover-Version", ["A second paragraph."]])
    raw["context"].append(["Cover Version", ["A third paragraph."]])

    example = normalize_hotpotqa_example(raw, split=DatasetSplit.DEVELOPMENT)

    assert len(example.question.candidate_evidence_ids) == 5
    assert len(set(example.question.candidate_evidence_ids)) == 5


def test_hotpotqa_normalization_handles_punctuation_only_titles() -> None:
    raw = _raw_example()
    raw["context"].append(["!!!", ["A paragraph with an unusual title."]])

    example = normalize_hotpotqa_example(raw, split=DatasetSplit.DEVELOPMENT)

    assert any(chunk.evidence_id.startswith("document-") for chunk in example.evidence)


def test_evidence_conditions_are_deterministic_and_keep_distractors() -> None:
    example = normalize_hotpotqa_example(_raw_example(), split=DatasetSplit.DEVELOPMENT)
    partial_a = construct_evidence_set(example, EvidenceCondition.PARTIALLY_INCOMPLETE, seed=7)
    partial_b = construct_evidence_set(example, EvidenceCondition.PARTIALLY_INCOMPLETE, seed=7)
    complete = construct_evidence_set(example, EvidenceCondition.COMPLETE)
    full = construct_evidence_set(example, EvidenceCondition.FULLY_INCOMPLETE)

    assert partial_a == partial_b
    assert not partial_a.gold_sufficient
    assert complete.gold_sufficient
    assert not full.gold_sufficient
    assert "distractor::0" in partial_a.evidence_ids
    assert set(full.evidence_ids) == {"distractor::0"}


def test_split_manifest_is_deterministic_and_mutually_exclusive() -> None:
    question_ids = [f"q{index}" for index in range(10)]
    first = build_split_manifest(question_ids, source_split="train", seed=9)
    second = build_split_manifest(question_ids, source_split="train", seed=9)

    assert first == second
    assignments = [row["project_split"] for row in first]
    assert assignments.count("development") == 8
    assert assignments.count("validation") == 1
    assert assignments.count("test") == 1


def test_subset_sampling_is_seeded_unique_nested_and_preserves_legacy_full_order():
    first = select_subset_indices(1000, subset_size=500)
    larger = select_subset_indices(1000, subset_size=700, seed=88)
    assert len(first) == len(set(first)) == 500
    assert first == select_subset_indices(1000, subset_size=500, seed=88)
    assert first == larger[:500]
    assert first != select_subset_indices(1000, subset_size=500, seed=89)
    assert select_subset_indices(10) == list(range(10))


@pytest.mark.parametrize("count,size,seed", [(0, None, 88), (10, 0, 88), (10, 11, 88), (10, 3, -1)])
def test_subset_sampling_rejects_invalid_sizes_and_seeds(count, size, seed):
    with pytest.raises(ValueError):
        select_subset_indices(count, subset_size=size, seed=seed)
