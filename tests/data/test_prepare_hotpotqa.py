"""Offline tests for the HotpotQA preparation command."""

import json
import sys

import pytest
from scripts import prepare_hotpotqa
from scripts.prepare_hotpotqa import _load_local_rows

from reliable_rag.contracts.enums import DatasetSplit
from reliable_rag.contracts.records import EvidenceChunk, QuestionRecord
from reliable_rag.data import prepare_split
from reliable_rag.data.splits import select_subset_indices
from reliable_rag.io.jsonl import read_jsonl


def test_local_hotpotqa_input_accepts_jsonl_objects(tmp_path) -> None:
    path = tmp_path / "hotpotqa.jsonl"
    rows = [{"_id": "q1", "question": "Q?"}, {"_id": "q2", "question": "Q2?"}]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    assert _load_local_rows(path) == rows


def test_prepare_split_writes_contract_validated_outputs(tmp_path) -> None:
    rows = [
        {
            "_id": "q1",
            "question": "Where is the manufacturer based?",
            "answer": "Germany",
            "supporting_facts": [["Lost Gravity", 0], ["Mack Rides", 0]],
            "context": [
                ["Lost Gravity", ["Lost Gravity was manufactured by Mack Rides."]],
                ["Mack Rides", ["Mack Rides is headquartered in Germany."]],
            ],
        }
    ]
    questions_path, evidence_path, manifest_path = prepare_split(
        rows,
        source_split="train",
        project_split=DatasetSplit.DEVELOPMENT,
        dataset_revision="fixture-v1",
        output_dir=tmp_path / "processed",
        raw_path=tmp_path / "raw" / "train.json",
    )

    questions = read_jsonl(questions_path, QuestionRecord)
    assert len(questions) == 1
    manifest = [line for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    assert '"project_split":"development"' in manifest[0]
    assert read_jsonl(evidence_path, EvidenceChunk)
    assert evidence_path.read_text(encoding="utf-8").count("\n") == 2


def test_prepare_split_disambiguates_conflicting_global_evidence_ids(tmp_path) -> None:
    rows = [
        {
            "_id": "q1",
            "question": "Question one?",
            "answer": "A",
            "supporting_facts": [["Shared", 0]],
            "context": [["Shared", ["Text one."]]],
        },
        {
            "_id": "q2",
            "question": "Question two?",
            "answer": "B",
            "supporting_facts": [["Shared", 0]],
            "context": [["Shared", ["Text two."]]],
        },
    ]

    questions_path, evidence_path, _ = prepare_split(
        rows,
        source_split="train",
        project_split=DatasetSplit.DEVELOPMENT,
        dataset_revision="fixture-v1",
        output_dir=tmp_path / "processed",
    )

    questions = read_jsonl(questions_path, QuestionRecord)
    evidence = read_jsonl(evidence_path, EvidenceChunk)
    assert len(evidence) == 2
    assert questions[0].gold_support_ids != questions[1].gold_support_ids


def _raw_question(index):
    title = f"Source {index}"
    return {
        "_id": f"q{index}",
        "question": f"What is the value for source {index}?",
        "answer": str(index),
        "supporting_facts": [[title, 0]],
        "context": [[title, [f"The value is {index}."]]],
    }


def test_subset_cli_freezes_exact_size_ids_and_provenance(monkeypatch, tmp_path):
    monkeypatch.delenv("RELIABLE_RAG_PROJECT_SEED", raising=False)
    source = tmp_path / "raw.json"
    source.write_text(json.dumps([_raw_question(i) for i in range(40)]))
    output = tmp_path / "prepared"
    args = [
        "prepare_hotpotqa.py",
        "--input",
        str(source),
        "--subset-size",
        "10",
        "--output-dir",
        str(output),
        "--manifest-dir",
        str(tmp_path / "published"),
    ]
    monkeypatch.setattr(sys, "argv", args)
    assert prepare_hotpotqa.main() == 0
    questions = read_jsonl(output / "questions-development.jsonl", QuestionRecord)
    selection = json.loads((output / "subset-development.json").read_text())
    expected = select_subset_indices(40, subset_size=10, seed=88)
    assert [q.question_id for q in questions] == [f"q{i}" for i in expected]
    assert selection["sampling"]["seed"] == 88
    assert selection["source"]["question_count"] == 40
    assert selection["sampling"]["selected_source_indices"] == expected
    assert selection["sampling"]["selected_question_ids"] == [q.question_id for q in questions]
    assert len(selection["source"]["input_sha256"]) == 64
    published = tmp_path / "published/hotpotqa-subset-10-seed88-development.json"
    assert json.loads(published.read_text()) == selection
    first_outputs = {p.name: p.read_bytes() for p in output.iterdir()}
    assert prepare_hotpotqa.main() == 0
    assert first_outputs == {p.name: p.read_bytes() for p in output.iterdir()}


def test_remote_subset_materializes_only_selected_rows(monkeypatch, tmp_path):
    class ArrowSource:
        _fingerprint = "pinned-source-fingerprint"

        def __init__(self):
            self.accessed = []

        def __len__(self):
            return 1000

        def __getitem__(self, index):
            self.accessed.append(index)
            return _raw_question(index)

    source = ArrowSource()
    monkeypatch.setattr(prepare_hotpotqa, "_load_remote_source", lambda *args, **kwargs: source)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_hotpotqa.py",
            "--subset-size",
            "5",
            "--seed",
            "88",
            "--output-dir",
            str(tmp_path / "prepared"),
        ],
    )
    assert prepare_hotpotqa.main() == 0
    assert source.accessed == select_subset_indices(1000, subset_size=5, seed=88)
    selection = json.loads((tmp_path / "prepared/subset-development.json").read_text())
    assert selection["source"]["dataset_revision"] == "1908d6afbbead072334abe2965f91bd2709910ab"
    assert selection["source"]["fingerprint"] == source._fingerprint


def test_oversized_subset_fails_instead_of_silently_using_small_input(monkeypatch, tmp_path):
    source = tmp_path / "raw.json"
    source.write_text(json.dumps([_raw_question(0), _raw_question(1)]))
    output = tmp_path / "prepared"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_hotpotqa.py",
            "--input",
            str(source),
            "--subset-size",
            "500",
            "--output-dir",
            str(output),
        ],
    )
    with pytest.raises(ValueError, match="source question count"):
        prepare_hotpotqa.main()
    assert not output.exists()


def test_prepared_cache_skips_source_load_and_invalidates_corruption_seed_and_force(
    monkeypatch, tmp_path
):
    calls = []

    def source(*args, **kwargs):
        calls.append(True)
        return [_raw_question(i) for i in range(20)]

    monkeypatch.setattr(prepare_hotpotqa, "_load_remote_source", source)
    output = tmp_path / "prepared"
    args = ["prepare_hotpotqa.py", "--subset-size", "5", "--output-dir", str(output)]
    monkeypatch.setattr(sys, "argv", args)
    assert prepare_hotpotqa.main() == 0
    assert prepare_hotpotqa.main() == 0
    assert len(calls) == 1
    (output / "evidence-development.jsonl").write_text("corrupted")
    assert prepare_hotpotqa.main() == 0
    assert len(calls) == 2
    monkeypatch.setattr(sys, "argv", args + ["--seed", "89"])
    assert prepare_hotpotqa.main() == 0
    assert len(calls) == 3
    monkeypatch.setattr(sys, "argv", args + ["--seed", "89", "--force"])
    assert prepare_hotpotqa.main() == 0
    assert len(calls) == 4


def test_source_loader_rejects_offline_fallback_to_another_revision(monkeypatch, tmp_path):
    from types import SimpleNamespace

    revision = "a" * 40
    monkeypatch.setattr(
        "datasets.load_dataset",
        lambda *args, **kwargs: SimpleNamespace(
            cache_files=[{"filename": str(tmp_path / ("b" * 40) / "train.arrow")}]
        ),
    )
    with pytest.raises(ValueError, match="pinned revision"):
        prepare_hotpotqa._load_remote_source(
            "hotpotqa/hotpot_qa", "distractor", "train", revision, cache_dir=tmp_path
        )
