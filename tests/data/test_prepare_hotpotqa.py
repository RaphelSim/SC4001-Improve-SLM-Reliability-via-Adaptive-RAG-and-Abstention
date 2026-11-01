"""Offline tests for the HotpotQA preparation command."""

from reliable_rag.contracts.enums import DatasetSplit
from reliable_rag.contracts.records import EvidenceChunk, QuestionRecord
from reliable_rag.data import prepare_split
from reliable_rag.io.jsonl import read_jsonl


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
    manifest = [
        line
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
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
