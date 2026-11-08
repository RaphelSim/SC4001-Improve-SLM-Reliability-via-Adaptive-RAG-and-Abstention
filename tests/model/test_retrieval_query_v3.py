"""V3 exact output schema, repair, and gold-free prompt payload."""

import json

import pytest

from reliable_rag.config import load_settings
from reliable_rag.contracts.enums import DatasetSplit
from reliable_rag.contracts.records import EvidenceChunk, ModelEvidence, QuestionRecord
from reliable_rag.model.retrieval_query_v3 import V3QueryAssessor, parse_v3_response
from reliable_rag.model.structured_output import StructuredOutputValidationError


@pytest.mark.parametrize(
    "raw",
    [
        "{}",
        '{"status":"sufficient"}',
        '{"search_query":""}',
        '{"search_query":"x","missing_information":"y"}',
        '{"search_query":"x","search_query":null}',
    ],
)
def test_exact_single_field_schema_rejects_invalid_output(raw):
    with pytest.raises(StructuredOutputValidationError):
        parse_v3_response(raw)


def test_null_and_nonempty_query_are_valid():
    assert parse_v3_response('{"search_query":null}').search_query is None
    assert (
        parse_v3_response('{"search_query":"Sugar Ray Leonard nationality"}').search_query
        == "Sugar Ray Leonard nationality"
    )


def test_prompt_uses_original_question_all_evidence_and_no_gold():
    class Client:
        def __init__(self):
            self.prompts = []

        def generate(self, prompt, *, max_new_tokens=None):
            self.prompts.append(prompt)
            return json.dumps({"search_query": "Named bridge entity nationality"})

    q = QuestionRecord(
        question_id="q1",
        question="What nationality was X's brother?",
        gold_answer="X",
        split=DatasetSplit.DEVELOPMENT,
        gold_support_ids=["secret-gold"],
        gold_paragraph_ids=["secret-gold"],
        candidate_evidence_ids=["a", "b"],
    )
    evidence = ModelEvidence(
        question_id="q1",
        evidence=[
            EvidenceChunk(
                evidence_id="a", document_id="a", text="X's brother is Named bridge entity."
            ),
            EvidenceChunk(evidence_id="b", document_id="b", text="Another accumulated paragraph."),
        ],
    )
    client = Client()
    assessor = V3QueryAssessor(client, load_settings("configs/retrieval_study_v3.yaml"))
    decision = assessor.assess(q, evidence, previous_search_queries=[q.question])
    assert decision.search_query == "Named bridge entity nationality"
    prompt = client.prompts[0]
    assert "X's brother is Named bridge entity." in prompt
    assert "Another accumulated paragraph." in prompt
    assert q.question in prompt
    assert "Your own knowledge is not evidence" in prompt
    assert "concrete bridge entity" in prompt
    assert "secret-gold" not in prompt
    assert "gold_complete" not in prompt
    assert "gold_evidence_percentage" not in prompt
