"""Checked-in JSON Schema snapshots are present and parse as valid JSON."""

import json
from pathlib import Path

SCHEMA_DIR = Path(__file__).parents[2] / "schemas"
EXPECTED = {
    "question-record.schema.json",
    "evidence-chunk.schema.json",
    "evidence-set.schema.json",
    "retrieval-request.schema.json",
    "retrieval-result.schema.json",
    "sufficiency-decision.schema.json",
    "answer-result.schema.json",
    "experiment-record.schema.json",
}


def test_generated_schemas_exist_and_are_valid_json() -> None:
    assert EXPECTED.issubset({path.name for path in SCHEMA_DIR.glob("*.schema.json")})
    for filename in EXPECTED:
        schema = json.loads((SCHEMA_DIR / filename).read_text(encoding="utf-8"))
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert "title" in schema
