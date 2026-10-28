#!/usr/bin/env python3
"""Generate checked-in JSON Schemas from canonical Pydantic contracts."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from reliable_rag.contracts.records import (
    AnswerResult,
    EvidenceChunk,
    EvidenceSet,
    ExperimentRecord,
    QuestionRecord,
    RetrievalRequest,
    RetrievalResult,
    SufficiencyDecision,
)

SCHEMAS = {
    "answer-result": AnswerResult,
    "evidence-chunk": EvidenceChunk,
    "evidence-set": EvidenceSet,
    "experiment-record": ExperimentRecord,
    "question-record": QuestionRecord,
    "retrieval-request": RetrievalRequest,
    "retrieval-result": RetrievalResult,
    "sufficiency-decision": SufficiencyDecision,
}


def main() -> int:
    """Write schemas to the repository's schemas/ directory."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    output_dir = Path("schemas")
    output_dir.mkdir(parents=True, exist_ok=True)
    for slug, model in SCHEMAS.items():
        destination = output_dir / f"{slug}.schema.json"
        schema = model.model_json_schema()
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        serialized = json.dumps(schema, indent=2, sort_keys=True) + "\n"
        json.loads(serialized)
        destination.write_text(serialized, encoding="utf-8")
        logging.getLogger("reliable_rag.schemas").info("Wrote %s", destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
