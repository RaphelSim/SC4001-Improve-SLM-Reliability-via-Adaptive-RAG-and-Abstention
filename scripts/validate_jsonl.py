#!/usr/bin/env python3
"""Validate JSONL rows using one of the canonical project record schemas."""

from __future__ import annotations

import argparse
import logging

from reliable_rag.contracts.records import (
    AdaptiveRoundRecord,
    AnswerResult,
    EvidenceChunk,
    EvidenceSet,
    ExperimentRecord,
    QuestionRecord,
    RetrievalRequest,
    RetrievalResult,
    SufficiencyDecision,
)
from reliable_rag.io.jsonl import read_jsonl

SCHEMAS = {
    "adaptive-round-record": AdaptiveRoundRecord,
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
    """Validate the complete JSONL file without skipping malformed rows."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", choices=sorted(SCHEMAS), required=True)
    parser.add_argument("path")
    args = parser.parse_args()
    records = read_jsonl(args.path, SCHEMAS[args.schema])
    logging.getLogger("reliable_rag.jsonl").info(
        "Valid: %s rows conform to %s", len(records), args.schema
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
