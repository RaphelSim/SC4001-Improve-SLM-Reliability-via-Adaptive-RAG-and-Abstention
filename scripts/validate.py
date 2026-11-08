#!/usr/bin/env python3
"""Validate configuration/JSONL or regenerate schemas from canonical contracts."""

import argparse
from pathlib import Path

from reliable_rag.config import load_settings
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
from reliable_rag.contracts.retrieval_study import RetrievalStudyRecord
from reliable_rag.io.artifacts import write_json
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
    "retrieval-study-record": RetrievalStudyRecord,
    "sufficiency-decision": SufficiencyDecision,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    config = sub.add_parser("config", help="Validate a shared YAML configuration")
    config.add_argument("--config", default="configs/base.yaml")
    jsonl = sub.add_parser("jsonl", help="Validate canonical contract rows")
    jsonl.add_argument("path", type=Path)
    jsonl.add_argument("--schema", choices=sorted(SCHEMAS), required=True)
    schemas = sub.add_parser("schemas", help="Regenerate checked-in JSON Schemas")
    schemas.add_argument("--output-dir", type=Path, default=Path("schemas"))
    args = parser.parse_args()
    if args.operation == "config":
        settings = load_settings(args.config)
        print(f"Valid configuration: {args.config}; seed={settings.project_seed}")
    elif args.operation == "jsonl":
        rows = read_jsonl(args.path, SCHEMAS[args.schema])
        print(f"Valid: {len(rows)} rows conform to {args.schema}")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for slug, model in SCHEMAS.items():
            if slug == "adaptive-round-record":
                continue
            schema = model.model_json_schema()
            schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
            write_json(args.output_dir / f"{slug}.schema.json", schema)
        print(f"Schemas written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
