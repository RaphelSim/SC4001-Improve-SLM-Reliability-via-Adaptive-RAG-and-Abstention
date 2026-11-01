#!/usr/bin/env python3
"""Run the configured local SLM against synthetic Member 2 sufficiency cases."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from reliable_rag.config import load_settings
from reliable_rag.contracts.enums import SufficiencyStatus
from reliable_rag.contracts.records import EvidenceSet, QuestionRecord
from reliable_rag.model.slm import MLXLMClient
from reliable_rag.model.sufficiency import SLMSufficiencyAssessor

LOGGER = logging.getLogger("reliable_rag.manual_sufficiency")
ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "member2_sufficiency_cases.json"


def main() -> int:
    """Load MLX only after explicit script invocation and run synthetic diagnostics."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow mlx-lm to fetch a remote model ID from the configured model hub.",
    )
    args = parser.parse_args()

    settings = load_settings(args.config)
    client = MLXLMClient.from_settings(
        settings,
        allow_model_download=args.allow_model_download,
    )
    assessor = SLMSufficiencyAssessor(client, settings)
    fixture_data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    for case in fixture_data["cases"]:
        question = QuestionRecord.model_validate(case["question"])
        evidence = EvidenceSet.model_validate(case["evidence"])
        decision = assessor.assess(question, evidence)
        result = {
            "case_id": case["case_id"],
            "question_id": question.question_id,
            "gold_sufficient": evidence.gold_sufficient,
            "predicted_sufficient": decision.status == SufficiencyStatus.SUFFICIENT,
            "sufficiency_status": decision.status.value,
            "missing_information": decision.missing_information,
            "search_query": decision.search_query,
        }
        print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
