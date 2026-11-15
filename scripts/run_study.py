#!/usr/bin/env python3
"""Run one configured retrieval study through the original experiment runner."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> int:
    """Translate the short experiment config to the existing, audited CLI flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    args = parser.parse_args()
    data = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        parser.error("experiment config must be a mapping")
    keys = {
        "base_config",
        "run_id",
        "os",
        "mode",
        "sample_size",
        "pool_size",
        "seed",
        "baseline_top_k",
        "adaptive_top_k",
        "max_rounds",
        "model_id",
        "model_revision",
        "reuse_question_ids_from",
    }
    if unknown := set(data) - keys:
        parser.error(f"unknown experiment config keys: {sorted(unknown)}")
    required = keys - {"model_id", "model_revision", "reuse_question_ids_from"}
    if missing := required - set(data):
        parser.error(f"missing experiment config keys: {sorted(missing)}")
    if data["mode"] not in {"baseline", "adaptive"} or data["os"] not in {"mac", "windows"}:
        parser.error("mode must be baseline/adaptive and os must be mac/windows")
    command = [
        sys.executable,
        "scripts/run_retrieval_study.py",
        "--config",
        str(data["base_config"]),
        "--mode",
        data["mode"],
        "--os",
        data["os"],
    ]
    for key in (
        "run_id",
        "sample_size",
        "pool_size",
        "seed",
        "baseline_top_k",
        "adaptive_top_k",
        "max_rounds",
        "model_id",
        "model_revision",
        "reuse_question_ids_from",
    ):
        if data.get(key) is not None:
            command.extend(["--" + key.replace("_", "-"), str(data[key])])
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
