#!/usr/bin/env python3
"""Prepare the configured dataset and its MiniLM index without changing either adapter."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from reliable_rag.config import load_settings


def main() -> int:
    """Dispatch to the original dataset adapter, then build or reuse its index."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--subset-size", type=int, help="Override the configured sample size")
    parser.add_argument("--seed", type=int, help="Override the configured seed")
    args = parser.parse_args()
    data = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        parser.error("data config must be a mapping")
    allowed = {"base_config", "subset_size", "seed", "input_path", "source_repo", "revision"}
    if unknown := set(data) - allowed:
        parser.error(f"unknown data config keys: {sorted(unknown)}")
    base = Path(data["base_config"])
    settings = load_settings(base)
    size = args.subset_size if args.subset_size is not None else data["subset_size"]
    seed = args.seed if args.seed is not None else data["seed"]
    if not isinstance(size, int) or size < 1 or not isinstance(seed, int) or seed < 0:
        parser.error("subset_size must be positive and seed non-negative")
    common = ["--config", str(base)]
    source = data.get("input_path")
    if settings.dataset_key == "hotpotqa":
        prepare = [
            sys.executable,
            "scripts/prepare_hotpotqa.py",
            *common,
            "--subset-size",
            str(size),
            "--seed",
            str(seed),
        ]
        if source:
            prepare.extend(["--input", str(source)])
    elif settings.dataset_key == "musique":
        if size != 100 or seed != 88:
            parser.error("the frozen MuSiQue adapter supports exactly 100 questions and seed 88")
        prepare = [sys.executable, "scripts/prepare_musique.py", *common]
        if source:
            prepare.extend(["--input", str(source)])
        if data.get("source_repo"):
            prepare.extend(["--source-repo", str(data["source_repo"])])
        if data.get("revision"):
            prepare.extend(["--revision", str(data["revision"])])
    else:
        parser.error(f"unsupported dataset: {settings.dataset.dataset_name}")
    subprocess.run(prepare, check=True)
    subprocess.run(
        [
            sys.executable,
            "scripts/build_embedding_index.py",
            *common,
            "--subset-size",
            str(size),
            "--seed",
            str(seed),
        ],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
