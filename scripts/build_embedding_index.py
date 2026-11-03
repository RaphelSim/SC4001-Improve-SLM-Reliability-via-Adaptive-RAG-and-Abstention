#!/usr/bin/env python3
"""Build or reuse the embedding index for the configured prepared dataset."""

import argparse
import logging
from pathlib import Path

from reliable_rag.config import load_settings
from reliable_rag.retrieval.embedder import prepare_embedding_index


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--subset-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--evidence", type=Path, help="Override the derived evidence JSONL")
    parser.add_argument("--index", type=Path, help="Override the derived index location")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--force", action="store_true", help="Rebuild a compatible index too")
    args = parser.parse_args()
    if args.subset_size is not None and args.subset_size < 1:
        parser.error("--subset-size must be positive")
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    settings = load_settings(args.config)
    if args.subset_size is not None:
        settings.dataset.subset_size = args.subset_size
    if args.seed is not None:
        settings.project_seed = args.seed
    prepare_embedding_index(
        settings,
        args.evidence or settings.evidence_path,
        args.index or settings.index_path,
        batch_size=args.batch_size,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
