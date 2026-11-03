#!/usr/bin/env python3
"""Ensure the configured embedding and generation models are cached and complete."""

import argparse
import logging

from reliable_rag.assets import resolve_model_path
from reliable_rag.config import load_settings


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--offline", action="store_true", help="Verify cached/local models without downloading"
    )
    args = parser.parse_args()
    settings = load_settings(args.config)
    for model_id, revision in [
        (settings.retrieval.embedding_model, settings.retrieval.embedding_revision),
        (settings.slm.model_id, settings.slm.revision),
    ]:
        resolve_model_path(
            model_id,
            revision=revision,
            cache_root=settings.cache_root,
            allow_download=not args.offline,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
