#!/usr/bin/env python3
"""Load and validate one project YAML configuration."""

from __future__ import annotations

import argparse
import logging

from reliable_rag.config import load_settings


def main() -> int:
    """Validate the selected configuration and print its key run labels."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="configs/base.yaml")
    args = parser.parse_args()
    settings = load_settings(args.config)
    logging.getLogger("reliable_rag.config").info(
        "Valid configuration: %s (seed=%s, variant=%s, condition=%s)",
        args.config,
        settings.project_seed,
        settings.experiment.system_variant.value,
        settings.experiment.evidence_condition.value,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
