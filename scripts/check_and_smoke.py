#!/usr/bin/env python3
"""Check a Mac or Windows environment with one temporary HotpotQA inference case."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from reliable_rag.config import load_settings


def main() -> int:
    """Verify imports and device, run one case, then discard its artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--os", choices=("mac", "windows"), required=True)
    args = parser.parse_args()
    host = platform.system()
    if (args.os == "mac" and (host != "Darwin" or platform.machine() != "arm64")) or (
        args.os == "windows" and host != "Windows"
    ):
        parser.error(f"--os {args.os} does not match this {host} host")
    if sys.version_info[:2] != (3, 11):
        parser.error("Python 3.11 is required")
    required = [
        "pydantic",
        "yaml",
        "numpy",
        "pandas",
        "datasets",
        "sentence_transformers",
        "sklearn",
        "huggingface_hub",
        "transformers",
    ]
    required += ["torch"] if args.os == "windows" else ["mlx", "mlx_lm"]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        parser.error(f"missing packages: {', '.join(missing)}; install the README dependencies")
    if args.os == "mac":
        import mlx.core as mx

        mx.eval(mx.array([1]) + 1)
        print("MLX Metal operation: OK")
    else:
        import torch

        print(f"PyTorch device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Use the same default HotpotQA source, MiniLM, V3 prompts and Qwen family as
    # the main study. The temporary config only changes the Windows inference backend.
    settings = load_settings("configs/retrieval_study_v3.yaml")
    settings.dataset.subset_size = 1
    if args.os == "windows":
        settings.slm.runtime = "transformers"
        settings.slm.model_id = "Qwen/Qwen3-4B"
        settings.slm.revision = None
        settings.slm.quantization = None
    with tempfile.TemporaryDirectory(prefix="sc4001-smoke-") as temporary:
        settings.data_root = Path(temporary) / "data"
        settings.output_root = Path(temporary) / "outputs"
        run_id = f"smoke_{args.os}_n1"
        index = settings.index_path
        index_metadata = index.with_suffix(index.suffix + ".json")
        existing_index = index.exists() or index_metadata.exists()
        config = Path(temporary) / "smoke.yaml"
        config.write_text(yaml.safe_dump(settings.model_dump(mode="json")), encoding="utf-8")
        commands = [
            [sys.executable, "scripts/prepare_hotpotqa.py", "--config", str(config)],
            [sys.executable, "scripts/prepare_models.py", "--config", str(config)],
            [sys.executable, "scripts/build_embedding_index.py", "--config", str(config)],
            [
                sys.executable,
                "scripts/run_retrieval_study.py",
                "--config",
                str(config),
                "--mode",
                "adaptive",
                "--os",
                args.os,
                "--sample-size",
                "1",
                "--pool-size",
                "1",
                "--seed",
                str(settings.project_seed),
                "--adaptive-top-k",
                "2",
                "--max-rounds",
                "5",
                "--run-id",
                run_id,
            ],
        ]
        try:
            for command in commands:
                print("Running:", " ".join(command), flush=True)
                subprocess.run(command, check=True)
            folder = settings.output_root / run_id
            summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            if summary["attempted"] != 1 or summary["valid"] != 1 or summary["errors"]:
                raise RuntimeError("Smoke inference did not produce one valid result")
            print("Smoke test passed: one valid inference case; temporary output removed")
        finally:
            if not existing_index:
                index.unlink(missing_ok=True)
                index_metadata.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
