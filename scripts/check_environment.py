#!/usr/bin/env python3
"""Report the active Python/platform and optional ML runtime availability."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import logging
import platform
import sys

LOGGER = logging.getLogger("reliable_rag.environment")
CORE_PACKAGES = (
    "pydantic",
    "PyYAML",
    "numpy",
    "pandas",
    "datasets",
    "sentence-transformers",
    "scikit-learn",
    "tqdm",
    "pytest",
    "ruff",
)


def main() -> int:
    """Print an environment summary; missing optional packages are informational."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    system = platform.system()
    machine = platform.machine()
    if system == "Darwin" and machine.lower() in {"arm64", "aarch64"}:
        platform_kind = "Apple Silicon (native ARM)"
    elif system == "Linux":
        platform_kind = "Linux"
    else:
        platform_kind = f"{system} ({machine})"

    LOGGER.info("Python: %s", sys.version.split()[0])
    LOGGER.info("Operating system: %s", system)
    LOGGER.info("Architecture: %s", machine)
    LOGGER.info("Platform class: %s", platform_kind)
    for package in CORE_PACKAGES:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "not installed"
        LOGGER.info("Package %-24s %s", package + ":", version)

    mlx_available = importlib.util.find_spec("mlx") is not None
    LOGGER.info("Optional MLX available: %s", "yes" if mlx_available else "no")
    torch_available = importlib.util.find_spec("torch") is not None
    LOGGER.info("Optional PyTorch available: %s", "yes" if torch_available else "no")
    if torch_available:
        try:
            import torch

            LOGGER.info("CUDA available: %s", "yes" if torch.cuda.is_available() else "no")
        except Exception as exc:  # optional runtime can be installed but not loadable on this host
            LOGGER.info("CUDA available: unknown (PyTorch import failed: %s)", exc)
    else:
        LOGGER.info("CUDA available: not checked (PyTorch is not installed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
