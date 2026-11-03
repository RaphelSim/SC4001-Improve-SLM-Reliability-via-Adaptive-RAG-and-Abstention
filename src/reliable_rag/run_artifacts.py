"""Stream experiment artifacts and retain provenance and incomplete-run state."""

import importlib.metadata
import json
import logging
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from reliable_rag.io.artifacts import file_sha256, write_json


def run_directory(output_root: Path, run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("run_id must use letters, numbers, dots, underscores or hyphens")
    return output_root / run_id


class RunWriter(logging.Handler):
    """Flush each result/error and persist progress after every attempted question."""

    def __init__(self, settings, run_id: str, *, inputs: list[Path], force: bool = False):
        super().__init__(level=logging.INFO)
        self.directory = run_directory(settings.output_root, run_id)
        resolved = self.directory.resolve()
        prompt_paths = [
            settings.slm.sufficiency_prompt_path,
            settings.slm.final_answer_prompt_path,
            settings.slm.structured_retry_prompt_path,
        ]
        if any(path.resolve().is_relative_to(resolved) for path in inputs + prompt_paths):
            raise ValueError("Run outputs must not overwrite their inputs")
        if self.directory.exists() and not force:
            raise FileExistsError(
                f"Run already exists: {self.directory}; choose a new --run-id or use --force"
            )
        input_hashes = {str(path): file_sha256(path) for path in inputs}
        prompt_hashes = {str(path): file_sha256(path) for path in prompt_paths}
        if self.directory.exists():
            if self.directory.is_symlink():
                raise ValueError("Refusing to replace a symlink run directory")
            shutil.rmtree(self.directory)
        self.directory.mkdir(parents=True)
        self.streams = {
            name: (self.directory / f"{name}.jsonl").open("w", encoding="utf-8")
            for name in ("results", "errors", "events")
        }
        versions = {}
        for package in (
            "reliable-rag",
            "datasets",
            "mlx",
            "mlx-lm",
            "sentence-transformers",
            "transformers",
            "numpy",
        ):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            dirty = bool(
                subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
            )
        except (OSError, subprocess.CalledProcessError):
            commit, dirty = None, None
        self.metadata = {
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "status": "running",
            "scope": "single_round_rag_abstention",
            "config": settings.model_dump(mode="json"),
            "git_commit": commit,
            "git_dirty": dirty,
            "packages": versions,
            "inputs": input_hashes,
            "prompts": prompt_hashes,
            "summary": {
                "planned": 0,
                "attempted": 0,
                "records_written": 0,
                "model_errors": 0,
                "unrecorded_assessment_failures": 0,
                "recorded_answered": 0,
                "recorded_abstained": 0,
                "runtime_errors": 0,
            },
        }
        self.save()

    def append(self, stream: str, row: dict) -> None:
        self.streams[stream].write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        self.streams[stream].flush()

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith("reliable_rag.model."):
            self.append(
                "events",
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "level": record.levelname,
                    "component": record.name,
                    "message": record.getMessage(),
                },
            )

    def save(self) -> None:
        write_json(self.directory / "run_metadata.json", self.metadata)

    def close(self) -> None:
        for handle in getattr(self, "streams", {}).values():
            handle.close()
        super().close()
