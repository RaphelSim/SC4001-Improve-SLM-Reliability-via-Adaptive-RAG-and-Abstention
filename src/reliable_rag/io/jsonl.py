"""Strict JSON Lines helpers for validated Pydantic record exchange."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

ModelT = TypeVar("ModelT", bound=BaseModel)


class JSONLValidationError(ValueError):
    """A JSONL row could not be parsed or validated; includes its source line number."""


def read_jsonl(path: str | Path, model_type: type[ModelT]) -> list[ModelT]:
    """Read UTF-8 JSONL and validate every physical line as one model.

    Blank lines are rejected so line numbers and record counts cannot change silently.
    """
    source = Path(path)
    records: list[ModelT] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise JSONLValidationError(f"{source}:{line_number}: blank lines are not allowed")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise JSONLValidationError(
                    f"{source}:{line_number}: invalid JSON at column {exc.colno}: {exc.msg}"
                ) from exc
            try:
                records.append(model_type.model_validate(value))
            except ValidationError as exc:
                errors = "; ".join(
                    f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
                    for item in exc.errors()
                )
                raise JSONLValidationError(
                    f"{source}:{line_number}: {model_type.__name__} validation failed: {errors}"
                ) from exc
    return records


def write_jsonl(path: str | Path, records: list[BaseModel]) -> None:
    """Write models as deterministic UTF-8 JSONL, creating parent directories."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for index, record in enumerate(records):
            if not isinstance(record, BaseModel):
                raise TypeError(f"record at index {index} is not a Pydantic BaseModel")
            serialized = json.dumps(
                record.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.write(serialized + "\n")
