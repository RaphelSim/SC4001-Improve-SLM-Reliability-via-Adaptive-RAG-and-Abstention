"""Strict JSON parsing, prompt rendering, and bounded formatting-retry helpers."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

ModelT = TypeVar("ModelT", bound=BaseModel)
_PLACEHOLDER = re.compile(r"\{\{([a-zA-Z0-9_]+)\}\}")


class StructuredOutputValidationError(ValueError):
    """The raw response is not a strict JSON object matching its output contract."""


def load_prompt_template(path: str | Path) -> str:
    """Read one prompt template from a UTF-8 text file and reject an empty template."""
    prompt_path = Path(path)
    try:
        template = prompt_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FileNotFoundError(f"Unable to read prompt template {prompt_path}: {exc}") from exc
    if not template.strip():
        raise ValueError(f"Prompt template {prompt_path} is empty")
    return template


def render_prompt(template: str, values: Mapping[str, str]) -> str:
    """Replace declared `{{name}}` placeholders and fail if any remain unresolved."""
    rendered = template
    for name, value in values.items():
        rendered = rendered.replace("{{" + name + "}}", value)
    unresolved = sorted(set(_PLACEHOLDER.findall(rendered)))
    if unresolved:
        raise ValueError(f"Prompt template has unresolved placeholders: {', '.join(unresolved)}")
    return rendered


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """JSON object hook that rejects duplicate keys rather than taking the last value."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise StructuredOutputValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> None:
    """Reject JavaScript-style NaN and Infinity constants in strict JSON."""
    raise StructuredOutputValidationError(f"non-standard JSON value is not allowed: {value}")


def parse_json_model(
    raw_output: str,
    model_type: type[ModelT],
    *,
    required_keys: set[str],
) -> ModelT:
    """Parse a strict JSON object with exact keys and validate it as a Pydantic model."""
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise StructuredOutputValidationError("model output must be a non-empty string")
    try:
        payload = json.loads(
            raw_output,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonstandard_constant,
        )
    except json.JSONDecodeError as exc:
        raise StructuredOutputValidationError(
            f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise StructuredOutputValidationError("model output must be one JSON object")
    actual_keys = set(payload)
    if actual_keys != required_keys:
        missing = sorted(required_keys - actual_keys)
        extra = sorted(actual_keys - required_keys)
        details = []
        if missing:
            details.append(f"missing keys: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected keys: {', '.join(extra)}")
        raise StructuredOutputValidationError("; ".join(details))
    try:
        return model_type.model_validate(payload)
    except ValidationError as exc:
        errors = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors(include_input=False, include_url=False)
        )
        raise StructuredOutputValidationError(errors) from exc


def build_retry_prompt(
    template: str,
    *,
    task: str,
    schema: dict[str, object],
    invalid_output: str,
    validation_error: str,
    original_prompt: str | None = None,
) -> str:
    """Render a repair-only prompt with all untrusted output JSON-encoded as data."""
    return render_prompt(
        template,
        {
            "task_name_json": json.dumps(task),
            "schema_json": json.dumps(schema, ensure_ascii=False, sort_keys=True),
            "invalid_output_json": json.dumps(invalid_output, ensure_ascii=False),
            "validation_error_json": json.dumps(validation_error, ensure_ascii=False),
            "original_prompt_json": json.dumps(original_prompt, ensure_ascii=False),
        },
    )
