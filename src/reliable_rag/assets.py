"""Revision-aware model resolution with offline-first, completeness-checked reuse."""

import json
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)
PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"]


def validate_model_snapshot(path: Path) -> None:
    """Check native model/tokenizer assets before considering a snapshot usable."""
    json.loads((path / "config.json").read_text())
    if not any(
        (path / name).is_file() for name in ["tokenizer.json", "tokenizer.model", "vocab.txt"]
    ):
        raise ValueError("Missing tokenizer assets")
    json.loads((path / "tokenizer_config.json").read_text())
    shards = list(path.glob("*.safetensors"))
    index = path / "model.safetensors.index.json"
    if index.exists():
        files = set(json.loads(index.read_text())["weight_map"].values())
        if any(Path(name).name != name for name in files):
            raise ValueError("Invalid weight shard path")
        shards = [path / name for name in files]
    if not shards or any(not shard.is_file() or shard.stat().st_size == 0 for shard in shards):
        raise ValueError("Missing or empty model weights")
    modules = path / "modules.json"
    if modules.exists():
        for module in json.loads(modules.read_text()):
            relative = Path(module.get("path", ""))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Invalid Sentence Transformer module path")
            # Normalize has no parameters or on-disk configuration.
            if module.get("type", "").endswith(".Normalize"):
                continue
            if not (path / relative).is_dir():
                raise ValueError("Missing Sentence Transformer module")
            if module.get("type", "").endswith("Pooling"):
                json.loads((path / relative / "config.json").read_text())


def resolve_model_path(
    model_id: str, *, revision: str | None, cache_root: Path, allow_download: bool = False
) -> Path:
    local = Path(model_id).expanduser()
    if local.is_dir():
        validate_model_snapshot(local)
        LOGGER.info("Using local model: %s", local)
        return local.resolve()
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    kwargs = dict(
        repo_id=model_id,
        revision=revision,
        cache_dir=cache_root / "huggingface" / "hub",
        allow_patterns=PATTERNS,
    )
    try:
        path = Path(snapshot_download(**kwargs, local_files_only=True))
        validate_model_snapshot(path)
        LOGGER.info("Using cached model: %s @ %s", model_id, revision or path.name)
        return path
    except (LocalEntryNotFoundError, OSError, ValueError, KeyError):
        if not allow_download:
            raise ValueError(
                f"Model {model_id!r} at {revision!r} is unavailable or incomplete. "
                "Run scripts/prepare_models.py, or explicitly use allow_model_download=True."
            ) from None
    LOGGER.info("Downloading missing model assets: %s @ %s", model_id, revision)
    path = Path(snapshot_download(**kwargs, local_files_only=False))
    validate_model_snapshot(path)
    LOGGER.info("Downloaded and verified model: %s", model_id)
    return path
