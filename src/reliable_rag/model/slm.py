"""Backend-neutral SLM client contract and optional Apple Silicon MLX-LM adapter."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from reliable_rag.config.settings import Settings, SLMSettings

LOGGER = logging.getLogger(__name__)


@runtime_checkable
class SLMClient(Protocol):
    """Minimal text generation API used by sufficiency and answer components."""

    def generate(self, prompt: str, *, max_new_tokens: int | None = None) -> str:
        """Generate text from a prompt without retaining task-specific conversation state."""
        ...


class MLXLMClient:
    """Small lazy-loaded adapter for the optional `mlx-lm` Python API."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        settings: SLMSettings,
        generate_fn: Callable[..., str],
        sampler_factory: Callable[..., Any],
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._settings = settings
        self._generate_fn = generate_fn
        self._sampler_factory = sampler_factory

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        allow_model_download: bool = False,
    ) -> MLXLMClient:
        """Load a configured MLX model; remote loading requires an explicit opt-in.

        Unit tests should inject a fake `SLMClient` and must not call this method.
        """
        slm_settings = settings.slm
        if slm_settings.runtime.lower() not in {"auto", "mlx"}:
            raise ValueError(
                "MLXLMClient requires slm.runtime to be 'mlx' or 'auto'; "
                "use another SLMClient implementation for this runtime"
            )
        if not slm_settings.model_id.strip() or slm_settings.model_id.strip().lower() == "tbd":
            raise ValueError(
                "Set slm.model_id to a local MLX model directory or model repository ID"
            )
        model_location = Path(slm_settings.model_id).expanduser()
        if not model_location.exists() and not allow_model_download:
            raise ValueError(
                "The configured model is not a local path. Pass allow_model_download=True "
                "only when you intend to fetch the configured model."
            )
        try:
            import mlx.core as mx
            from mlx_lm import generate, load
            from mlx_lm.sample_utils import make_sampler
        except ImportError as exc:
            raise RuntimeError(
                "The optional MLX runtime is unavailable. On Apple Silicon, install `.[dev,mac]`."
            ) from exc

        mx.random.seed(settings.project_seed)
        model, tokenizer = load(slm_settings.model_id)
        LOGGER.info(
            "Loaded MLX model model_id=%s quantization=%s project_seed=%s",
            slm_settings.model_id,
            slm_settings.quantization,
            settings.project_seed,
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            settings=slm_settings,
            generate_fn=generate,
            sampler_factory=make_sampler,
        )

    def generate(self, prompt: str, *, max_new_tokens: int | None = None) -> str:
        """Generate within configured token limits, without logging prompt contents."""
        formatted_prompt = prompt
        if getattr(self._tokenizer, "chat_template", None):
            formatted_prompt = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        prompt_tokens = self._tokenizer.encode(formatted_prompt)
        if len(prompt_tokens) > self._settings.max_input_tokens:
            raise ValueError(
                f"Prompt has {len(prompt_tokens)} tokens, exceeding configured "
                f"max_input_tokens={self._settings.max_input_tokens}; evidence was not truncated."
            )
        token_limit = self._settings.max_new_tokens if max_new_tokens is None else max_new_tokens
        if token_limit < 1:
            raise ValueError("max_new_tokens must be at least 1")
        temperature = self._settings.temperature if self._settings.do_sample else 0.0
        sampler = self._sampler_factory(temp=temperature)
        output = self._generate_fn(
            self._model,
            self._tokenizer,
            prompt=formatted_prompt,
            max_tokens=token_limit,
            sampler=sampler,
            verbose=False,
        )
        if output is None:
            return ""
        if not isinstance(output, str):
            raise RuntimeError(f"MLX-LM returned {type(output).__name__}, expected text")
        return output
