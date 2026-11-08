"""Backend-neutral SLM client contract and optional Apple Silicon MLX-LM and Transformers adapters."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from reliable_rag.assets import resolve_model_path
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
        model_path: Path | None = None,
        model_type: str | None = None,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._settings = settings
        self._generate_fn = generate_fn
        self._sampler_factory = sampler_factory
        self.model_path = model_path
        self._is_qwen3 = model_type == "qwen3" or "qwen3" in settings.model_id.lower()

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
        model_location = resolve_model_path(
            slm_settings.model_id,
            revision=slm_settings.revision,
            cache_root=settings.cache_root,
            allow_download=allow_model_download,
        )
        model_config = json.loads((model_location / "config.json").read_text(encoding="utf-8"))
        if slm_settings.quantization == "4bit":
            quantization = model_config.get(
                "quantization", model_config.get("quantization_config", {})
            )
            if quantization.get("bits") != 4:
                raise ValueError("Configured 4bit model does not contain 4-bit quantized weights")
        try:
            import mlx.core as mx
            from mlx_lm import generate, load
            from mlx_lm.sample_utils import make_sampler
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError(
                "MLX could not access its runtime. Use native ARM Python on Apple Silicon, "
                "install `.[dev,mac]` in the active environment, and run from a terminal "
                "with Metal GPU access."
            ) from exc

        mx.random.seed(settings.project_seed)
        model, tokenizer = load(model_location)
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
            model_path=model_location,
            model_type=model_config.get("model_type"),
        )

    def generate(self, prompt: str, *, max_new_tokens: int | None = None) -> str:
        """Generate within configured token limits, without logging prompt contents."""
        formatted_prompt = prompt
        if getattr(self._tokenizer, "chat_template", None):
            formatted_prompt = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self._settings.enable_thinking,
            )
            # Some official MLX conversions retain a legacy template that ignores
            # enable_thinking. Match Qwen3's canonical disabled-thinking suffix.
            if (
                self._is_qwen3
                and not self._settings.enable_thinking
                and not formatted_prompt.rstrip().endswith("</think>")
            ):
                formatted_prompt += "<think>\n\n</think>\n\n"
        elif self._is_qwen3:
            raise ValueError("Qwen3 requires its chat template; check the tokenizer assets")
        prompt_tokens = self._tokenizer.encode(formatted_prompt)
        if len(prompt_tokens) > self._settings.max_input_tokens:
            raise ValueError(
                f"Prompt has {len(prompt_tokens)} tokens, exceeding configured "
                f"max_input_tokens={self._settings.max_input_tokens}; evidence was not truncated."
            )
        token_limit = self._settings.max_new_tokens if max_new_tokens is None else max_new_tokens
        if not 1 <= token_limit <= self._settings.max_new_tokens:
            raise ValueError("max_new_tokens must be between 1 and the configured output limit")
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


class TransformersClient:
    """Hugging Face Transformers adapter.

    Mirrors MLXLMClient: same chat template, disabled Qwen3 thinking, greedy decoding and
    token limits, so only inference engine differs between runtimes.
    """

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        settings: SLMSettings,
        inference_context: Callable[[], AbstractContextManager[Any]] = nullcontext,
        model_path: Path | None = None,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._settings = settings
        self._inference_context = inference_context
        self.model_path = model_path

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        allow_model_download: bool = False,
    ) -> TransformersClient:
        """Load a configured Transformers model."""
        slm_settings = settings.slm
        if slm_settings.runtime.lower() != "transformers":
            raise ValueError("TransformersClient requires slm.runtime to be 'transformers'")
        if slm_settings.quantization not in (None, "none"):
            raise ValueError("TransformersClient loads full-precision weights, set quantization")
        model_location = resolve_model_path(
            slm_settings.model_id,
            revision=slm_settings.revision,
            cache_root=settings.cache_root,
            allow_download=allow_model_download,
        )
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Transformers runtime needs torch and transformers. Install `.[dev,colab]` "
                "with a CUDA build of torch for GPU inference."
            ) from exc

        torch.manual_seed(settings.project_seed)
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            device = "cuda"
        else:
            dtype, device = torch.float32, "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_location)
        model = AutoModelForCausalLM.from_pretrained(model_location, torch_dtype=dtype)
        model.to(device).eval()
        LOGGER.info(
            "Loaded Transformers model model_id=%s device=%s dtype=%s project_seed=%s",
            slm_settings.model_id,
            device,
            str(dtype).removeprefix("torch."),
            settings.project_seed,
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            settings=slm_settings,
            inference_context=torch.inference_mode,
            model_path=model_location,
        )

    def generate(self, prompt: str, *, max_new_tokens: int | None = None) -> str:
        """Generate within configured token limits, without logging prompt contents."""
        formatted_prompt = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self._settings.enable_thinking,
        )
        inputs = self._tokenizer(formatted_prompt, return_tensors="pt").to(self._model.device)
        prompt_length = len(inputs["input_ids"][0])
        if prompt_length > self._settings.max_input_tokens:
            raise ValueError(
                f"Prompt has {prompt_length} tokens, exceeding configured "
                f"max_input_tokens={self._settings.max_input_tokens}; evidence was not truncated."
            )
        token_limit = self._settings.max_new_tokens if max_new_tokens is None else max_new_tokens
        if not 1 <= token_limit <= self._settings.max_new_tokens:
            raise ValueError("max_new_tokens must be between 1 and the configured output limit")
        with self._inference_context():
            output = self._model.generate(
                **inputs,
                max_new_tokens=token_limit,
                do_sample=self._settings.do_sample,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        # Decode only the newly generated tokens, not the echoed prompt.
        return self._tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)
