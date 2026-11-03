"""Runtime adapter contract tests without importing or loading MLX weights."""

from pathlib import Path
from typing import Any

import pytest

from reliable_rag.config import load_settings
from reliable_rag.model.slm import MLXLMClient

ROOT = Path(__file__).parents[2]


class FakeTokenizer:
    chat_template = None

    def encode(self, prompt: str) -> list[str]:
        return prompt.split()


def test_mlx_adapter_is_injectable_and_uses_configured_limits() -> None:
    settings = load_settings(ROOT / "configs" / "base.yaml", environ={})
    settings.slm.model_id = "fake"
    calls: dict[str, Any] = {}

    def generate_fn(model, tokenizer, **kwargs):
        calls["model"] = model
        calls["tokenizer"] = tokenizer
        calls.update(kwargs)
        return "model output"

    def sampler_factory(*, temp: float):
        return {"temperature": temp}

    client = MLXLMClient(
        model="fake-model",
        tokenizer=FakeTokenizer(),
        settings=settings.slm,
        generate_fn=generate_fn,
        sampler_factory=sampler_factory,
    )
    assert client.generate("two tokens", max_new_tokens=7) == "model output"
    assert calls["max_tokens"] == 7
    assert calls["sampler"] == {"temperature": 0.0}
    assert calls["verbose"] is False


def test_mlx_adapter_rejects_overlong_input_and_zero_output_limit() -> None:
    settings = load_settings(ROOT / "configs" / "base.yaml", environ={})
    settings.slm.model_id = "fake"
    settings.slm.max_input_tokens = 1
    client = MLXLMClient(
        model="fake-model",
        tokenizer=FakeTokenizer(),
        settings=settings.slm,
        generate_fn=lambda *args, **kwargs: "output",
        sampler_factory=lambda **kwargs: kwargs,
    )
    with pytest.raises(ValueError, match="max_input_tokens"):
        client.generate("two tokens")
    with pytest.raises(ValueError, match="max_new_tokens"):
        client.generate("one", max_new_tokens=0)


def test_remote_model_load_requires_explicit_opt_in() -> None:
    settings = load_settings(ROOT / "configs" / "base.yaml", environ={})
    settings.slm.runtime = "mlx"
    settings.slm.model_id = "mlx-community/example-model"
    with pytest.raises(ValueError, match="allow_model_download=True"):
        MLXLMClient.from_settings(settings)


@pytest.mark.parametrize("canonical_template", [False, True])
def test_qwen3_legacy_and_canonical_templates_disable_thinking(canonical_template) -> None:
    settings = load_settings(ROOT / "configs/base.yaml", environ={})
    settings.slm.model_id = "/local/model-without-a-qwen-name"
    calls = {}

    class ChatTokenizer(FakeTokenizer):
        chat_template = "fake-template"

        def apply_chat_template(self, messages, **kwargs):
            calls["template_kwargs"] = kwargs
            prompt = "<|im_start|>assistant\n"
            return prompt + "<think>\n\n</think>\n\n" if canonical_template else prompt

    def generate_fn(*args, **kwargs):
        calls["prompt"] = kwargs["prompt"]
        return '{"ok":true}'

    client = MLXLMClient(
        model="fake",
        tokenizer=ChatTokenizer(),
        settings=settings.slm,
        generate_fn=generate_fn,
        sampler_factory=lambda **kwargs: kwargs,
        model_type="qwen3",
    )
    assert client.generate("Return JSON") == '{"ok":true}'
    assert calls["template_kwargs"]["enable_thinking"] is False
    assert calls["prompt"].count("<think>") == 1
    assert calls["prompt"].endswith("<think>\n\n</think>\n\n")
