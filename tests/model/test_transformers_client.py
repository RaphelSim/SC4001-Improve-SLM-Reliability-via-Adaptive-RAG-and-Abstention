"""Transformers adapter behaviour tests."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from reliable_rag import runtime
from reliable_rag.config import load_settings
from reliable_rag.model.slm import TransformersClient

ROOT = Path(__file__).parents[2]


class FakeInputs(dict):
    """Mimics a tokenizer BatchEncoding: a dict that can be moved to a device."""

    def to(self, device):
        self["device"] = device
        return self


class FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        self.template_kwargs = kwargs
        return "<user>" + messages[0]["content"]

    def __call__(self, text, return_tensors):
        return FakeInputs(input_ids=[[1] * len(text.split())])

    def decode(self, tokens, skip_special_tokens):
        return f"decoded {list(tokens)}"


class FakeModel:
    device = "cuda"

    def generate(self, **kwargs):
        self.kwargs = kwargs
        # Echo the prompt tokens, then append two new tokens.
        return [kwargs["input_ids"][0] + [7, 8]]


def make_client(**slm_overrides):
    settings = load_settings(ROOT / "configs" / "transformers.yaml", environ={})
    for key, value in slm_overrides.items():
        setattr(settings.slm, key, value)
    model, tokenizer = FakeModel(), FakeTokenizer()
    client = TransformersClient(model=model, tokenizer=tokenizer, settings=settings.slm)
    return client, model, tokenizer


def test_generate_decodes_only_new_tokens_with_greedy_decoding() -> None:
    client, model, tokenizer = make_client()

    assert client.generate("two tokens", max_new_tokens=7) == "decoded [7, 8]"
    assert tokenizer.template_kwargs["enable_thinking"] is False
    assert model.kwargs["max_new_tokens"] == 7
    assert model.kwargs["do_sample"] is False


def test_generate_rejects_overlong_input_and_bad_output_limit() -> None:
    client, _, _ = make_client(max_input_tokens=1)
    with pytest.raises(ValueError, match="max_input_tokens"):
        client.generate("two tokens")
    client, _, _ = make_client()
    with pytest.raises(ValueError, match="max_new_tokens"):
        client.generate("x", max_new_tokens=0)


def test_runtime_selects_transformers_client(monkeypatch) -> None:
    settings = load_settings(ROOT / "configs" / "transformers.yaml", environ={})
    fake_client = SimpleNamespace(model_path=None)
    monkeypatch.setattr(
        runtime.TransformersClient, "from_settings", classmethod(lambda cls, *a, **k: fake_client)
    )

    assert runtime.create_model_components(settings).client is fake_client


def test_from_settings_rejects_quantized_and_wrong_runtime() -> None:
    settings = load_settings(ROOT / "configs" / "transformers.yaml", environ={})
    settings.slm.quantization = "4bit"
    with pytest.raises(ValueError, match="full-precision"):
        TransformersClient.from_settings(settings)
    settings = load_settings(ROOT / "configs" / "base.yaml", environ={})
    with pytest.raises(ValueError, match="'transformers'"):
        TransformersClient.from_settings(settings)
