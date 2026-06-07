import sys
import types

import torch


transformers_stub = types.ModuleType("transformers")
transformers_stub.AutoProcessor = object
transformers_stub.SeamlessM4Tv2ForSpeechToSpeech = object
sys.modules.setdefault("transformers", transformers_stub)

from polyglot_api import translator


def test_select_torch_device_auto_prefers_cuda_when_available(monkeypatch):
    monkeypatch.setenv("POLYGLOT_DEVICE", "auto")
    monkeypatch.setattr(translator.torch.cuda, "is_available", lambda: True)

    assert translator.select_torch_device() == "cuda"


def test_select_torch_device_falls_back_when_cuda_requested_but_missing(monkeypatch):
    monkeypatch.setenv("POLYGLOT_DEVICE", "cuda")
    monkeypatch.setattr(translator.torch.cuda, "is_available", lambda: False)

    assert translator.select_torch_device() == "cpu"


def test_select_torch_dtype_auto_uses_float16_on_cuda(monkeypatch):
    monkeypatch.setenv("POLYGLOT_TORCH_DTYPE", "auto")

    assert translator.select_torch_dtype("cuda") == torch.float16
    assert translator.select_torch_dtype("cpu") is None


def test_select_torch_dtype_accepts_explicit_float32(monkeypatch):
    monkeypatch.setenv("POLYGLOT_TORCH_DTYPE", "float32")

    assert translator.select_torch_dtype("cuda") == torch.float32


def test_deterministic_translator_mode_skips_model_load(monkeypatch):
    monkeypatch.setenv("POLYGLOT_TRANSLATOR_MODE", "deterministic")
    monkeypatch.setenv("POLYGLOT_DEVICE", "cpu")

    loaded = translator.SeamlessTranslator.load(logger=NullLogger())

    assert isinstance(loaded, translator.DeterministicTranslator)
    assert loaded.health()["model_loaded"] is True
    assert loaded.health()["device"] == "cpu"


class NullLogger:
    def info(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None
