from types import SimpleNamespace

import pytest

from app.asr import faster_engine
from app.asr.faster_engine import FasterWhisperEngine


@pytest.fixture(autouse=True)
def clean_model_cache():
    faster_engine._model_cache.clear()
    yield
    faster_engine._model_cache.clear()


def test_device_compute_type_branches():
    assert FasterWhisperEngine(device="cpu").compute_type == "int8"
    assert FasterWhisperEngine(device="cuda").compute_type == "float16"
    with pytest.raises(ValueError):
        FasterWhisperEngine(device="rocm")


def test_model_cached_per_key(monkeypatch):
    created = []

    class StubModel:
        def __init__(self, name, device=None, compute_type=None):
            created.append((name, device, compute_type))

    import faster_whisper

    monkeypatch.setattr(faster_whisper, "WhisperModel", StubModel)
    a = faster_engine._get_model("small", "cpu", "int8")
    b = faster_engine._get_model("small", "cpu", "int8")
    c = faster_engine._get_model("small", "cuda", "float16")
    assert a is b
    assert a is not c
    assert created == [("small", "cpu", "int8"), ("small", "cuda", "float16")]


def _stub_model(language="zh"):
    words = [
        SimpleNamespace(word="嗯", start=0.0, end=0.3, probability=0.9),
        SimpleNamespace(word="大家好", start=0.3, end=1.0, probability=0.95),
    ]
    seg = SimpleNamespace(start=0.0, end=1.0, text="嗯,大家好", words=words)
    seg2 = SimpleNamespace(start=1.0, end=2.0, text="然后呢", words=None)
    info = SimpleNamespace(language=language, duration=2.0)

    class StubModel:
        def transcribe(self, path, **kwargs):
            self.kwargs = kwargs
            return iter([seg, seg2]), info

    return StubModel()


def test_transcribe_result_shape_and_progress(monkeypatch):
    stub = _stub_model()
    monkeypatch.setattr(faster_engine, "_get_model", lambda *a: stub)
    engine = FasterWhisperEngine(device="cpu", model="small", language="zh")
    progress = []
    result = engine.transcribe("dummy.wav", progress_cb=progress.append)
    assert stub.kwargs["language"] == "zh"
    assert stub.kwargs["vad_filter"] is True
    assert stub.kwargs["word_timestamps"] is True
    assert result["language"] == "zh"
    assert result["duration"] == 2.0
    assert progress == [1.0, 2.0]
    assert result["segments"][0]["words"] == [
        {"word": "嗯", "start": 0.0, "end": 0.3, "probability": 0.9},
        {"word": "大家好", "start": 0.3, "end": 1.0, "probability": 0.95},
    ]
    assert result["segments"][1]["words"] == []


def test_transcribe_language_auto(monkeypatch):
    stub = _stub_model(language="en")
    monkeypatch.setattr(faster_engine, "_get_model", lambda *a: stub)
    engine = FasterWhisperEngine(language="auto")
    result = engine.transcribe("dummy.wav")
    assert stub.kwargs["language"] is None
    assert result["language"] == "en"