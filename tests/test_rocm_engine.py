import sys
import wave
from types import SimpleNamespace

import pytest

from app.asr.rocm_engine import RocmWhisperEngine, _chunks_to_segments


@pytest.fixture()
def wav_1s(tmp_path):
    path = tmp_path / "tone.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 16000)
    return path


def test_missing_torch_raises_clear_error(monkeypatch, wav_1s):
    monkeypatch.setitem(sys.modules, "torch", None)
    engine = RocmWhisperEngine()
    with pytest.raises(RuntimeError, match="ROCm 引擎不可用"):
        engine.transcribe(wav_1s)


def test_non_rocm_torch_raises_clear_error(monkeypatch, wav_1s):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(version=SimpleNamespace(hip=None), cuda=SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=lambda *a, **k: None))
    with pytest.raises(RuntimeError, match="非 ROCm"):
        RocmWhisperEngine().transcribe(wav_1s)


def test_no_gpu_raises_clear_error(monkeypatch, wav_1s):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(version=SimpleNamespace(hip="6.4.0"), cuda=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=lambda *a, **k: None))
    with pytest.raises(RuntimeError, match="未检测到可用 ROCm GPU"):
        RocmWhisperEngine().transcribe(wav_1s)


def test_transcribe_with_fake_pipeline(monkeypatch, wav_1s):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(version=SimpleNamespace(hip="6.4.0"), cuda=SimpleNamespace(is_available=lambda: True)),
    )
    captured = {}

    def fake_pipeline(task, model=None, device=None, **kwargs):
        captured.update(task=task, model=model, device=device)

        def pipe(audio, return_timestamps=None, generate_kwargs=None):
            captured.update(
                return_timestamps=return_timestamps,
                generate_kwargs=generate_kwargs,
                sampling_rate=audio["sampling_rate"],
            )
            return {
                "text": "大家好。项目进度",
                "chunks": [
                    {"text": "大家好", "timestamp": (0.0, 0.8)},
                    {"text": "。", "timestamp": (0.8, 0.9)},
                    {"text": "项目进度", "timestamp": (1.5, 2.5)},
                ],
            }

        return pipe

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=fake_pipeline))
    progress = []
    result = RocmWhisperEngine(model="small", language="zh").transcribe(
        wav_1s, progress_cb=progress.append
    )
    assert captured["task"] == "automatic-speech-recognition"
    assert captured["model"] == "openai/whisper-small"
    assert captured["device"] == "cuda:0"
    assert captured["return_timestamps"] == "word"
    assert captured["generate_kwargs"]["language"] == "zh"
    assert captured["sampling_rate"] == 16000
    assert result["language"] == "zh"
    assert result["duration"] == pytest.approx(1.0)
    assert len(result["segments"]) == 2
    seg = result["segments"][0]
    assert seg["text"] == "大家好。"
    assert seg["words"][0] == {
        "word": "大家好",
        "start": 0.0,
        "end": 0.8,
        "probability": 1.0,
    }
    assert result["segments"][1]["start"] == 1.5
    assert progress == [0.8, 0.9, 2.5]


def test_chunks_to_segments_handles_none_timestamps():
    chunks = [
        {"text": "词一", "timestamp": (0.0, 0.5)},
        {"text": "词二", "timestamp": (0.5, None)},
        {"text": "无时间", "timestamp": (None, None)},
    ]
    segments = _chunks_to_segments(chunks, None)
    assert len(segments) == 1
    assert segments[0]["words"][1]["end"] == 0.5
    assert segments[0]["end"] == 0.5


def test_invalid_model_rejected():
    with pytest.raises(ValueError):
        RocmWhisperEngine(model="tiny")