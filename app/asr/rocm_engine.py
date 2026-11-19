import wave
from pathlib import Path

import numpy as np

from .base import ASREngine, ProgressCallback

_MODEL_IDS = {
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large-v3": "openai/whisper-large-v3",
}
_SEGMENT_GAP = 0.8
_MAX_SEGMENT_SPAN = 30.0
_SENT_END = "。！？!?"


def _load_wav(path: str | Path) -> dict:
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise RuntimeError("需要 16kHz 单声道 PCM wav")
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return {"raw": samples, "sampling_rate": rate}


def _chunks_to_segments(chunks: list, progress_cb: ProgressCallback | None) -> list:
    segments = []
    buf: list = []

    def flush() -> None:
        if not buf:
            return
        segments.append(
            {
                "start": buf[0]["start"],
                "end": buf[-1]["end"],
                "text": "".join(w["word"] for w in buf),
                "words": list(buf),
            }
        )
        buf.clear()

    for chunk in chunks:
        text = chunk.get("text") or ""
        start, end = chunk.get("timestamp") or (None, None)
        if start is None:
            continue
        if end is None:
            end = start
        word = {
            "word": text,
            "start": float(start),
            "end": float(end),
            "probability": 1.0,
        }
        if buf and (
            word["start"] - buf[-1]["end"] > _SEGMENT_GAP
            or word["start"] - buf[0]["start"] > _MAX_SEGMENT_SPAN
        ):
            flush()
        buf.append(word)
        if any(ch in text for ch in _SENT_END):
            flush()
        if progress_cb is not None:
            progress_cb(word["end"])
    flush()
    return segments


class RocmWhisperEngine(ASREngine):
    def __init__(self, model: str = "small", language: str = "zh"):
        if model not in _MODEL_IDS:
            raise ValueError(f"RocmWhisperEngine 不支持 model={model}")
        self.model_name = model
        self.language = language

    def transcribe(self, path: str | Path, progress_cb: ProgressCallback | None = None) -> dict:
        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                f"ROCm 引擎不可用：缺少依赖 {exc.name or exc}（见 requirements-rocm.txt）"
            ) from exc
        if not getattr(torch.version, "hip", None):
            raise RuntimeError("ROCm 引擎不可用：当前 torch 非 ROCm 构建")
        if not torch.cuda.is_available():
            raise RuntimeError("ROCm 引擎不可用：未检测到可用 ROCm GPU")
        model_id = _MODEL_IDS[self.model_name]
        try:
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model_id,
                device="cuda:0",
                chunk_length_s=30,
                stride_length_s=5,
            )
        except Exception as exc:
            raise RuntimeError(f"ROCm 引擎加载模型 {model_id} 失败：{exc}") from exc
        generate_kwargs = {"task": "transcribe"}
        if self.language != "auto":
            generate_kwargs["language"] = self.language
        try:
            audio = _load_wav(path)
            output = pipe(
                audio, return_timestamps="word", generate_kwargs=generate_kwargs
            )
        except Exception as exc:
            raise RuntimeError(f"ROCm 引擎转写失败：{exc}") from exc
        segments = _chunks_to_segments(output.get("chunks") or [], progress_cb)
        duration = float(len(audio["raw"]) / audio["sampling_rate"])
        return {
            "segments": segments,
            "language": self.language,
            "duration": duration,
        }