import threading
from pathlib import Path

from .base import ASREngine, ProgressCallback

_model_cache: dict[tuple, object] = {}
_model_cache_lock = threading.Lock()


def _get_model(model_name: str, device: str, compute_type: str):
    key = (model_name, device, compute_type)
    with _model_cache_lock:
        if key not in _model_cache:
            from faster_whisper import WhisperModel

            _model_cache[key] = WhisperModel(
                model_name, device=device, compute_type=compute_type
            )
        return _model_cache[key]


class FasterWhisperEngine(ASREngine):
    def __init__(self, device: str = "cpu", model: str = "small", language: str = "zh"):
        if device not in ("cpu", "cuda"):
            raise ValueError(f"FasterWhisperEngine 不支持 device={device}")
        self.device = device
        self.model_name = model
        self.language = language
        self.compute_type = "int8" if device == "cpu" else "float16"

    def transcribe(self, path: str | Path, progress_cb: ProgressCallback | None = None) -> dict:
        engine_name = f"faster-whisper({self.device})"
        try:
            model = _get_model(self.model_name, self.device, self.compute_type)
        except Exception as exc:
            raise RuntimeError(f"{engine_name} 加载模型失败：{exc}") from exc
        language = None if self.language == "auto" else self.language
        try:
            segments_iter, info = model.transcribe(
                str(path),
                language=language,
                vad_filter=True,
                word_timestamps=True,
            )
            segments = []
            for seg in segments_iter:
                words = [
                    {
                        "word": w.word,
                        "start": float(w.start),
                        "end": float(w.end),
                        "probability": float(w.probability),
                    }
                    for w in (seg.words or [])
                ]
                segments.append(
                    {
                        "start": float(seg.start),
                        "end": float(seg.end),
                        "text": seg.text,
                        "words": words,
                    }
                )
                if progress_cb is not None:
                    progress_cb(float(seg.end))
        except Exception as exc:
            raise RuntimeError(f"{engine_name} 转写失败：{exc}") from exc
        return {
            "segments": segments,
            "language": info.language,
            "duration": float(info.duration),
        }