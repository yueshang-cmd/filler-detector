import time
from pathlib import Path

from app import audio
from app.asr.base import ASREngine


class FakeEngine(ASREngine):
    def __init__(self, delay: float = 0.0, segment_len: float = 2.0):
        self.delay = delay
        self.segment_len = segment_len

    def transcribe(self, path: str | Path, progress_cb=None) -> dict:
        duration = audio.probe_duration(path)
        segments = []
        t = 0.0
        while t < duration:
            end = min(t + self.segment_len, duration)
            words = [
                {"word": "嗯", "start": t, "end": t + 0.3, "probability": 0.9},
                {"word": "大家好", "start": t + 0.3, "end": end, "probability": 0.95},
            ]
            segments.append(
                {"start": t, "end": end, "text": "嗯,大家好", "words": words}
            )
            if progress_cb is not None:
                progress_cb(end)
            if self.delay:
                time.sleep(self.delay)
            t = end
        return {"segments": segments, "language": "zh", "duration": duration}