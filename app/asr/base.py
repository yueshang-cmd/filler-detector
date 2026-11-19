from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

ProgressCallback = Callable[[float], None]


class ASREngine(ABC):
    @abstractmethod
    def transcribe(self, path: str | Path, progress_cb: ProgressCallback | None = None) -> dict:
        raise NotImplementedError