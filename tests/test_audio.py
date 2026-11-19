import wave
from pathlib import Path

import pytest

from app import audio

SAMPLE_MP4 = Path(__file__).resolve().parent / "assets" / "sample_10s.mp4"


def test_find_ffmpeg():
    exe, source = audio.find_ffmpeg()
    assert Path(exe).exists()
    assert source in ("system", "imageio-ffmpeg")


def test_extract_audio_16k_mono(tmp_path):
    dst = tmp_path / "out.wav"
    audio.extract_audio(SAMPLE_MP4, dst)
    with wave.open(str(dst), "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getnchannels() == 1
        duration = wf.getnframes() / wf.getframerate()
    assert duration == pytest.approx(10.0, abs=0.5)


def test_probe_duration_mp4():
    assert audio.probe_duration(SAMPLE_MP4) == pytest.approx(10.0, abs=0.5)


def test_probe_duration_wav(tmp_path):
    dst = tmp_path / "out.wav"
    audio.extract_audio(SAMPLE_MP4, dst)
    assert audio.probe_duration(dst) == pytest.approx(10.0, abs=0.5)


def test_unsupported_extension(tmp_path):
    with pytest.raises(ValueError):
        audio.check_extension("evil.exe")
    with pytest.raises(ValueError):
        audio.extract_audio("evil.txt", tmp_path / "out.wav")