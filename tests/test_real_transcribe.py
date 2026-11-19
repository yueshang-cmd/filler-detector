import pytest

from app.asr.faster_engine import FasterWhisperEngine
from tests.conftest import ASSETS_DIR, wait_done

WAV = ASSETS_DIR / "chinese_speech_60s.wav"
MP4 = ASSETS_DIR / "chinese_speech_60s.mp4"


@pytest.mark.real
def test_real_engine_chinese_wav():
    engine = FasterWhisperEngine(device="cpu", model="small", language="zh")
    progress = []
    result = engine.transcribe(WAV, progress_cb=progress.append)
    assert result["language"] == "zh"
    assert result["duration"] == pytest.approx(66.9, abs=1.5)
    text = "".join(s["text"] for s in result["segments"])
    for keyword in ("大家好", "项目进度", "上周", "散会"):
        assert keyword in text, text[:200]
    assert progress and progress == sorted(progress)
    assert progress[-1] == pytest.approx(result["duration"], abs=1.5)
    words = [w for s in result["segments"] for w in s["words"]]
    assert len(words) > 50
    starts = [w["start"] for w in words]
    assert starts == sorted(starts)
    for w in words:
        assert w["end"] >= w["start"]
        assert 0.0 <= w["probability"] <= 1.0


@pytest.mark.real
def test_real_api_chinese_mp4(client):
    with MP4.open("rb") as f:
        resp = client.post(
            "/api/jobs",
            files={"file": ("chinese_speech_60s.mp4", f, "video/mp4")},
            data={"engine": "cpu", "model": "small"},
        )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    data = wait_done(client, job_id, timeout=300)
    assert data["status"] == "done", data
    assert data["progress"] == 100
    result = client.get(f"/api/jobs/{job_id}/result").json()
    assert "项目进度" in result["text"]
    assert result["language"] == "zh"
    assert result["segments"][0]["words"]