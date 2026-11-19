"""M2 边界回归（测试工程师补充，仅测试代码，不改业务逻辑）。

覆盖：引擎异常→failed 且 error 清晰；进度回调越界/回退不破坏 0..100 单调；
空转写正常 done；空文件/静音音频走链路不崩。
"""
import subprocess
import time
from pathlib import Path

import pytest

from app import audio, jobs
from tests.conftest import SAMPLE_MP4, wait_done


def _upload(client, name="a.mp4", payload=None, path=SAMPLE_MP4):
    if payload is None:
        with path.open("rb") as f:
            resp = client.post(
                "/api/jobs",
                files={"file": (name, f, "video/mp4")},
                data={"engine": "cpu", "model": "small"},
            )
    else:
        resp = client.post(
            "/api/jobs",
            files={"file": (name, payload, "application/octet-stream")},
            data={"engine": "cpu", "model": "small"},
        )
    assert resp.status_code == 200
    return resp.json()["job_id"]


class ExplodingEngine:
    def transcribe(self, path, progress_cb=None):
        raise RuntimeError("引擎内部爆炸：模型权重损坏 (测试注入)")


class WildProgressEngine:
    """回调值越界（超总时长、负数、回退），验证管线 clamp 与单调保护。"""

    def transcribe(self, path, progress_cb=None):
        duration = audio.probe_duration(path)
        for end in (duration * 5, -3.0, duration * 0.4, duration * 0.2, duration * 20):
            if progress_cb is not None:
                progress_cb(end)
            time.sleep(0.05)
        return {
            "segments": [
                {"start": 0.0, "end": duration, "text": "嗯", "words": [
                    {"word": "嗯", "start": 0.0, "end": 0.3, "probability": 0.9}
                ]}
            ],
            "language": "zh",
            "duration": duration,
        }


class EmptyTranscriptEngine:
    def transcribe(self, path, progress_cb=None):
        return {"segments": [], "language": "zh", "duration": audio.probe_duration(path)}


def test_engine_exception_marks_failed_with_clear_error(client, monkeypatch):
    monkeypatch.setattr(jobs, "build_engine", lambda *a: ExplodingEngine())
    job_id = _upload(client)
    data = wait_done(client, job_id)
    assert data["status"] == "failed", data
    assert "引擎内部爆炸" in data["error"] and "测试注入" in data["error"]
    assert data["result"] is None
    assert not (jobs.UPLOADS_DIR / job_id / "result.json").exists()
    assert client.get(f"/api/jobs/{job_id}/result").status_code == 409
    # 服务不崩，后续任务仍可正常完成（恢复 FakeEngine 再上传）
    assert client.get("/api/settings").status_code == 200
    from tests.fakes import FakeEngine

    monkeypatch.setattr(jobs, "build_engine", lambda *a: FakeEngine())
    ok_id = _upload(client, name="b.mp4")
    assert wait_done(client, ok_id)["status"] == "done"


def test_progress_never_exceeds_100_and_monotonic(client, monkeypatch):
    monkeypatch.setattr(jobs, "build_engine", lambda *a: WildProgressEngine())
    job_id = _upload(client)
    samples = []
    deadline = time.time() + 30
    while time.time() < deadline:
        d = client.get(f"/api/jobs/{job_id}").json()
        samples.append((d["status"], d["progress"]))
        if d["status"] in ("done", "failed"):
            break
        time.sleep(0.01)
    assert samples[-1] == ("done", 100), samples
    progresses = [p for _, p in samples]
    assert all(0 <= p <= 100 for p in progresses), samples
    assert progresses == sorted(progresses), samples  # 回退回调不得使进度倒退


def test_empty_transcript_completes_done(client, monkeypatch):
    monkeypatch.setattr(jobs, "build_engine", lambda *a: EmptyTranscriptEngine())
    job_id = _upload(client)
    data = wait_done(client, job_id)
    assert data["status"] == "done", data
    assert data["progress"] == 100
    result = client.get(f"/api/jobs/{job_id}/result").json()
    assert result["segments"] == []
    assert result["text"] == ""
    # 空转写也落盘、可恢复
    assert (jobs.UPLOADS_DIR / job_id / "result.json").exists()
    with jobs._jobs_lock:
        jobs._jobs.clear()
    assert jobs.restore_jobs() == 1
    assert client.get(f"/api/jobs/{job_id}/result").status_code == 200


def test_zero_byte_upload_fails_gracefully(client):
    job_id = _upload(client, name="empty.wav", payload=b"")
    data = wait_done(client, job_id)
    assert data["status"] == "failed", data
    assert data["error"] and data["error"].strip()
    assert client.get("/api/jobs").status_code == 200  # 服务不崩


def _make_silence_wav(dst: Path, seconds: float = 1.0) -> Path:
    ffmpeg, _ = audio.find_ffmpeg()
    proc = subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i",
         f"anullsrc=r=16000:cl=mono", "-t", str(seconds), str(dst)],
        capture_output=True, text=True, errors="replace",
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    return dst


def test_silence_wav_pipeline_with_fake_engine(client, tmp_path):
    wav = _make_silence_wav(tmp_path / "silence_1s.wav")
    assert audio.probe_duration(wav) == pytest.approx(1.0, abs=0.1)
    job_id = _upload(client, name="silence_1s.wav", path=wav)
    data = wait_done(client, job_id)
    assert data["status"] == "done", data


@pytest.mark.real
def test_silence_wav_real_engine_vad_empty_done(client, tmp_path):
    """真实引擎 + VAD：1s 静音应正常 done，segments 可为空，不崩不挂。"""
    wav = _make_silence_wav(tmp_path / "silence_1s.wav")
    job_id = _upload(client, name="silence_1s.wav", path=wav)
    data = wait_done(client, job_id, timeout=240)
    assert data["status"] == "done", data
    assert data["progress"] == 100
    result = client.get(f"/api/jobs/{job_id}/result").json()
    assert isinstance(result["segments"], list)  # VAD 过滤后允许为空
    assert result["text"] == "".join(s["text"] for s in result["segments"])
    assert result["duration"] == pytest.approx(1.0, abs=0.2)