import json
import time

from app import jobs
from app.asr.base import ASREngine
from tests.conftest import SAMPLE_MP4, wait_done
from tests.fakes import FakeEngine


def _probe(cuda=False, rocm=False, detail="模拟不可用"):
    return {
        "cpu": {"available": True, "detail": ""},
        "cuda": {"available": cuda, "detail": detail},
        "rocm": {"available": rocm, "detail": detail},
    }


def _upload(client, engine):
    with SAMPLE_MP4.open("rb") as f:
        resp = client.post(
            "/api/jobs",
            files={"file": ("a.mp4", f, "video/mp4")},
            data={"engine": engine},
        )
    assert resp.status_code == 200
    return resp.json()["job_id"]


class StallEngine(ASREngine):
    def transcribe(self, path, progress_cb=None):
        time.sleep(3)
        return {"segments": [], "language": "zh", "duration": 0.0}


class ExplodingEngine(ASREngine):
    def transcribe(self, path, progress_cb=None):
        raise RuntimeError("CUDA 驱动崩溃（模拟）")


def test_cuda_unavailable_falls_back_to_cpu(client, monkeypatch):
    monkeypatch.setattr(jobs.engines_probe, "probe_engines", lambda: _probe(detail="无 CUDA 设备"))
    seen = []

    def spy(engine, model, language):
        seen.append(engine)
        return FakeEngine()

    monkeypatch.setattr(jobs, "build_engine", spy)
    job_id = _upload(client, "cuda")
    data = wait_done(client, job_id)
    assert data["status"] == "done", data
    assert seen == ["cpu"]
    assert data["engine_fallback"] == {
        "from": "cuda",
        "to": "cpu",
        "reason": "无 CUDA 设备",
    }
    assert any("回退 CPU" in w for w in data["warnings"])
    assert data["result"]["engine_used"] == "cpu"
    assert data["result"]["engine_fallback"]["from"] == "cuda"


def test_rocm_unavailable_falls_back_to_cpu(client, monkeypatch):
    monkeypatch.setattr(jobs.engines_probe, "probe_engines", lambda: _probe(detail="torch 未安装"))
    monkeypatch.setattr(jobs, "build_engine", lambda engine, model, language: FakeEngine())
    job_id = _upload(client, "rocm")
    data = wait_done(client, job_id)
    assert data["status"] == "done", data
    assert data["engine_fallback"]["from"] == "rocm"
    assert data["result"]["engine_used"] == "cpu"
    assert data["warnings"]


def test_cuda_engine_error_falls_back_to_cpu(client, monkeypatch):
    monkeypatch.setattr(jobs.engines_probe, "probe_engines", lambda: _probe(cuda=True))

    def spy(engine, model, language):
        return ExplodingEngine() if engine == "cuda" else FakeEngine()

    monkeypatch.setattr(jobs, "build_engine", spy)
    job_id = _upload(client, "cuda")
    data = wait_done(client, job_id)
    assert data["status"] == "done", data
    assert data["engine_fallback"]["from"] == "cuda"
    assert "CUDA 驱动崩溃" in data["engine_fallback"]["reason"]
    assert data["warnings"]


def test_watchdog_stall_falls_back_to_cpu(client, monkeypatch):
    assert client.put("/api/settings", json={"watchdog_seconds": 1}).status_code == 200
    monkeypatch.setattr(jobs.engines_probe, "probe_engines", lambda: _probe(cuda=True))

    def spy(engine, model, language):
        return StallEngine() if engine == "cuda" else FakeEngine()

    monkeypatch.setattr(jobs, "build_engine", spy)
    job_id = _upload(client, "cuda")
    data = wait_done(client, job_id, timeout=30)
    assert data["status"] == "done", data
    assert data["engine_fallback"]["from"] == "cuda"
    assert "看门狗" in data["engine_fallback"]["reason"]
    assert any("看门狗" in w for w in data["warnings"])
    assert data["result"]["engine_used"] == "cpu"


def test_gpu_with_progress_not_killed_by_watchdog(client, monkeypatch):
    assert client.put("/api/settings", json={"watchdog_seconds": 1}).status_code == 200
    monkeypatch.setattr(jobs.engines_probe, "probe_engines", lambda: _probe(cuda=True))
    monkeypatch.setattr(
        jobs, "build_engine", lambda engine, model, language: FakeEngine(delay=0.3)
    )
    job_id = _upload(client, "cuda")
    data = wait_done(client, job_id, timeout=30)
    assert data["status"] == "done", data
    assert data["engine_fallback"] is None
    assert data["result"]["engine_used"] == "cuda"


def test_cpu_job_no_fallback_fields(client):
    job_id = _upload(client, "cpu")
    data = wait_done(client, job_id)
    assert data["status"] == "done", data
    assert data["engine_fallback"] is None
    assert data["warnings"] == []
    assert data["result"]["engine_used"] == "cpu"


def test_fallback_persisted_and_restored(client, monkeypatch):
    monkeypatch.setattr(jobs.engines_probe, "probe_engines", lambda: _probe(detail="无 CUDA 设备"))
    monkeypatch.setattr(jobs, "build_engine", lambda engine, model, language: FakeEngine())
    job_id = _upload(client, "cuda")
    wait_done(client, job_id)
    result_path = jobs.UPLOADS_DIR / job_id / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["result"]["engine_fallback"]["from"] == "cuda"
    assert payload["result"]["warnings"]
    with jobs._jobs_lock:
        jobs._jobs.clear()
    assert jobs.restore_jobs() >= 1
    view = client.get(f"/api/jobs/{job_id}").json()
    assert view["engine_fallback"]["from"] == "cuda"
    assert view["warnings"]