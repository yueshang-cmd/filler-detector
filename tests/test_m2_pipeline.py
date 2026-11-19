import json
import os
import time

from app import jobs
from tests.conftest import SAMPLE_MP4, wait_done
from tests.fakes import FakeEngine


def _upload(client, name="a.mp4", path=SAMPLE_MP4):
    with path.open("rb") as f:
        resp = client.post(
            "/api/jobs",
            files={"file": (name, f, "video/mp4")},
            data={"engine": "cpu", "model": "small"},
        )
    assert resp.status_code == 200
    return resp.json()["job_id"]


def test_state_machine_and_progress_monotonic(client, monkeypatch):
    monkeypatch.setattr(
        jobs, "build_engine", lambda engine, model, language: FakeEngine(delay=0.06)
    )
    job_id = _upload(client)
    samples = []
    deadline = time.time() + 30
    while time.time() < deadline:
        data = client.get(f"/api/jobs/{job_id}").json()
        samples.append((data["status"], data["progress"]))
        if data["status"] in ("done", "failed"):
            break
        time.sleep(0.02)
    assert samples[-1] == ("done", 100), samples
    progresses = [p for _, p in samples]
    assert progresses == sorted(progresses), samples
    order = ["queued", "extracting", "transcribing", "analyzing", "done"]
    seen = [s for s, _ in samples]
    idx = [order.index(s) for s in seen]
    assert idx == sorted(idx), samples
    assert "transcribing" in seen, samples
    assert any(s == "transcribing" and 0 < p < 100 for s, p in samples), samples


def test_result_persisted_and_endpoint(client):
    job_id = _upload(client)
    data = wait_done(client, job_id)
    assert data["status"] == "done", data
    result_path = jobs.UPLOADS_DIR / job_id / "result.json"
    assert result_path.exists()
    on_disk = json.loads(result_path.read_text(encoding="utf-8"))
    assert on_disk["job"]["job_id"] == job_id
    resp = client.get(f"/api/jobs/{job_id}/result")
    assert resp.status_code == 200
    result = resp.json()
    assert result == on_disk["result"]
    assert result["language"] == "zh"
    assert result["duration"] > 0
    assert result["text"]
    assert result["segments"]
    for seg in result["segments"]:
        assert seg["end"] >= seg["start"]
        assert seg["words"]
        for w in seg["words"]:
            assert set(w) == {"word", "start", "end", "probability"}
            assert w["end"] >= w["start"]
    assert data["result"] == result


def test_result_endpoint_not_ready(client, monkeypatch):
    monkeypatch.setattr(
        jobs, "build_engine", lambda engine, model, language: FakeEngine(delay=0.3)
    )
    job_id = _upload(client)
    resp = client.get(f"/api/jobs/{job_id}/result")
    assert resp.status_code == 409
    wait_done(client, job_id)
    assert client.get(f"/api/jobs/{job_id}/result").status_code == 200


def test_result_endpoint_not_found(client):
    assert client.get("/api/jobs/nonexistent/result").status_code == 404


def test_restore_history_after_restart(client):
    job_id = _upload(client)
    wait_done(client, job_id)
    with jobs._jobs_lock:
        jobs._jobs.clear()
    assert client.get("/api/jobs").json() == []
    assert jobs.restore_jobs() == 1
    history = client.get("/api/jobs").json()
    assert len(history) == 1
    assert history[0]["job_id"] == job_id
    assert history[0]["status"] == "done"
    assert history[0]["progress"] == 100
    restored = client.get(f"/api/jobs/{job_id}").json()
    assert restored["result"]["segments"]
    assert client.get(f"/api/jobs/{job_id}/result").status_code == 200
    assert jobs.restore_jobs() == 0


def test_restore_skips_corrupt_result_json(client):
    job_id = _upload(client)
    wait_done(client, job_id)
    bad_dir = jobs.UPLOADS_DIR / "badjob000000"
    bad_dir.mkdir(parents=True)
    (bad_dir / "result.json").write_text("{not json", encoding="utf-8")
    with jobs._jobs_lock:
        jobs._jobs.clear()
    assert jobs.restore_jobs() == 1
    assert client.get("/api/jobs/badjob000000").status_code == 404


def test_failed_job_no_result_file(client):
    resp = client.post(
        "/api/jobs",
        files={"file": ("corrupt.mp4", os.urandom(32 * 1024), "video/mp4")},
    )
    job_id = resp.json()["job_id"]
    data = wait_done(client, job_id)
    assert data["status"] == "failed"
    assert not (jobs.UPLOADS_DIR / job_id / "result.json").exists()
    assert client.get(f"/api/jobs/{job_id}/result").status_code == 409
    with jobs._jobs_lock:
        jobs._jobs.clear()
    assert jobs.restore_jobs() == 0


def test_language_setting_passed_to_engine(client, monkeypatch):
    captured = {}

    def spy(engine, model, language):
        captured["language"] = language
        return FakeEngine()

    monkeypatch.setattr(jobs, "build_engine", spy)
    assert client.put("/api/settings", json={"language": "auto"}).status_code == 200
    job_id = _upload(client)
    wait_done(client, job_id)
    assert captured["language"] == "auto"


def test_language_setting_invalid_rejected(client):
    assert client.put("/api/settings", json={"language": "fr"}).status_code == 400
    assert client.get("/api/settings").json()["language"] == "zh"


def test_rocm_unavailable_falls_back_to_cpu(client, monkeypatch):
    monkeypatch.setattr(
        jobs.engines_probe,
        "probe_engines",
        lambda: {
            "cpu": {"available": True, "detail": ""},
            "cuda": {"available": False, "detail": "无 CUDA 设备"},
            "rocm": {"available": False, "detail": "torch 未安装（ROCm 可选依赖）"},
        },
    )
    seen = []

    def spy(engine, model, language):
        seen.append(engine)
        return FakeEngine()

    monkeypatch.setattr(jobs, "build_engine", spy)
    with SAMPLE_MP4.open("rb") as f:
        resp = client.post(
            "/api/jobs",
            files={"file": ("a.mp4", f, "video/mp4")},
            data={"engine": "rocm"},
        )
    job_id = resp.json()["job_id"]
    data = wait_done(client, job_id)
    assert data["status"] == "done", data
    assert seen == ["cpu"]
    assert data["engine_fallback"]["from"] == "rocm"
    assert data["engine_fallback"]["to"] == "cpu"
    assert "torch" in data["engine_fallback"]["reason"]
    assert data["warnings"]
    assert data["result"]["engine_used"] == "cpu"