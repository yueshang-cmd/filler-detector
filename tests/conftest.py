import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import jobs, settings
from app.main import app
from tests.fakes import FakeEngine

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
SAMPLE_MP4 = ASSETS_DIR / "sample_10s.mp4"

REAL_BUILD_ENGINE = jobs.build_engine


@pytest.fixture(autouse=True)
def isolated_jobs():
    with jobs._jobs_lock:
        jobs._jobs.clear()
    yield
    with jobs._jobs_lock:
        jobs._jobs.clear()


@pytest.fixture(autouse=True)
def fake_asr(request, monkeypatch):
    if request.node.get_closest_marker("real"):
        return
    monkeypatch.setattr(jobs, "build_engine", lambda engine, model, language: FakeEngine())


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(settings, "SETTINGS_FILE", tmp_path / "data" / "settings.json")
    with TestClient(app) as c:
        yield c


def wait_done(client, job_id, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = client.get(f"/api/jobs/{job_id}").json()
        if data["status"] in ("done", "failed"):
            return data
        time.sleep(0.2)
    raise AssertionError(f"job {job_id} 超时未完成")