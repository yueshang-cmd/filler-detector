import wave

from tests.conftest import SAMPLE_MP4, wait_done


def test_upload_mp4_job_done(client):
    with SAMPLE_MP4.open("rb") as f:
        resp = client.post(
            "/api/jobs",
            files={"file": ("sample_10s.mp4", f, "video/mp4")},
            data={"engine": "cpu", "model": "small"},
        )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    data = wait_done(client, job_id)
    assert data["status"] == "done", data
    assert data["progress"] == 100
    assert data["error"] is None
    wav_path = data["result"]["wav_path"]
    assert data["result"]["duration"] == 10.0 or abs(data["result"]["duration"] - 10.0) < 0.5
    with wave.open(wav_path, "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getnchannels() == 1


def test_job_defaults_and_history(client):
    with SAMPLE_MP4.open("rb") as f:
        resp = client.post("/api/jobs", files={"file": ("a.mp4", f, "video/mp4")})
    job_id = resp.json()["job_id"]
    data = wait_done(client, job_id)
    assert data["engine"] == "cpu"
    assert data["model"] == "small"
    history = client.get("/api/jobs").json()
    assert any(j["job_id"] == job_id for j in history)


def test_upload_bad_extension(client):
    resp = client.post(
        "/api/jobs", files={"file": ("virus.exe", b"MZ", "application/octet-stream")}
    )
    assert resp.status_code == 400


def test_upload_bad_engine(client):
    with SAMPLE_MP4.open("rb") as f:
        resp = client.post(
            "/api/jobs",
            files={"file": ("a.mp4", f, "video/mp4")},
            data={"engine": "tpu"},
        )
    assert resp.status_code == 400


def test_job_not_found(client):
    assert client.get("/api/jobs/nonexistent").status_code == 404


def test_settings_defaults(client):
    data = client.get("/api/settings").json()
    assert data["default_engine"] == "cpu"
    assert data["default_model"] == "small"
    assert "嗯" in data["filler_words"]["pure"]
    assert "然后" in data["filler_words"]["contextual"]


def test_settings_put_persist(client):
    resp = client.put("/api/settings", json={"default_model": "medium"})
    assert resp.status_code == 200
    assert client.get("/api/settings").json()["default_model"] == "medium"


def test_settings_put_invalid(client):
    resp = client.put("/api/settings", json={"default_engine": "tpu"})
    assert resp.status_code == 400


def test_static_index(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "废话检测器" in resp.text