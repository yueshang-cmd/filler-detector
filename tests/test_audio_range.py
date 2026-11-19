import pytest

from app import jobs
from tests.conftest import SAMPLE_MP4, wait_done


@pytest.fixture()
def audio_job(client):
    with SAMPLE_MP4.open("rb") as f:
        resp = client.post(
            "/api/jobs",
            files={"file": ("a.mp4", f, "video/mp4")},
            data={"engine": "cpu", "model": "small"},
        )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    assert wait_done(client, job_id)["status"] == "done"
    wav_path = jobs.UPLOADS_DIR / job_id / "audio.wav"
    return client, job_id, wav_path.read_bytes()


def test_no_range_returns_full_200(audio_job):
    client, job_id, data = audio_job
    resp = client.get(f"/api/audio/{job_id}")
    assert resp.status_code == 200
    assert resp.content == data
    assert resp.headers["accept-ranges"] == "bytes"
    assert resp.headers["content-type"] == "audio/wav"
    assert int(resp.headers["content-length"]) == len(data)


def test_range_prefix_206(audio_job):
    client, job_id, data = audio_job
    resp = client.get(f"/api/audio/{job_id}", headers={"Range": "bytes=0-99"})
    assert resp.status_code == 206
    assert resp.content == data[:100]
    assert resp.headers["content-range"] == f"bytes 0-99/{len(data)}"
    assert resp.headers["accept-ranges"] == "bytes"
    assert int(resp.headers["content-length"]) == 100


def test_range_middle_bytes_exact(audio_job):
    client, job_id, data = audio_job
    resp = client.get(f"/api/audio/{job_id}", headers={"Range": "bytes=1000-1999"})
    assert resp.status_code == 206
    assert resp.content == data[1000:2000]
    assert resp.headers["content-range"] == f"bytes 1000-1999/{len(data)}"


def test_range_open_ended(audio_job):
    client, job_id, data = audio_job
    resp = client.get(f"/api/audio/{job_id}", headers={"Range": "bytes=500-"})
    assert resp.status_code == 206
    assert resp.content == data[500:]
    assert resp.headers["content-range"] == f"bytes 500-{len(data) - 1}/{len(data)}"


def test_range_suffix(audio_job):
    client, job_id, data = audio_job
    resp = client.get(f"/api/audio/{job_id}", headers={"Range": "bytes=-200"})
    assert resp.status_code == 206
    assert resp.content == data[-200:]
    assert resp.headers["content-range"] == (
        f"bytes {len(data) - 200}-{len(data) - 1}/{len(data)}"
    )


def test_range_end_clamped_to_eof(audio_job):
    client, job_id, data = audio_job
    resp = client.get(
        f"/api/audio/{job_id}", headers={"Range": f"bytes=100-{len(data) + 500}"}
    )
    assert resp.status_code == 206
    assert resp.content == data[100:]
    assert resp.headers["content-range"] == f"bytes 100-{len(data) - 1}/{len(data)}"


def test_range_start_beyond_eof_416(audio_job):
    client, job_id, data = audio_job
    resp = client.get(f"/api/audio/{job_id}", headers={"Range": f"bytes={len(data)}-"})
    assert resp.status_code == 416
    assert resp.headers["content-range"] == f"bytes */{len(data)}"


def test_range_start_after_end_416(audio_job):
    client, job_id, _ = audio_job
    resp = client.get(f"/api/audio/{job_id}", headers={"Range": "bytes=200-100"})
    assert resp.status_code == 416


def test_range_suffix_zero_416(audio_job):
    client, job_id, _ = audio_job
    resp = client.get(f"/api/audio/{job_id}", headers={"Range": "bytes=-0"})
    assert resp.status_code == 416


def test_unparsable_range_ignored(audio_job):
    client, job_id, data = audio_job
    resp = client.get(f"/api/audio/{job_id}", headers={"Range": "bytes=abc"})
    assert resp.status_code == 200
    assert resp.content == data


def test_multi_range_served_as_full(audio_job):
    client, job_id, data = audio_job
    resp = client.get(f"/api/audio/{job_id}", headers={"Range": "bytes=0-99,200-299"})
    assert resp.status_code == 200
    assert resp.content == data


def test_unknown_job_404(client):
    assert client.get("/api/audio/nonexistent").status_code == 404


def test_missing_wav_404(audio_job):
    client, job_id, _ = audio_job
    (jobs.UPLOADS_DIR / job_id / "audio.wav").unlink()
    assert client.get(f"/api/audio/{job_id}").status_code == 404