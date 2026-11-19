import json
import time

from app import audio, filler, jobs
from tests.conftest import SAMPLE_MP4, wait_done
from tests.test_filler import STATS_SEGMENTS


class FillerEngine:
    def transcribe(self, path, progress_cb=None):
        duration = audio.probe_duration(path)
        if progress_cb is not None:
            progress_cb(duration)
        return {"segments": STATS_SEGMENTS, "language": "zh", "duration": duration}


def _upload(client):
    with SAMPLE_MP4.open("rb") as f:
        resp = client.post(
            "/api/jobs",
            files={"file": ("a.mp4", f, "video/mp4")},
            data={"engine": "cpu", "model": "small"},
        )
    assert resp.status_code == 200
    return resp.json()["job_id"]


def test_full_pipeline_analysis_smoke(client, monkeypatch):
    monkeypatch.setattr(jobs, "build_engine", lambda *a: FillerEngine())
    job_id = _upload(client)
    data = wait_done(client, job_id)
    assert data["status"] == "done", data
    result = client.get(f"/api/jobs/{job_id}/result").json()
    analysis = result["analysis"]
    assert analysis == filler.analyze(STATS_SEGMENTS)
    assert analysis["filler_count"] == 2
    assert analysis["total_tokens"] == 13
    assert analysis["total_speech_duration"] == 4.0
    assert analysis["filler_count_ratio"] == round(2 / 13, 6)
    assert analysis["filler_duration_ratio"] == 0.225
    assert [f["word"] for f in analysis["fillers"]] == ["嗯", "这个"]
    assert result["text"] == "嗯,这个这个,我想说进度.然后测试也跑通了"
    on_disk = json.loads(
        (jobs.UPLOADS_DIR / job_id / "result.json").read_text(encoding="utf-8")
    )
    assert on_disk["result"]["analysis"] == analysis


def test_analyzing_stage_between_transcribing_and_done(client, monkeypatch):
    real_analyze = filler.analyze

    def slow_analyze(segments, *args, **kwargs):
        time.sleep(0.4)
        return real_analyze(segments, *args, **kwargs)

    monkeypatch.setattr(filler, "analyze", slow_analyze)
    job_id = _upload(client)
    seen = []
    deadline = time.time() + 30
    while time.time() < deadline:
        d = client.get(f"/api/jobs/{job_id}").json()
        seen.append((d["status"], d["progress"]))
        if d["status"] in ("done", "failed"):
            break
        time.sleep(0.02)
    assert seen[-1] == ("done", 100), seen
    statuses = [s for s, _ in seen]
    assert "analyzing" in statuses, seen
    order = ["queued", "extracting", "transcribing", "analyzing", "done"]
    idx = [order.index(s) for s in statuses]
    assert idx == sorted(idx), seen
    progresses = [p for _, p in seen]
    assert progresses == sorted(progresses), seen


def test_hallucination_segments_filtered_in_pipeline(client):
    job_id = _upload(client)
    data = wait_done(client, job_id)
    assert data["status"] == "done", data
    result = client.get(f"/api/jobs/{job_id}/result").json()
    assert len(result["segments"]) == 1
    assert result["text"] == "嗯,大家好"
    analysis = result["analysis"]
    assert analysis["filler_count"] == 1
    assert analysis["fillers"][0]["word"] == "嗯"
    assert analysis["total_tokens"] == 3
    assert analysis["total_speech_duration"] == 2.0
    assert analysis["filler_count_ratio"] == round(1 / 3, 6)
    assert analysis["filler_duration_ratio"] == 0.15


def test_vocab_from_settings_used_in_pipeline(client, monkeypatch):
    monkeypatch.setattr(jobs, "build_engine", lambda *a: FillerEngine())
    resp = client.put(
        "/api/settings",
        json={"filler_words": {"pure": ["测试"], "contextual": []}},
    )
    assert resp.status_code == 200
    job_id = _upload(client)
    assert wait_done(client, job_id)["status"] == "done"
    result = client.get(f"/api/jobs/{job_id}/result").json()
    assert [f["word"] for f in result["analysis"]["fillers"]] == ["测试"]