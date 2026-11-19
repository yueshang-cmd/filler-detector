import json

from app import audio, jobs
from tests.conftest import SAMPLE_MP4, wait_done
from tests.test_keypoints import build_speech_segments

MEETING_SENTENCES = [
    "今天我们讨论项目进度。",
    "上周我们完成了上传模块的开发。",
    "进度比计划慢了一点，转写引擎遇到兼容性问题。",
    "本周先把语音转写做完，下周开始做废话统计。",
    "测试音频还需要补充真实会议录音。",
]


class MeetingEngine:
    def transcribe(self, path, progress_cb=None):
        duration = audio.probe_duration(path)
        segments = build_speech_segments(MEETING_SENTENCES)
        if progress_cb is not None:
            progress_cb(duration)
        return {"segments": segments, "language": "zh", "duration": duration}


def _upload(client):
    with SAMPLE_MP4.open("rb") as f:
        resp = client.post(
            "/api/jobs",
            files={"file": ("a.mp4", f, "video/mp4")},
            data={"engine": "cpu", "model": "small"},
        )
    assert resp.status_code == 200
    return resp.json()["job_id"]


def test_fake_engine_pipeline_has_keypoints(client):
    job_id = _upload(client)
    data = wait_done(client, job_id)
    assert data["status"] == "done", data
    result = client.get(f"/api/jobs/{job_id}/result").json()
    assert result["keypoint_mode"] == "offline"
    assert isinstance(result["keypoints"], list)
    on_disk = json.loads(
        (jobs.UPLOADS_DIR / job_id / "result.json").read_text(encoding="utf-8")
    )
    assert on_disk["result"]["keypoint_mode"] == "offline"
    assert on_disk["result"]["keypoints"] == result["keypoints"]


def test_meeting_engine_keypoints_capture_substance(client, monkeypatch):
    monkeypatch.setattr(jobs, "build_engine", lambda *a: MeetingEngine())
    job_id = _upload(client)
    data = wait_done(client, job_id)
    assert data["status"] == "done", data
    result = client.get(f"/api/jobs/{job_id}/result").json()
    points = result["keypoints"]
    assert result["keypoint_mode"] == "offline"
    assert points
    joined = " ".join(p["point"] for p in points)
    assert "上传模块" in joined
    assert "项目进度" in joined
    assert "语音转写" in joined
    for p in points:
        assert set(p) == {"point", "importance", "start", "end"}
        assert 1 <= p["importance"] <= 5