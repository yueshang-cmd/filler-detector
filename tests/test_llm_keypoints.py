import json

import httpx
import pytest

from app import keypoints
from app.settings import DEFAULT_SETTINGS
from tests.test_keypoints import FIXTURE_SENTENCES, build_speech_segments

CFG = {
    "llm": {
        "enabled": True,
        "base_url": "http://127.0.0.1:9/v1",
        "key": "test",
        "model": "x",
    }
}

LLM_POINTS = [
    {"point": "上传模块开发完成", "importance": 4, "timestamp_start": 2.0, "timestamp_end": 4.0},
    {"point": "统计功能下周上线", "importance": 5, "timestamp_start": 6.0, "timestamp_end": 8.0},
]


class FakeResponse:
    def __init__(self, content, status_code=200):
        self._content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=None, response=None
            )

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


@pytest.fixture()
def segments():
    return build_speech_segments(FIXTURE_SENTENCES)


def _enable_llm(monkeypatch):
    monkeypatch.setattr(
        keypoints, "load_settings", lambda: {**DEFAULT_SETTINGS, "llm": CFG["llm"]}
    )


def test_llm_success_points_from_llm(monkeypatch, segments):
    _enable_llm(monkeypatch)
    captured = {}

    def fake_post(url, **kwargs):
        captured.update(
            url=url,
            payload=kwargs["json"],
            headers=kwargs["headers"],
            timeout=kwargs["timeout"],
        )
        return FakeResponse(json.dumps(LLM_POINTS, ensure_ascii=False))

    monkeypatch.setattr(keypoints.httpx, "post", fake_post)
    report = keypoints.extract_keypoints_detailed(segments, mode="llm")
    assert report["keypoint_mode"] == "llm"
    assert report["warnings"] == []
    assert report["keypoints"] == [
        {"point": "统计功能下周上线", "importance": 5, "start": 6.0, "end": 8.0},
        {"point": "上传模块开发完成", "importance": 4, "start": 2.0, "end": 4.0},
    ]
    assert captured["url"] == "http://127.0.0.1:9/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test"
    assert captured["payload"]["model"] == "x"
    assert captured["timeout"] == keypoints.LLM_TIMEOUT_SECONDS
    user_msg = captured["payload"]["messages"][-1]["content"]
    assert "上传模块开发完成" in user_msg
    assert "[" in user_msg and "]" in user_msg


def test_llm_json_embedded_in_prose(monkeypatch, segments):
    _enable_llm(monkeypatch)
    content = "好的，以下是重点：\n```json\n" + json.dumps(LLM_POINTS, ensure_ascii=False) + "\n```\n希望有帮助"
    monkeypatch.setattr(keypoints.httpx, "post", lambda *a, **k: FakeResponse(content))
    report = keypoints.extract_keypoints_detailed(segments, mode="llm")
    assert report["keypoint_mode"] == "llm"
    assert len(report["keypoints"]) == 2


def test_llm_bad_json_falls_back_offline(monkeypatch, segments):
    _enable_llm(monkeypatch)
    monkeypatch.setattr(keypoints.httpx, "post", lambda *a, **k: FakeResponse("这不是 JSON"))
    report = keypoints.extract_keypoints_detailed(segments, mode="llm")
    assert report["keypoint_mode"] == "offline"
    assert any("LLM 失败已回退离线" in w for w in report["warnings"])
    assert report["keypoints"] == keypoints.extract_keypoints(segments)


def test_llm_network_error_falls_back_offline(monkeypatch, segments):
    _enable_llm(monkeypatch)

    def fake_post(*a, **k):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(keypoints.httpx, "post", fake_post)
    report = keypoints.extract_keypoints_detailed(segments, mode="llm")
    assert report["keypoint_mode"] == "offline"
    assert any("LLM 失败已回退离线" in w for w in report["warnings"])
    assert report["keypoints"]


def test_llm_timeout_falls_back_offline(monkeypatch, segments):
    _enable_llm(monkeypatch)

    def fake_post(*a, **k):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(keypoints.httpx, "post", fake_post)
    report = keypoints.extract_keypoints_detailed(segments, mode="llm")
    assert report["keypoint_mode"] == "offline"
    assert any("LLM 失败已回退离线" in w for w in report["warnings"])


def test_llm_http_error_falls_back_offline(monkeypatch, segments):
    _enable_llm(monkeypatch)
    monkeypatch.setattr(keypoints.httpx, "post", lambda *a, **k: FakeResponse("", status_code=500))
    report = keypoints.extract_keypoints_detailed(segments, mode="llm")
    assert report["keypoint_mode"] == "offline"
    assert report["warnings"]


def test_llm_empty_points_falls_back_offline(monkeypatch, segments):
    _enable_llm(monkeypatch)
    monkeypatch.setattr(keypoints.httpx, "post", lambda *a, **k: FakeResponse("[]"))
    report = keypoints.extract_keypoints_detailed(segments, mode="llm")
    assert report["keypoint_mode"] == "offline"
    assert any("LLM 失败已回退离线" in w for w in report["warnings"])


def test_llm_disabled_never_calls_http(monkeypatch, segments):
    monkeypatch.setattr(
        keypoints, "load_settings", lambda: dict(DEFAULT_SETTINGS)
    )

    def fake_post(*a, **k):
        raise AssertionError("LLM 未启用时不应发起 HTTP 请求")

    monkeypatch.setattr(keypoints.httpx, "post", fake_post)
    report = keypoints.extract_keypoints_detailed(segments, mode="llm")
    assert report["keypoint_mode"] == "offline"
    assert report["warnings"] == []


def test_llm_chunks_merge_dedupe_sort(monkeypatch):
    _enable_llm(monkeypatch)
    long_segments = [
        {"start": float(i * 100), "end": float(i * 100 + 90), "text": "内容" * 700, "words": []}
        for i in range(3)
    ]
    assert len(keypoints._build_chunks(long_segments)) == 2
    responses = [
        FakeResponse(json.dumps([
            {"point": "重点甲", "importance": 3, "timestamp_start": 0, "timestamp_end": 90},
            {"point": "重点乙", "importance": 5, "timestamp_start": 0, "timestamp_end": 90},
        ], ensure_ascii=False)),
        FakeResponse(json.dumps([
            {"point": "重点甲", "importance": 5, "timestamp_start": 100, "timestamp_end": 190},
            {"point": "重点丙", "importance": 2, "timestamp_start": 200, "timestamp_end": 290},
        ], ensure_ascii=False)),
    ]
    calls = []

    def fake_post(*a, **k):
        calls.append(k.get("json") or (a[1] if len(a) > 1 else None))
        return responses[len(calls) - 1]

    monkeypatch.setattr(keypoints.httpx, "post", fake_post)
    report = keypoints.extract_keypoints_detailed(long_segments, mode="llm")
    assert len(calls) == 2
    assert report["keypoint_mode"] == "llm"
    points = report["keypoints"]
    assert [p["point"] for p in points] == ["重点乙", "重点甲", "重点丙"]
    by_text = {p["point"]: p for p in points}
    assert by_text["重点甲"]["importance"] == 5
    assert by_text["重点甲"]["start"] == 100.0
    imps = [p["importance"] for p in points]
    assert imps == sorted(imps, reverse=True)


def test_llm_tolerates_invalid_items(monkeypatch, segments):
    _enable_llm(monkeypatch)
    content = json.dumps([
        {"point": "合法重点", "importance": "high", "timestamp_start": 1, "timestamp_end": 2},
        {"point": "缺时间戳", "importance": 4},
        {"point": "", "importance": 5, "timestamp_start": 0, "timestamp_end": 1},
        "不是对象",
        {"point": "倒序时间戳", "importance": 9, "timestamp_start": 5, "timestamp_end": 3},
    ], ensure_ascii=False)
    monkeypatch.setattr(keypoints.httpx, "post", lambda *a, **k: FakeResponse(content))
    report = keypoints.extract_keypoints_detailed(segments, mode="llm")
    points = report["keypoints"]
    assert {p["point"] for p in points} == {"合法重点", "缺时间戳", "倒序时间戳"}
    by_text = {p["point"]: p for p in points}
    assert by_text["合法重点"]["importance"] == 3
    assert by_text["缺时间戳"]["start"] == 0.0
    assert by_text["倒序时间戳"]["importance"] == 5
    assert by_text["倒序时间戳"]["end"] == 5.0


def test_llm_empty_segments_no_call(monkeypatch):
    _enable_llm(monkeypatch)

    def fake_post(*a, **k):
        raise AssertionError("空转写不应调用 LLM")

    monkeypatch.setattr(keypoints.httpx, "post", fake_post)
    report = keypoints.extract_keypoints_detailed([], mode="llm")
    assert report["keypoint_mode"] == "llm"
    assert report["keypoints"] == []


def test_job_llm_failure_warning_in_result(client, monkeypatch):
    assert client.put("/api/settings", json={
        "llm": {"enabled": True, "base_url": "http://127.0.0.1:9/v1", "key": "test", "model": "x"}
    }).status_code == 200

    def fake_post(*a, **k):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(keypoints.httpx, "post", fake_post)
    from tests.conftest import SAMPLE_MP4, wait_done

    with SAMPLE_MP4.open("rb") as f:
        resp = client.post(
            "/api/jobs",
            files={"file": ("a.mp4", f, "video/mp4")},
            data={"engine": "cpu"},
        )
    job_id = resp.json()["job_id"]
    data = wait_done(client, job_id)
    assert data["status"] == "done", data
    assert data["result"]["keypoint_mode"] == "offline"
    assert any("LLM 失败已回退离线" in w for w in data["warnings"])
    assert isinstance(data["result"]["keypoints"], list)