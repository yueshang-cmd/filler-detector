"""M5 回归验收对抗测试（QA 补充）。

覆盖：引擎探测 monkeypatch 对抗、看门狗（含线程不泄漏）、GPU 回退后与
直接 CPU 任务结果一致性、LLM 成功/容错/回退/分块/禁用零请求、settings 校验。
只补测试，不改业务逻辑；已知 bug 用 xfail 记录（见文件末尾）。
"""

import json
import sys
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from app import audio, engines, jobs, keypoints, settings
from app.asr.base import ASREngine
from app.asr.watchdog import WatchdogTimeout, run_with_watchdog
from app.main import app
from app.settings import DEFAULT_SETTINGS
from tests.conftest import SAMPLE_MP4, wait_done
from tests.fakes import FakeEngine
from tests.test_keypoints import FIXTURE_SENTENCES, build_speech_segments


# ---------------------------------------------------------------- engines


def _fake_ct2(monkeypatch, count):
    monkeypatch.setitem(
        sys.modules, "ctranslate2", SimpleNamespace(get_cuda_device_count=lambda: count)
    )


def test_qa_cuda_detected_via_ct2_device_count(monkeypatch):
    _fake_ct2(monkeypatch, 1)
    monkeypatch.setitem(sys.modules, "torch", None)
    cuda = engines.probe_engines()["cuda"]
    assert cuda["available"] is True
    assert "1" in cuda["detail"]


def test_qa_rocm_true_when_hip_and_gpu(monkeypatch):
    _fake_ct2(monkeypatch, 0)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            version=SimpleNamespace(hip="7.0.0"),
            cuda=SimpleNamespace(is_available=lambda: True),
        ),
    )
    rocm = engines.probe_engines()["rocm"]
    assert rocm["available"] is True
    assert "7.0.0" in rocm["detail"]


def test_qa_cuda_import_exception_no_crash(monkeypatch):
    # sys.modules 置 None → import 抛 ImportError，必须被吞掉
    monkeypatch.setitem(sys.modules, "ctranslate2", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    data = engines.probe_engines()
    assert data["cuda"]["available"] is False
    assert data["cuda"]["detail"]
    assert data["cpu"]["available"] is True


class _ExplodingTorch:
    @property
    def version(self):
        raise RuntimeError("hip runtime 初始化爆炸（模拟）")


def test_qa_rocm_generic_exception_no_crash(monkeypatch):
    _fake_ct2(monkeypatch, 0)
    monkeypatch.setitem(sys.modules, "torch", _ExplodingTorch())
    rocm = engines.probe_engines()["rocm"]
    assert rocm["available"] is False
    assert "爆炸" in rocm["detail"]


def test_qa_rocm_torch_without_version_attr(monkeypatch):
    _fake_ct2(monkeypatch, 0)
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    )
    rocm = engines.probe_engines()["rocm"]
    assert rocm["available"] is False
    assert rocm["detail"]


def test_qa_probe_both_broken_still_returns_three_keys(monkeypatch):
    monkeypatch.setitem(sys.modules, "ctranslate2", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    data = engines.probe_engines()
    assert set(data) == {"cpu", "cuda", "rocm"}
    assert data["cpu"]["available"] is True
    assert all(isinstance(v["detail"], str) and v["detail"] for v in data.values())


# ---------------------------------------------------------------- watchdog


class _SpeechEngine(ASREngine):
    """确定性真实内容引擎。

    注意：FakeEngine 的"嗯,大家好"重复转写会被幻觉过滤删段，剩余句仅含
    停用词 → 离线重点为空属算法预期，故回退完整性断言改用本引擎。
    """

    def transcribe(self, path, progress_cb=None):
        duration = audio.probe_duration(path)
        segments = build_speech_segments(FIXTURE_SENTENCES)
        for seg in segments:
            if progress_cb is not None:
                progress_cb(seg["end"])
        return {"segments": segments, "language": "zh", "duration": duration}


class _StallEngine(ASREngine):
    """模拟 GPU 挂起：睡 stall 秒后才返回（看门狗应先中止并回退）。"""

    def __init__(self, stall=2.5):
        self.stall = stall

    def transcribe(self, path, progress_cb=None):
        time.sleep(self.stall)
        return {"segments": [], "language": "zh", "duration": 0.0}


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


def test_qa_watchdog_stall_fallback_and_no_thread_leak(client, monkeypatch):
    assert client.put("/api/settings", json={"watchdog_seconds": 1}).status_code == 200
    monkeypatch.setattr(jobs.engines_probe, "probe_engines", lambda: _probe(cuda=True))
    monkeypatch.setattr(
        jobs,
        "build_engine",
        lambda engine, model, language: _StallEngine(2.5) if engine == "cuda" else _SpeechEngine(),
    )
    baseline = len(threading.enumerate())
    job_id = _upload(client, "cuda")
    data = wait_done(client, job_id, timeout=30)
    assert data["status"] == "done", data
    assert data["engine_fallback"]["from"] == "cuda"
    assert "看门狗" in data["engine_fallback"]["reason"]
    assert any("看门狗" in w for w in data["warnings"])
    assert data["result"]["engine_used"] == "cpu"
    # 回退后分析功能完整
    assert data["result"]["analysis"]["total_tokens"] > 0
    assert data["result"]["keypoints"]
    # 挂起的 worker 线程（daemon）自行结束后线程数回落，无泄漏
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if len(threading.enumerate()) <= baseline:
            break
        time.sleep(0.2)
    assert len(threading.enumerate()) <= baseline, [t.name for t in threading.enumerate()]


def test_qa_watchdog_heartbeat_engine_not_killed(client, monkeypatch):
    assert client.put("/api/settings", json={"watchdog_seconds": 1}).status_code == 200
    monkeypatch.setattr(jobs.engines_probe, "probe_engines", lambda: _probe(cuda=True))
    monkeypatch.setattr(
        jobs, "build_engine", lambda engine, model, language: FakeEngine(delay=0.25)
    )
    job_id = _upload(client, "cuda")
    data = wait_done(client, job_id, timeout=30)
    assert data["status"] == "done", data
    assert data["engine_fallback"] is None
    assert data["warnings"] == []
    assert data["result"]["engine_used"] == "cuda"


def test_qa_watchdog_worker_daemon_and_join_controllable():
    before = set(threading.enumerate())

    def target(heartbeat):
        time.sleep(1.0)
        return "late"

    start = time.monotonic()
    with pytest.raises(WatchdogTimeout, match="无进度"):
        run_with_watchdog(target, 0.2, poll_interval=0.05)
    assert time.monotonic() - start < 0.9  # 超时及时抛出，不等 worker
    new_threads = set(threading.enumerate()) - before
    assert new_threads, "应能观察到 worker 线程"
    for t in new_threads:
        assert t.daemon, "看门狗 worker 必须是 daemon，不得阻塞进程退出"
        t.join(timeout=3)
        assert not t.is_alive(), "worker 应在有限时间内结束（join 超时可控）"


def test_qa_watchdog_rocm_stall_fallback(client, monkeypatch):
    assert client.put("/api/settings", json={"watchdog_seconds": 1}).status_code == 200
    monkeypatch.setattr(jobs.engines_probe, "probe_engines", lambda: _probe(rocm=True))
    monkeypatch.setattr(
        jobs,
        "build_engine",
        lambda engine, model, language: _StallEngine(2.5) if engine == "rocm" else _SpeechEngine(),
    )
    job_id = _upload(client, "rocm")
    data = wait_done(client, job_id, timeout=30)
    assert data["status"] == "done", data
    assert data["engine_fallback"]["from"] == "rocm"
    assert "看门狗" in data["engine_fallback"]["reason"]
    assert data["result"]["engine_used"] == "cpu"


# ---------------------------------------------------------------- GPU 回退一致性


def _result_of(client, job_id):
    resp = client.get(f"/api/jobs/{job_id}/result")
    assert resp.status_code == 200
    return resp.json()


@pytest.mark.parametrize("gpu", ["cuda", "rocm"])
def test_qa_gpu_fallback_result_identical_to_cpu(client, monkeypatch, gpu):
    monkeypatch.setattr(
        jobs.engines_probe, "probe_engines", lambda: _probe(detail=f"{gpu} 不可用（模拟）")
    )
    monkeypatch.setattr(jobs, "build_engine", lambda engine, model, language: _SpeechEngine())
    gpu_id = _upload(client, gpu)
    cpu_id = _upload(client, "cpu")
    gpu_data = wait_done(client, gpu_id)
    cpu_data = wait_done(client, cpu_id)
    assert gpu_data["status"] == "done", gpu_data
    assert cpu_data["status"] == "done", cpu_data
    assert gpu_data["engine_fallback"] == {
        "from": gpu,
        "to": "cpu",
        "reason": f"{gpu} 不可用（模拟）",
    }
    assert gpu_data["result"]["engine_used"] == "cpu"
    assert any("回退 CPU" in w for w in gpu_data["warnings"])
    gr, cr = _result_of(client, gpu_id), _result_of(client, cpu_id)
    # 同音频夹具：回退后的分析/重点/转写与直接 CPU 任务完全一致
    for key in ("text", "segments", "analysis", "keypoints", "keypoint_mode", "duration"):
        assert gr[key] == cr[key], f"回退后 {key} 与 CPU 任务不一致"
    assert gr["analysis"]["total_tokens"] > 0
    assert gr["keypoints"]


def test_qa_gpu_build_engine_failure_falls_back(client, monkeypatch):
    monkeypatch.setattr(jobs.engines_probe, "probe_engines", lambda: _probe(cuda=True))

    def build(engine, model, language):
        if engine == "cuda":
            raise RuntimeError("CUDA 模型加载失败（模拟）")
        return _SpeechEngine()

    monkeypatch.setattr(jobs, "build_engine", build)
    job_id = _upload(client, "cuda")
    data = wait_done(client, job_id)
    assert data["status"] == "done", data
    assert data["engine_fallback"]["from"] == "cuda"
    assert "模型加载失败" in data["engine_fallback"]["reason"]
    assert data["result"]["engine_used"] == "cpu"
    assert data["result"]["keypoints"]


# ---------------------------------------------------------------- LLM


class _Resp:
    def __init__(self, content, status_code=200):
        self._content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(f"HTTP {self.status_code}", request=None, response=None)

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def _enable_llm(monkeypatch, **overrides):
    cfg = {"enabled": True, "base_url": "http://127.0.0.1:9/v1", "key": "k", "model": "m"}
    cfg.update(overrides)
    monkeypatch.setattr(keypoints, "load_settings", lambda: {**DEFAULT_SETTINGS, "llm": cfg})


@pytest.fixture()
def speech_segments():
    return build_speech_segments(FIXTURE_SENTENCES)


def test_qa_llm_success_sorted_and_dedup(monkeypatch, speech_segments):
    _enable_llm(monkeypatch)
    content = json.dumps(
        [
            {"point": "预算 超支", "importance": 3, "timestamp_start": 1, "timestamp_end": 2},
            {"point": "预算超支", "importance": 5, "timestamp_start": 3, "timestamp_end": 4},
            {"point": "上线延期", "importance": 4, "timestamp_start": 5, "timestamp_end": 6},
        ],
        ensure_ascii=False,
    )
    monkeypatch.setattr(keypoints.httpx, "post", lambda *a, **k: _Resp(content))
    report = keypoints.extract_keypoints_detailed(speech_segments, mode="llm")
    assert report["keypoint_mode"] == "llm"
    assert report["warnings"] == []
    points = report["keypoints"]
    # 空白差异归一化去重，保留高 importance；整体降序
    assert [p["point"] for p in points] == ["预算超支", "上线延期"]
    assert [p["importance"] for p in points] == [5, 4]
    assert points[0]["start"] == 3.0


def test_qa_llm_non_json_falls_back_offline(monkeypatch, speech_segments):
    _enable_llm(monkeypatch)
    monkeypatch.setattr(keypoints.httpx, "post", lambda *a, **k: _Resp("抱歉，我无法回答。"))
    report = keypoints.extract_keypoints_detailed(speech_segments, mode="llm")
    assert report["keypoint_mode"] == "offline"
    assert any("LLM 失败已回退离线" in w for w in report["warnings"])
    assert report["keypoints"] == keypoints.extract_keypoints(speech_segments)


def test_qa_llm_json_object_not_array_falls_back(monkeypatch, speech_segments):
    _enable_llm(monkeypatch)
    monkeypatch.setattr(
        keypoints.httpx, "post", lambda *a, **k: _Resp('{"point": "不是数组"}')
    )
    report = keypoints.extract_keypoints_detailed(speech_segments, mode="llm")
    assert report["keypoint_mode"] == "offline"
    assert report["warnings"]


def test_qa_llm_timeout_falls_back_offline(monkeypatch, speech_segments):
    _enable_llm(monkeypatch)

    def fake_post(*a, **k):
        raise httpx.ReadTimeout("read timed out")

    monkeypatch.setattr(keypoints.httpx, "post", fake_post)
    report = keypoints.extract_keypoints_detailed(speech_segments, mode="llm")
    assert report["keypoint_mode"] == "offline"
    assert any("LLM 失败已回退离线" in w for w in report["warnings"])
    assert report["keypoints"]


def test_qa_llm_disabled_zero_http(monkeypatch, speech_segments):
    monkeypatch.setattr(keypoints, "load_settings", lambda: dict(DEFAULT_SETTINGS))

    def no_http(*a, **k):
        raise AssertionError("enabled=false 时不得发起任何 HTTP 请求")

    monkeypatch.setattr(keypoints.httpx, "post", no_http)
    report = keypoints.extract_keypoints_detailed(speech_segments, mode="llm")
    assert report["keypoint_mode"] == "offline"
    assert report["warnings"] == []
    assert report["keypoints"]


def test_qa_llm_enabled_but_unconfigured_zero_http(monkeypatch, speech_segments):
    # 绕过 settings 校验的脏配置：enabled=true 但 base_url 为空 → 不得发请求
    _enable_llm(monkeypatch, base_url="")

    def no_http(*a, **k):
        raise AssertionError("base_url 为空时不得发起 HTTP 请求")

    monkeypatch.setattr(keypoints.httpx, "post", no_http)
    report = keypoints.extract_keypoints_detailed(speech_segments, mode="llm")
    assert report["keypoint_mode"] == "offline"
    assert report["warnings"] == []


def test_qa_llm_long_transcript_chunked_multiple_requests(monkeypatch):
    _enable_llm(monkeypatch)
    # 6 段 × 1000 字 = 6000 字 > 3000 → 必须分块 ≥2 次请求
    segments = [
        {"start": float(i * 60), "end": float(i * 60 + 50), "text": f"段落{i}：" + "内容" * 500, "words": []}
        for i in range(6)
    ]
    assert sum(len(s["text"]) for s in segments) > keypoints.LLM_CHUNK_CHARS
    calls = []

    def fake_post(url, **kwargs):
        calls.append(kwargs["json"]["messages"][-1]["content"])
        idx = len(calls)
        return _Resp(
            json.dumps(
                [{"point": f"重点{idx}", "importance": 3, "timestamp_start": 0, "timestamp_end": 1}],
                ensure_ascii=False,
            )
        )

    monkeypatch.setattr(keypoints.httpx, "post", fake_post)
    report = keypoints.extract_keypoints_detailed(segments, mode="llm")
    assert len(calls) >= 2
    assert report["keypoint_mode"] == "llm"
    assert len(report["keypoints"]) == len(calls)
    chunks = [c[len(keypoints._LLM_PROMPT):] for c in calls]
    assert all(len(c) <= keypoints.LLM_CHUNK_CHARS for c in chunks)
    # 每段转写恰好出现在一个块中（不丢不重）
    for seg in segments:
        assert sum(seg["text"] in c for c in chunks) == 1


def test_qa_job_llm_success_end_to_end(client, monkeypatch):
    assert (
        client.put(
            "/api/settings",
            json={"llm": {"enabled": True, "base_url": "http://127.0.0.1:9/v1", "key": "k", "model": "m"}},
        ).status_code
        == 200
    )
    content = json.dumps(
        [
            {"point": "LLM 重点甲", "importance": 5, "timestamp_start": 0, "timestamp_end": 2},
            {"point": "LLM 重点乙", "importance": 3, "timestamp_start": 2, "timestamp_end": 4},
        ],
        ensure_ascii=False,
    )
    monkeypatch.setattr(keypoints.httpx, "post", lambda *a, **k: _Resp(content))
    job_id = _upload(client, "cpu")
    data = wait_done(client, job_id)
    assert data["status"] == "done", data
    result = data["result"]
    assert result["keypoint_mode"] == "llm"
    assert [p["point"] for p in result["keypoints"]] == ["LLM 重点甲", "LLM 重点乙"]
    assert result["warnings"] == []
    assert result["engine_fallback"] is None


def test_qa_job_llm_timeout_warning_and_offline(client, monkeypatch):
    assert (
        client.put(
            "/api/settings",
            json={"llm": {"enabled": True, "base_url": "http://127.0.0.1:9/v1", "key": "k", "model": "m"}},
        ).status_code
        == 200
    )

    def fake_post(*a, **k):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(keypoints.httpx, "post", fake_post)
    monkeypatch.setattr(jobs, "build_engine", lambda engine, model, language: _SpeechEngine())
    job_id = _upload(client, "cpu")
    data = wait_done(client, job_id)
    assert data["status"] == "done", data
    assert data["result"]["keypoint_mode"] == "offline"
    assert any("LLM 失败已回退离线" in w for w in data["warnings"])
    assert data["result"]["keypoints"]


# ---------------------------------------------------------------- settings


def test_qa_llm_enabled_missing_base_url_400(client):
    assert client.put("/api/settings", json={"llm": {"enabled": True, "model": "m"}}).status_code == 400
    assert client.put("/api/settings", json={"llm": {"enabled": True, "base_url": "http://x/v1"}}).status_code == 400


@pytest.mark.parametrize("bad", [-1, -0.5, 0, "600", "abc", None, True, [], {}])
def test_qa_watchdog_seconds_invalid_400(client, bad):
    resp = client.put("/api/settings", json={"watchdog_seconds": bad})
    assert resp.status_code == 400, bad


def test_qa_settings_put_roundtrip(client):
    payload = {
        "default_engine": "cpu",
        "default_model": "small",
        "language": "zh",
        "watchdog_seconds": 12.5,
        "llm": {"enabled": True, "base_url": "http://127.0.0.1:9/v1", "key": "k", "model": "m"},
        "filler_words": {"pure": ["嗯", "呃"], "contextual": ["然后", "就是"]},
    }
    assert client.put("/api/settings", json=payload).status_code == 200
    assert client.get("/api/settings").json() == payload
    # 落盘往返一致
    on_disk = json.loads(settings.SETTINGS_FILE.read_text(encoding="utf-8"))
    assert on_disk == payload
    # 部分更新不丢其余字段
    assert client.put("/api/settings", json={"watchdog_seconds": 30}).status_code == 200
    got = client.get("/api/settings").json()
    assert got["watchdog_seconds"] == 30
    assert got["llm"] == payload["llm"]
    assert got["filler_words"] == payload["filler_words"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_qa_watchdog_seconds_nan_inf_should_400(tmp_path, monkeypatch, bad):
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(settings, "SETTINGS_FILE", tmp_path / "data" / "settings.json")
    with TestClient(app, raise_server_exceptions=False) as c:
        resp = c.put(
            "/api/settings",
            content=json.dumps({"watchdog_seconds": bad}),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400