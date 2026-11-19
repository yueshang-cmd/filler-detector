"""M4 回归验收 — QA 对抗夹具（keypoints 边界 + Range 边界）。

只补测试，不改业务逻辑。复用 tests.test_keypoints 的造句工具与
tests.test_audio_range 的 audio_job 夹具。
"""
import pytest

from app import keypoints
from tests.test_keypoints import build_speech_segments
from tests.test_audio_range import audio_job  # noqa: F401  (pytest fixture)


# ---------------------------------------------------------------- keypoints


def test_all_filler_adversarial_returns_empty():
    """各种形态的全废话输入都必须返回 []，不崩、不产出。"""
    sentences = [
        "嗯。", "啊。", "呃。", "额。", "唔。",
        "这个这个。", "就是就是。", "然后然后。", "其实其实。",
        "嗯，啊，呃。",
        "那个，那个，那个。",
        "嗯嗯嗯嗯。",
        "就是说，怎么说呢，对吧。",
    ]
    segments = build_speech_segments(sentences, gap=0.6)
    assert keypoints.extract_keypoints(segments) == []


def test_all_filler_text_only_segments_returns_empty():
    """无词级时间戳（words 缺失）的纯文本全废话段同样返回 []。"""
    segments = [
        {"start": 0.0, "end": 2.0, "text": "嗯，啊，呃。", "words": []},
        {"start": 2.0, "end": 4.0, "text": "这个这个，就是就是。", "words": []},
        {"start": 4.0, "end": 5.0, "text": "。。。", "words": []},
    ]
    assert keypoints.extract_keypoints(segments) == []


def test_single_sentence_does_not_crash():
    segments = build_speech_segments(["今天会议讨论项目进度安排。"])
    points = keypoints.extract_keypoints(segments)
    assert isinstance(points, list)
    assert len(points) == 1
    p = points[0]
    assert p["point"] == "今天会议讨论项目进度安排"
    assert 1 <= p["importance"] <= 5
    assert p["start"] == pytest.approx(segments[0]["start"], abs=1e-3)
    assert p["end"] == pytest.approx(segments[0]["end"], abs=1e-3)


def test_single_filler_sentence_returns_empty():
    assert keypoints.extract_keypoints(build_speech_segments(["嗯。"])) == []
    assert keypoints.extract_keypoints(build_speech_segments(["这个这个。"])) == []


def test_long_text_200_sentences_cap_and_order():
    """200 句长大文本：数量 ≤15（长音频应取满 15），importance 降序且 1-5。"""
    sentences = [
        f"议题{i}关于模块{i % 20}的开发进度与测试安排讨论确认。"
        for i in range(200)
    ]
    segments = build_speech_segments(sentences, step=0.5)
    points = keypoints.extract_keypoints(segments)
    assert 8 <= len(points) <= 15
    assert len(points) == 15  # span 远超 120s，应取满上限
    importances = [p["importance"] for p in points]
    assert importances == sorted(importances, reverse=True)
    assert all(isinstance(v, int) and 1 <= v <= 5 for v in importances)
    # 无重复句、时间戳单调来源可追溯
    assert len({p["point"] for p in points}) == len(points)


def test_position_weight_unit():
    n = 11
    assert keypoints._position_weight(0, n) == keypoints.POSITION_BOOST
    assert keypoints._position_weight(n - 1, n) == keypoints.POSITION_BOOST
    assert keypoints._position_weight(n // 2, n) == 1.0
    assert keypoints._position_weight(0, 1) == keypoints.POSITION_BOOST


def test_position_weight_head_tail_beat_identical_middle():
    """首/中/尾放同一句（TextRank/覆盖/密度全同），仅位置权重不同：
    首尾必须排在中间之前，且重要性不低于中间。"""
    twin = "项目进度安排确认。"
    sentences = [
        twin,                      # idx 0  头部
        "上传模块开发完成。",
        "转写引擎兼容问题。",
        "统计功能下周上线。",
        "测试录音需要补充。",
        twin,                      # idx 5  正中（p=0.5，无加权）
        "会议记录整理归档。",
        "接口文档同步更新。",
        "部署流程演练一遍。",
        "客户反馈汇总分析。",
        twin,                      # idx 10 尾部
    ]
    segments = build_speech_segments(sentences)
    points = keypoints.extract_keypoints(segments)
    twin_text = twin.rstrip("。")
    twins = [p for p in points if p["point"] == twin_text]
    assert len(twins) == 3, f"三个孪生句都应入选，实际 {len(twins)}"

    def at(seg_idx):
        return next(
            p for p in twins
            if abs(p["start"] - segments[seg_idx]["start"]) < 1e-3
        )

    head, mid, tail = at(0), at(5), at(10)
    assert head["importance"] >= mid["importance"]
    assert tail["importance"] >= mid["importance"]
    # 头部孪生句应为全场最高分（importance=5），中间句严格更低
    assert head["importance"] == 5
    assert mid["importance"] < head["importance"]
    # 列表按得分降序：首/尾（加权）必须排在中间（无加权）之前
    order = [p["start"] for p in twins]
    assert order.index(head["start"]) < order.index(mid["start"])
    assert order.index(tail["start"]) < order.index(mid["start"])


def test_importance_range_on_mixed_corpus():
    sentences = [
        "嗯，这个这个。",
        "项目进度会议现在开始。",
        "上传模块开发完成。",
        "转写引擎兼容问题。",
        "统计功能下周上线。",
        "然后，就是。",
        "项目进度上传模块转写引擎统计功能整体安排确认。",
        "天气不错适合散步。",
    ]
    points = keypoints.extract_keypoints(build_speech_segments(sentences, gap=0.6))
    assert points
    for p in points:
        assert isinstance(p["importance"], int)
        assert 1 <= p["importance"] <= 5


def test_timestamps_align_with_source_sentences():
    """每条重点的 start/end 必须与某个源句（segment）精确对齐。"""
    sentences = [
        "项目进度会议现在开始。",
        "上传模块开发完成。",
        "转写引擎兼容问题。",
        "统计功能下周上线。",
        "项目进度上传模块转写引擎统计功能整体安排确认。",
        "天气不错适合散步。",
    ]
    segments = build_speech_segments(sentences)
    points = keypoints.extract_keypoints(segments)
    assert points
    for p in points:
        src = next(
            (s for s in segments if abs(s["start"] - p["start"]) < 1e-3), None
        )
        assert src is not None, f"重点 start={p['start']} 找不到源句"
        assert p["end"] == pytest.approx(src["end"], abs=1e-3)
        assert src["text"].strip("。") == p["point"]
        assert 0 <= p["start"] < p["end"]


# ---------------------------------------------------------------- Range 边界


def test_range_single_first_byte(audio_job):  # noqa: F811
    client, job_id, data = audio_job
    resp = client.get(f"/api/audio/{job_id}", headers={"Range": "bytes=0-0"})
    assert resp.status_code == 206
    assert resp.content == data[:1]
    assert len(resp.content) == 1
    assert resp.headers["content-range"] == f"bytes 0-0/{len(data)}"
    assert int(resp.headers["content-length"]) == 1


def test_range_suffix_last_100(audio_job):  # noqa: F811
    client, job_id, data = audio_job
    resp = client.get(f"/api/audio/{job_id}", headers={"Range": "bytes=-100"})
    assert resp.status_code == 206
    assert resp.content == data[-100:]
    assert resp.headers["content-range"] == (
        f"bytes {len(data) - 100}-{len(data) - 1}/{len(data)}"
    )


def test_range_start_beyond_size_416(audio_job):  # noqa: F811
    client, job_id, data = audio_job
    resp = client.get(
        f"/api/audio/{job_id}", headers={"Range": f"bytes={len(data) + 1}-"}
    )
    assert resp.status_code == 416
    assert resp.headers["content-range"] == f"bytes */{len(data)}"


def test_multi_range_returns_full_200_no_crash(audio_job):  # noqa: F811
    client, job_id, data = audio_job
    resp = client.get(
        f"/api/audio/{job_id}", headers={"Range": "bytes=0-99,200-299,500-599"}
    )
    assert resp.status_code == 200
    assert resp.content == data
    assert "content-range" not in resp.headers


def test_audio_unknown_job_404(client):
    resp = client.get("/api/audio/does-not-exist", headers={"Range": "bytes=0-0"})
    assert resp.status_code == 404