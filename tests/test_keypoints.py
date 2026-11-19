import jieba
import pytest

from app import keypoints


def build_speech_segments(sentences, start=0.0, step=0.2, gap=0.0):
    segments = []
    t = start
    for text in sentences:
        words = []
        for tok in jieba.lcut(text):
            if not tok.strip():
                continue
            words.append({"word": tok, "start": t, "end": t + step, "probability": 0.9})
            t += step
        if not words:
            continue
        segments.append(
            {"start": words[0]["start"], "end": words[-1]["end"],
             "text": "".join(w["word"] for w in words), "words": words}
        )
        t += gap
    return segments


KEY_SENTENCE = "项目进度上传模块转写引擎统计功能整体安排确认。"
FIXTURE_SENTENCES = [
    "项目进度会议现在开始。",
    "上传模块开发完成。",
    "转写引擎兼容问题。",
    "统计功能下周上线。",
    KEY_SENTENCE,
    "天气不错适合散步。",
]


def test_key_sentence_ranks_first():
    segments = build_speech_segments(FIXTURE_SENTENCES)
    points = keypoints.extract_keypoints(segments)
    assert points
    assert points[0]["point"] == KEY_SENTENCE.rstrip("。")
    assert points[0]["importance"] == 5


def test_filler_short_sentences_filtered():
    sentences = [
        "嗯，这个这个。",
        "然后，就是。",
        "上周我们完成了上传模块的开发。",
        "本周先把语音转写做完。",
    ]
    segments = build_speech_segments(sentences, gap=0.6)
    points = keypoints.extract_keypoints(segments)
    texts = [p["point"] for p in points]
    assert "嗯，这个这个" not in texts
    assert "然后，就是" not in texts
    assert any("上传模块" in t for t in texts)
    assert any("语音转写" in t for t in texts)


def test_position_weight_head_beats_middle():
    twin = "项目进度安排确认。"
    sentences = [
        twin,
        "上传模块开发完成。",
        "转写引擎兼容问题。",
        "统计功能下周上线。",
        twin,
        "测试录音需要补充。",
        "会议记录整理归档。",
        "接口文档同步更新。",
        "部署流程演练一遍。",
    ]
    segments = build_speech_segments(sentences)
    points = keypoints.extract_keypoints(segments)
    twins = [p for p in points if p["point"] == twin.rstrip("。")]
    assert len(twins) == 2
    head_start = segments[0]["start"]
    mid_start = segments[4]["start"]
    assert twins[0]["start"] == pytest.approx(head_start, abs=1e-3)
    assert twins[1]["start"] == pytest.approx(mid_start, abs=1e-3)
    assert twins[0]["importance"] >= twins[1]["importance"]


def test_top_n_cap_long_audio():
    sentences = [f"模块{10 + i}的开发进度确认。" for i in range(30)]
    segments = build_speech_segments(sentences, step=1.0)
    points = keypoints.extract_keypoints(segments)
    assert 8 <= len(points) <= 15


def test_short_audio_fewer_than_eight():
    segments = build_speech_segments(["上传模块开发完成。", "转写引擎兼容问题。"])
    points = keypoints.extract_keypoints(segments)
    assert 0 < len(points) <= 2


def test_empty_transcript_returns_empty():
    assert keypoints.extract_keypoints([]) == []
    assert keypoints.extract_keypoints([{"start": 0, "end": 1, "text": "", "words": []}]) == []
    assert keypoints.extract_keypoints([{"start": 0, "end": 1, "text": "，。！", "words": []}]) == []


def test_all_filler_returns_empty():
    segments = build_speech_segments(["嗯。", "啊。", "呃。", "这个这个。", "就是就是。"])
    assert keypoints.extract_keypoints(segments) == []


def test_output_schema_and_order():
    segments = build_speech_segments(FIXTURE_SENTENCES)
    points = keypoints.extract_keypoints(segments)
    assert points
    importances = []
    for p in points:
        assert set(p) == {"point", "importance", "start", "end"}
        assert isinstance(p["point"], str) and p["point"]
        assert isinstance(p["importance"], int) and 1 <= p["importance"] <= 5
        assert 0 <= p["start"] < p["end"]
        importances.append(p["importance"])
    assert importances == sorted(importances, reverse=True)


def test_timestamps_match_source_sentence():
    segments = build_speech_segments(FIXTURE_SENTENCES)
    points = keypoints.extract_keypoints(segments)
    key = next(p for p in points if p["point"] == KEY_SENTENCE.rstrip("。"))
    src = segments[4]
    assert key["start"] == pytest.approx(src["start"], abs=1e-3)
    assert key["end"] == pytest.approx(src["end"], abs=1e-3)


def test_segment_boundary_splits_sentences():
    segments = build_speech_segments(["上传模块开发完成", "转写引擎兼容问题"])
    points = keypoints.extract_keypoints(segments)
    assert len(points) == 2
    assert {p["point"] for p in points} == {"上传模块开发完成", "转写引擎兼容问题"}


def test_punctuation_inside_segment_splits():
    words = [
        {"word": "上传模块。", "start": 0.0, "end": 1.0, "probability": 0.9},
        {"word": "开发完成。", "start": 1.0, "end": 2.0, "probability": 0.9},
    ]
    segments = [{"start": 0.0, "end": 2.0, "text": "上传模块。开发完成。", "words": words}]
    points = keypoints.extract_keypoints(segments)
    assert {p["point"] for p in points} == {"上传模块", "开发完成"}


def test_long_sentence_split_at_commas():
    text = (
        "今天我们这个会议主要讨论一下项目进度,然后呢,"
        "就是先回顾一下上周的工作,上周我们完成了上传模块的开发"
    )
    words = [
        {"word": ch, "start": i * 0.1, "end": (i + 1) * 0.1, "probability": 0.9}
        for i, ch in enumerate(text)
    ]
    segments = [{"start": 0.0, "end": len(text) * 0.1, "text": text, "words": words}]
    points = keypoints.extract_keypoints(segments)
    assert any(p["point"] == "上周我们完成了上传模块的开发" for p in points)
    for p in points:
        assert len(p["point"]) <= keypoints.MAX_SENTENCE_CHARS


def test_llm_mode_unconfigured_uses_offline(monkeypatch):
    from app.settings import DEFAULT_SETTINGS

    monkeypatch.setattr(keypoints, "load_settings", lambda: dict(DEFAULT_SETTINGS))
    segments = build_speech_segments(FIXTURE_SENTENCES)
    report = keypoints.extract_keypoints_detailed(segments, mode="llm")
    assert report["keypoint_mode"] == "offline"
    assert report["warnings"] == []
    assert report["keypoints"] == keypoints.extract_keypoints(segments)


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        keypoints.extract_keypoints([], mode="magic")