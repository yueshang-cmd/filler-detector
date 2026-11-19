"""M3 QA 回归：废话误判边界对抗用例（独立于实现者测试）。

直接调用 app.filler 公开函数：analyze / count_tokens / filter_hallucination_segments。
原 xfail 固化的 4 个漏计 bug 已修复，对应用例转为正常断言：
  BUG-M3-1 叠词识别任意 >=2 倍连续重复（"嗯嗯嗯嗯"/"这个这个这个这个"）
  BUG-M3-2 总词数各段 text 以空格拼接，跨段英文单词不再被合并少计
  BUG-M3-3 跨段停顿取下一有效词元 start，跳过纯标点段不再低估漏判
  BUG-M3-4 归一化做 NFKC，全角英文（ＵＭ）计入
"""

import pytest

from app import filler


def wd(word, start, end):
    return {"word": word, "start": start, "end": end}


def seg(words, text=None, start=None, end=None):
    return {
        "start": words[0]["start"] if start is None else start,
        "end": words[-1]["end"] if end is None else end,
        "text": "".join(w["word"] for w in words) if text is None else text,
        "words": words,
    }


def words_of(analysis):
    return [f["word"] for f in analysis["fillers"]]


# ---------- 语境词正常用法不得误伤 ----------

def test_ranhou_single_normal_continuation_not_counted():
    # 非句首、词后停顿 0.1s < 0.5、全篇仅 1 次、无重复无逗号
    segments = [seg([
        wd("我们", 0.0, 0.4), wd("然后", 0.4, 0.7), wd("开始", 0.8, 1.2),
        wd("讨论", 1.2, 1.6), wd("进度", 1.6, 2.0),
    ])]
    assert filler.analyze(segments)["fillers"] == []


def test_trailing_contextual_without_following_speech_not_counted():
    # 末尾语境词，其后无任何语音 → 无停顿证据，不计
    segments = [seg([wd("我们", 0.0, 0.6), wd("然后", 0.6, 1.2)])]
    assert filler.analyze(segments)["fillers"] == []


# ---------- 规则 2：停顿阈值边界 ----------

def test_pause_exactly_050_counted():
    segments = [seg([wd("我们", 0.0, 0.4), wd("就是", 0.4, 0.8), wd("测试", 1.3, 1.7)])]
    assert words_of(filler.analyze(segments)) == ["就是"]


def test_pause_0499_not_counted():
    segments = [seg([wd("我们", 0.0, 0.4), wd("就是", 0.4, 0.8), wd("测试", 1.299, 1.7)])]
    assert filler.analyze(segments)["fillers"] == []


def test_pause_across_punctuation_only_segment():
    s1 = seg([wd("我们", 0.0, 0.4), wd("就是", 0.4, 0.8)])
    s2 = {"start": 0.8, "end": 0.85, "text": "，", "words": [wd("，", 0.8, 0.85)]}
    s3 = seg([wd("测试", 1.4, 1.8)])  # 真实停顿 1.4-0.8=0.6s ≥0.5 应计
    assert words_of(filler.analyze([s1, s2, s3])) == ["就是"]


# ---------- 规则 4：30s 窗口边界 ----------

def _window_segments(starts):
    words = []
    for s in starts:
        words += [wd("我们", s - 0.4, s), wd("然后", s, s + 0.4), wd("继续", s + 0.4, s + 0.8)]
    return [seg(words, start=0.0, end=starts[-1] + 0.8)]


def test_window_span_exactly_30_all_counted():
    analysis = filler.analyze(_window_segments([0.4, 15.4, 30.4]))
    assert words_of(analysis) == ["然后"] * 3
    assert [f["start"] for f in analysis["fillers"]] == [0.4, 15.4, 30.4]


def test_window_third_at_3001_not_counted():
    # 仅 3 次出现，第 3 次距第 1 次 30.01s → 任何 30s 窗口都只有 2 次
    assert filler.analyze(_window_segments([0.4, 15.4, 30.41]))["fillers"] == []


def test_window_two_occurrences_never_counted():
    assert filler.analyze(_window_segments([0.4, 29.0]))["fillers"] == []


def test_window_sliding_fourth_occurrence_forms_new_window():
    # 语义记录：滑动窗口，第 4 次(30.41)与第 2/3 次构成新 30s 窗口 → 也计
    analysis = filler.analyze(_window_segments([0.4, 10.4, 20.4, 30.41]))
    assert words_of(analysis) == ["然后"] * 4


def test_window_different_words_not_pooled():
    words = []
    for i, w in enumerate(["然后", "其实", "反正"]):
        s = 0.4 + i * 10
        words += [wd("我们", s - 0.4, s), wd(w, s, s + 0.4), wd("继续", s + 0.4, s + 0.8)]
    assert filler.analyze([seg(words, start=0.0, end=21.2)])["fillers"] == []


# ---------- 规则 3：句首 + 逗号/停顿 ----------

def test_sentence_initial_qishi_no_comma_no_pause_not_counted():
    segments = [seg([wd("其实", 0.0, 0.4), wd("我们", 0.4, 0.8), wd("继续", 0.8, 1.2)])]
    assert filler.analyze(segments)["fillers"] == []


def test_sentence_initial_qishi_with_comma_counted():
    segments = [seg([wd("其实，", 0.0, 0.4), wd("我们", 0.4, 0.8), wd("继续", 0.8, 1.2)])]
    assert words_of(filler.analyze(segments)) == ["其实"]


def test_sentence_initial_comma_separate_token_counted():
    segments = [seg([wd("其实", 0.0, 0.4), wd("，", 0.4, 0.45), wd("我们", 0.45, 0.9)])]
    assert words_of(filler.analyze(segments)) == ["其实"]


def test_sentence_initial_skips_leading_punctuation_token():
    segments = [seg([wd("。", 0.0, 0.05), wd("其实", 0.05, 0.4), wd("，", 0.4, 0.45), wd("好", 0.45, 0.8)])]
    assert words_of(filler.analyze(segments)) == ["其实"]


def test_sentence_initial_is_segment_level_semantics():
    # 语义记录：句首=段首（首个有效词元）。同段内句号后的"其实"不按句首处理；
    # PLAN 未定义段内多句情形，ASR 段通常≈句，此处固化现状供规格澄清。
    segments = [seg([wd("好。", 0.0, 0.4), wd("其实", 0.4, 0.8), wd("我们", 0.8, 1.2)])]
    assert filler.analyze(segments)["fillers"] == []


# ---------- 规则 1：叠词 / 归一化 / 多词短语 ----------

def test_reduplicated_jiushi_single_token_counted_once():
    segments = [seg([wd("就是就是", 0.0, 0.8), wd("这样", 0.8, 1.2)])]
    analysis = filler.analyze(segments)
    assert words_of(analysis) == ["就是"]
    assert (analysis["fillers"][0]["start"], analysis["fillers"][0]["end"]) == (0.0, 0.8)


def test_um_uppercase_with_comma_counted():
    segments = [seg([wd("UM,", 0.0, 0.3), wd("好", 0.3, 0.6)])]
    assert words_of(filler.analyze(segments)) == ["um"]


def test_you_know_split_tokens_counted_once():
    segments = [seg([wd("well", 0.0, 0.3), wd("you", 0.3, 0.6), wd(" know", 0.6, 0.9), wd(" test", 1.4, 1.8)])]
    analysis = filler.analyze(segments)
    assert words_of(analysis) == ["you know"]
    assert (analysis["fillers"][0]["start"], analysis["fillers"][0]["end"]) == (0.3, 0.9)


def test_you_know_normal_use_not_counted():
    segments = [seg([wd("you", 0.0, 0.3), wd(" know", 0.3, 0.6), wd(" the", 0.6, 0.9), wd(" answer", 0.9, 1.3)])]
    assert filler.analyze(segments)["fillers"] == []


def test_phrase_does_not_cross_segments():
    s1 = seg([wd("you", 0.0, 0.3)])
    s2 = seg([wd(" know", 0.3, 0.6), wd(" test", 1.2, 1.6)])
    assert filler.analyze([s1, s2])["fillers"] == []


# ---------- 纯语气词：独立词元才计，不误伤 ----------

def test_pure_embedded_in_normal_token_not_counted():
    segments = [seg([wd("嗯好", 0.0, 0.4), wd("继续", 0.4, 0.8)])]
    assert filler.analyze(segments)["fillers"] == []


def test_pure_reduplicated_two_form_counted_once():
    segments = [seg([wd("嗯嗯", 0.0, 0.8), wd("好", 0.8, 1.2)])]
    assert words_of(filler.analyze(segments)) == ["嗯"]


def test_pure_reduplicated_four_form_should_count():
    segments = [seg([wd("嗯嗯嗯嗯", 0.0, 0.8), wd("好", 0.8, 1.2)])]
    assert words_of(filler.analyze(segments)) == ["嗯"]


def test_contextual_reduplicated_four_form_should_count():
    segments = [seg([wd("这个这个这个这个", 0.0, 0.8), wd("方案", 0.8, 1.2)])]
    assert words_of(filler.analyze(segments)) == ["这个"]


def test_fullwidth_um_should_count():
    segments = [seg([wd("ＵＭ", 0.0, 0.3), wd("好", 0.3, 0.6)])]
    assert words_of(filler.analyze(segments)) == ["um"]


# ---------- 空转写 / 无 words ----------

def test_empty_transcript_all_zero():
    a = filler.analyze([])
    assert a["total_tokens"] == 0 and a["filler_count"] == 0
    assert a["filler_count_ratio"] == 0.0 and a["filler_duration_ratio"] == 0.0
    assert a["top10"] == [] and a["fillers"] == []


def test_segment_words_none_or_missing_no_crash():
    for s in (
        {"start": 0.0, "end": 2.0, "text": "嗯你好", "words": None},
        {"start": 0.0, "end": 2.0, "text": "嗯你好"},
    ):
        a = filler.analyze([s])
        assert a["filler_count"] == 0  # 无词级时间戳无法判定，合理
        assert a["total_tokens"] == 2  # 文本仍计入总词数
        assert a["total_speech_duration"] == 2.0


# ---------- 4.2 统计口径 ----------

@pytest.mark.parametrize("text,expected", [
    ("我们今天讨论project进度", 5),
    ("hello world 你好", 3),
    ("嗯这个这个方案不错", 5),
    ("", 0),
    ("，。！", 0),
])
def test_count_tokens_mixed_manual_expectation(text, expected):
    assert filler.count_tokens(text) == expected


def test_total_tokens_cross_segment_english_not_merged():
    segs = [seg([wd("hello", 0.0, 0.5)]), seg([wd("world", 1.0, 1.5)])]
    assert filler.analyze(segs)["total_tokens"] == 2


def test_duration_denominator_is_speech_not_audio_span():
    # 两段语音共 4s，中间静音 48s；分母必须是 4 而不是 52
    s1 = seg([wd("嗯", 0.0, 0.5), wd("内容", 0.5, 2.0)])
    s2 = seg([wd("继续", 50.0, 52.0)])
    a = filler.analyze([s1, s2])
    assert a["total_speech_duration"] == 4.0
    assert a["filler_duration"] == 0.5
    assert a["filler_duration_ratio"] == 0.125


def test_top10_merge_same_word_across_segments():
    s1 = seg([wd("嗯", 0.0, 0.5), wd("好", 0.5, 1.0)])
    s2 = seg([wd("嗯", 2.0, 2.5), wd("啊", 2.5, 3.0), wd("收尾", 3.0, 4.0)])
    a = filler.analyze([s1, s2])
    top = {g["word"]: g for g in a["top10"]}
    assert top["嗯"]["count"] == 2 and top["嗯"]["duration"] == 1.0
    assert top["啊"]["count"] == 1
    assert [g["word"] for g in a["top10"]][0] == "嗯"


# ---------- 幻觉段过滤 ----------

def _bare(text, start, end):
    return {"start": start, "end": end, "text": text, "words": []}


def test_hallucination_four_identical_collapsed_to_first():
    segs = [_bare("重复幻觉", i * 2.0, i * 2.0 + 2.0) for i in range(4)]
    segs.append(_bare("正常内容", 8.0, 12.0))
    kept = filler.filter_hallucination_segments(segs)
    # 语义记录：剔除重复、保留首段（与实现者测试一致），而非整段删除
    assert [s["text"] for s in kept] == ["重复幻觉", "正常内容"]
    assert kept[0]["start"] == 0.0


def test_hallucination_distinct_texts_untouched():
    segs = [_bare("第一句", 0.0, 2.0), _bare("第二句", 2.0, 4.0),
            _bare("第三句", 4.0, 6.0), _bare("第四句", 6.0, 8.0)]
    assert filler.filter_hallucination_segments(segs) == segs


def test_hallucination_run_below_ratio_untouched():
    segs = [_bare("好的", 0.0, 0.5), _bare("好的", 0.5, 1.0),
            _bare("好的", 1.0, 1.5), _bare("很长的正常讨论", 1.5, 30.0)]
    assert filler.filter_hallucination_segments(segs) == segs