import pytest

from app import filler


def wd(word, start, end):
    return {"word": word, "start": start, "end": end, "probability": 0.9}


def seg(words, text=None, start=None, end=None):
    return {
        "start": words[0]["start"] if start is None else start,
        "end": words[-1]["end"] if end is None else end,
        "text": "".join(w["word"] for w in words) if text is None else text,
        "words": words,
    }


def words_of(analysis):
    return [f["word"] for f in analysis["fillers"]]


@pytest.mark.parametrize("word", ["嗯", "啊", "呃", "额", "唔", "emmm", "um", "uh", "er"])
def test_pure_words_unconditional(word):
    segments = [seg([wd("我们", 0.0, 0.4), wd(word, 0.4, 0.7), wd("继续", 0.7, 1.2)])]
    analysis = filler.analyze(segments)
    assert words_of(analysis) == [word]
    assert analysis["fillers"][0]["start"] == 0.4
    assert analysis["fillers"][0]["end"] == 0.7
    assert analysis["fillers"][0]["sentence"] == "我们" + word + "继续"


def test_normalization_punctuation_and_case():
    segments = [seg([wd("嗯,", 0.0, 0.3), wd("UM", 0.3, 0.6), wd("呃。", 0.6, 0.9), wd("好", 0.9, 1.2)])]
    analysis = filler.analyze(segments)
    assert words_of(analysis) == ["嗯", "um", "呃"]


def test_punctuation_only_tokens_ignored():
    segments = [seg([wd("，", 0.0, 0.05), wd("测试", 0.05, 0.5), wd("。", 0.5, 0.55)])]
    assert filler.analyze(segments)["fillers"] == []


def test_rule1_repetition_two_tokens():
    segments = [seg([wd("这个", 0.0, 0.4), wd("这个", 0.4, 0.8), wd("方案", 0.8, 1.2), wd("不错", 1.2, 1.6)])]
    analysis = filler.analyze(segments)
    assert words_of(analysis) == ["这个", "这个"]
    assert [f["start"] for f in analysis["fillers"]] == [0.0, 0.4]


def test_rule1_repetition_single_token():
    segments = [seg([wd("这个这个", 0.0, 0.8), wd("方案", 0.8, 1.2), wd("不错", 1.2, 1.6)])]
    analysis = filler.analyze(segments)
    assert words_of(analysis) == ["这个"]
    assert analysis["fillers"][0]["start"] == 0.0
    assert analysis["fillers"][0]["end"] == 0.8


def test_rule1_repetition_jiushi():
    segments = [seg([wd("就是", 0.0, 0.4), wd("就是", 0.4, 0.8), wd("这样", 0.8, 1.2)])]
    assert words_of(filler.analyze(segments)) == ["就是", "就是"]


def test_rule1_negative_single_contextual_normal_use():
    segments = [seg([wd("这个", 0.0, 0.4), wd("方案", 0.4, 0.8), wd("不错", 0.8, 1.2)])]
    assert filler.analyze(segments)["fillers"] == []


def test_rule2_pause_half_second_counts():
    segments = [seg([wd("我们", 0.0, 0.4), wd("就是", 0.4, 0.8), wd("测试", 1.3, 1.7)])]
    assert words_of(filler.analyze(segments)) == ["就是"]


def test_rule2_pause_049_not_counted():
    segments = [seg([wd("我们", 0.0, 0.4), wd("就是", 0.4, 0.8), wd("测试", 1.29, 1.7)])]
    assert filler.analyze(segments)["fillers"] == []


def test_rule2_sentence_end_pause_to_next_segment():
    s1 = seg([wd("我们", 0.0, 0.4), wd("就是", 0.4, 0.8)])
    s2 = seg([wd("测试", 1.3, 1.7)])
    assert words_of(filler.analyze([s1, s2])) == ["就是"]
    s2_short = seg([wd("测试", 1.29, 1.7)])
    assert filler.analyze([s1, s2_short])["fillers"] == []


def test_rule3_sentence_initial_with_comma():
    segments = [seg([wd("然后,", 0.0, 0.4), wd("测试", 0.4, 0.9), wd("跑通", 0.9, 1.3)])]
    assert words_of(filler.analyze(segments)) == ["然后"]


def test_rule3_sentence_initial_comma_as_separate_token():
    segments = [seg([wd("然后", 0.0, 0.4), wd(",", 0.4, 0.45), wd("测试", 0.45, 0.9)])]
    assert words_of(filler.analyze(segments)) == ["然后"]


def test_rule3_sentence_initial_without_comma_not_counted():
    segments = [seg([wd("然后", 0.0, 0.4), wd("测试", 0.4, 0.9), wd("跑通", 0.9, 1.3)])]
    assert filler.analyze(segments)["fillers"] == []


def test_normal_ranhou_continuation_not_counted():
    segments = [seg([wd("然后", 0.0, 0.3), wd("测试", 0.3, 0.8), wd("也", 0.8, 0.9), wd("跑通", 0.9, 1.4), wd("了", 1.4, 1.6)])]
    assert filler.analyze(segments)["fillers"] == []


def _window_segments(starts):
    words = []
    t = 0.0
    for i, s in enumerate(starts):
        words.append(wd("我们", s - 0.4, s))
        words.append(wd("然后", s, s + 0.4))
        words.append(wd("继续", s + 0.4, s + 0.8))
    return [seg(words, start=0.0, end=starts[-1] + 0.8)]


def test_rule4_window_three_occurrences_all_counted():
    analysis = filler.analyze(_window_segments([0.4, 10.4, 20.4]))
    assert words_of(analysis) == ["然后", "然后", "然后"]
    assert [f["start"] for f in analysis["fillers"]] == [0.4, 10.4, 20.4]


def test_rule4_window_two_occurrences_not_counted():
    assert filler.analyze(_window_segments([0.4, 10.4]))["fillers"] == []


def test_rule4_window_spread_over_30s_not_counted():
    assert filler.analyze(_window_segments([0.4, 20.4, 40.4]))["fillers"] == []


def test_rule4_window_exactly_30s_boundary_counted():
    analysis = filler.analyze(_window_segments([0.4, 15.4, 30.4]))
    assert words_of(analysis) == ["然后", "然后", "然后"]


def test_phrase_single_token_with_pause():
    segments = [seg([wd("我们", 0.0, 0.4), wd("就是说", 0.4, 1.0), wd("测试", 1.5, 1.9)])]
    analysis = filler.analyze(segments)
    assert words_of(analysis) == ["就是说"]
    assert analysis["fillers"][0]["end"] == 1.0


def test_phrase_split_tokens_matched_as_one():
    segments = [seg([wd("我们", 0.0, 0.4), wd("就是", 0.4, 0.7), wd("说", 0.7, 1.0), wd("测试", 1.5, 1.9)])]
    analysis = filler.analyze(segments)
    assert words_of(analysis) == ["就是说"]
    assert analysis["fillers"][0]["start"] == 0.4
    assert analysis["fillers"][0]["end"] == 1.0


def test_phrase_you_know():
    segments = [seg([wd("well", 0.0, 0.3), wd("you", 0.3, 0.6), wd(" know", 0.6, 0.9), wd(" test", 1.4, 1.8)])]
    analysis = filler.analyze(segments)
    assert words_of(analysis) == ["you know"]


def test_phrase_zenme_shuone():
    segments = [seg([wd("大家", 0.0, 0.4), wd("怎么", 0.4, 0.7), wd("说呢", 0.7, 1.0), wd("我们", 1.5, 1.9)])]
    assert words_of(filler.analyze(segments)) == ["怎么说呢"]


def test_phrase_normal_use_not_counted():
    segments = [seg([wd("就是说", 0.0, 0.6), wd("测试", 0.6, 1.0), wd("跑通", 1.0, 1.4)])]
    assert filler.analyze(segments)["fillers"] == []


def test_custom_vocab_from_parameter():
    segments = [seg([wd("测试", 0.0, 0.4), wd("跑通", 0.4, 0.8)])]
    analysis = filler.analyze(segments, filler_words={"pure": ["测试"], "contextual": []})
    assert words_of(analysis) == ["测试"]


STATS_SEGMENTS = [
    {
        "start": 0.0, "end": 2.0, "text": "嗯,这个这个,我想说进度.",
        "words": [
            wd("嗯,", 0.0, 0.3), wd("这个这个,", 0.3, 0.9), wd("我", 0.9, 1.1),
            wd("想说", 1.1, 1.6), wd("进度.", 1.6, 2.0),
        ],
    },
    {
        "start": 3.0, "end": 5.0, "text": "然后测试也跑通了",
        "words": [
            wd("然后", 3.0, 3.3), wd("测试", 3.3, 3.8), wd("也", 3.8, 3.9),
            wd("跑通", 3.9, 4.4), wd("了", 4.4, 5.0),
        ],
    },
]


def test_statistics_exact_dual_ratios():
    analysis = filler.analyze(STATS_SEGMENTS)
    assert analysis["total_tokens"] == 13
    assert analysis["total_speech_duration"] == 4.0
    assert analysis["filler_count"] == 2
    assert analysis["filler_count_ratio"] == round(2 / 13, 6)
    assert analysis["filler_duration"] == 0.9
    assert analysis["filler_duration_ratio"] == 0.225
    assert analysis["fillers"] == [
        {"word": "嗯", "start": 0.0, "end": 0.3, "sentence": "嗯,这个这个,我想说进度."},
        {"word": "这个", "start": 0.3, "end": 0.9, "sentence": "嗯,这个这个,我想说进度."},
    ]


TOP10_SEGMENTS = [
    seg(
        [
            wd("嗯,", 0.0, 0.5), wd("嗯,", 0.5, 1.0), wd("嗯,", 1.0, 1.5), wd("嗯,", 1.5, 2.0),
            wd("这个", 2.0, 2.5), wd("这个,", 2.5, 3.0),
            wd("就是", 3.0, 3.5), wd("就是,", 3.5, 4.0),
            wd("啊", 4.0, 4.5), wd("收尾", 4.5, 10.0),
        ],
        start=0.0, end=10.0,
    )
]


def test_top10_order_merge_and_examples():
    analysis = filler.analyze(TOP10_SEGMENTS)
    assert analysis["filler_count"] == 9
    top = analysis["top10"]
    assert [g["word"] for g in top] == ["嗯", "就是", "这个", "啊"]
    assert [g["count"] for g in top] == [4, 2, 2, 1]
    assert [g["duration"] for g in top] == [2.0, 1.0, 1.0, 0.5]
    en = top[0]
    assert len(en["examples"]) == 3
    assert [e["start"] for e in en["examples"]] == [0.0, 0.5, 1.0]
    assert all(set(e) == {"start", "end", "sentence"} for e in en["examples"])
    assert en["examples"][0]["sentence"] == "嗯,嗯,嗯,嗯,这个这个,就是就是,啊收尾"


def test_empty_segments_zero_safe():
    analysis = filler.analyze([])
    assert analysis["total_tokens"] == 0
    assert analysis["total_speech_duration"] == 0.0
    assert analysis["filler_count"] == 0
    assert analysis["filler_count_ratio"] == 0.0
    assert analysis["filler_duration"] == 0.0
    assert analysis["filler_duration_ratio"] == 0.0
    assert analysis["top10"] == []
    assert analysis["fillers"] == []


def _bare_seg(text, start, end):
    return {"start": start, "end": end, "text": text, "words": []}


def test_hallucination_run_removed_keeps_first():
    segments = [
        _bare_seg("幻觉重复字幕", 0.0, 2.0),
        _bare_seg("幻觉重复字幕", 2.0, 4.0),
        _bare_seg("幻觉重复字幕", 4.0, 6.0),
        _bare_seg("正常讨论内容", 6.0, 10.0),
    ]
    kept = filler.filter_hallucination_segments(segments)
    assert [s["text"] for s in kept] == ["幻觉重复字幕", "正常讨论内容"]


def test_hallucination_low_ratio_kept():
    segments = [
        _bare_seg("好的", 0.0, 0.5),
        _bare_seg("好的", 0.5, 1.0),
        _bare_seg("好的", 1.0, 1.5),
        _bare_seg("正常讨论内容", 1.5, 20.0),
    ]
    assert filler.filter_hallucination_segments(segments) == segments


def test_hallucination_two_repeats_kept():
    segments = [_bare_seg("幻觉重复字幕", 0.0, 5.0), _bare_seg("幻觉重复字幕", 5.0, 10.0)]
    assert filler.filter_hallucination_segments(segments) == segments


def test_hallucination_distinct_texts_kept():
    segments = [_bare_seg("第一句", 0.0, 2.0), _bare_seg("第二句", 2.0, 4.0), _bare_seg("第三句", 4.0, 6.0)]
    assert filler.filter_hallucination_segments(segments) == segments


def test_hallucination_empty_input():
    assert filler.filter_hallucination_segments([]) == []