import json
import math
import re
from collections import Counter

import httpx
import jieba

from . import filler
from .settings import load_settings

_SENT_END = "。！？!?"
_CLAUSE_BREAKS = "，,"
_WORD_CHAR_RE = re.compile(r"[0-9A-Za-z\u4e00-\u9fff]")
_EDGE_PUNCT = "。，、；：,.;:！？!? \t\r\n\u3000"
MAX_SENTENCE_CHARS = 40

STOPWORDS = frozenset(
    """的 了 是 在 我 你 他 她 它 我们 你们 他们 它们 这 那 这个 那个 就 就是 也 都 和 与 及
    吧 呢 啊 吗 嘛 哦 呀 哈 嗯 呃 一个 一些 一下 一点 什么 怎么 怎样 这么 那么 这样 那样
    现在 今天 明天 昨天 大家 自己 各自 可以 需要 还是 但是 因为 所以 如果 虽然 没有
    不 很 太 再 先 把 被 让 给 对 从 到 上 下 里 中 个 们 地 得 着 过 之 其 以及 然后 其实
    反正 对吧 就是说 怎么说呢 好 等 等等 的话 来说 有点 稍微 主要 另外 接下来 回去 随时
    一起 这里 那边 这边 一块 基本 还 又 才 只 可能 应该 觉得 知道 想 说 做 弄 搞 东西
    时候 地方 方面 情况 比较 通过 进行 之后 以后 以前 目前 暂时 大概 差不多""".split()
)

DAMPING = 0.85
MAX_ITER = 100
TOL = 1e-6
HEAD_TAIL_FRACTION = 0.2
POSITION_BOOST = 1.25
COVERAGE_WEIGHT = 0.5
DENSITY_WEIGHT = 0.2
MAX_DENSITY_TOKENS = 10
FILLER_RATIO_THRESHOLD = 0.5
SHORT_SENTENCE_CHARS = 15
PURE_FILLER_RATIO = 0.9
TOP_KEYWORDS = 10
MIN_POINTS = 8
MAX_POINTS = 15


def _make_sentence(words: list) -> dict | None:
    text = "".join(w["word"] for w in words).strip()
    if not _WORD_CHAR_RE.search(text):
        return None
    return {"text": text, "start": float(words[0]["start"]), "end": float(words[-1]["end"])}


def _char_len(text: str) -> int:
    return len(_WORD_CHAR_RE.findall(text))


def _emit_words(words: list) -> list:
    sent = _make_sentence(words)
    if sent is None:
        return []
    if _char_len(sent["text"]) <= MAX_SENTENCE_CHARS:
        return [sent]
    out = []
    buf = []
    for wd in words:
        buf.append(wd)
        if any(ch in wd["word"] for ch in _CLAUSE_BREAKS):
            piece = _make_sentence(buf)
            if piece:
                out.append(piece)
            buf = []
    if buf:
        piece = _make_sentence(buf)
        if piece:
            out.append(piece)
    return out or [sent]


def _emit_text(text: str, start: float, end: float) -> list:
    text = text.strip()
    if not _WORD_CHAR_RE.search(text):
        return []
    if _char_len(text) <= MAX_SENTENCE_CHARS:
        return [{"text": text, "start": start, "end": end}]
    pieces = [p for p in re.split(r"(?<=[，,])", text) if _WORD_CHAR_RE.search(p)]
    if len(pieces) <= 1:
        return [{"text": text, "start": start, "end": end}]
    total_chars = sum(len(p) for p in pieces)
    out = []
    cursor = 0
    for piece in pieces:
        frac_a = cursor / total_chars
        cursor += len(piece)
        frac_b = cursor / total_chars
        out.append(
            {
                "text": piece.strip(),
                "start": start + (end - start) * frac_a,
                "end": start + (end - start) * frac_b,
            }
        )
    return out


def _split_sentences(segments: list) -> list:
    sentences = []
    for seg in segments:
        words = seg.get("words") or []
        if words:
            buf = []
            for wd in words:
                buf.append(wd)
                if any(ch in wd["word"] for ch in _SENT_END):
                    sentences.extend(_emit_words(buf))
                    buf = []
            if buf:
                sentences.extend(_emit_words(buf))
        else:
            text = seg.get("text", "")
            pieces = [p for p in re.split(r"(?<=[。！？!?])", text) if _WORD_CHAR_RE.search(p)]
            if not pieces:
                continue
            total_chars = sum(len(p) for p in pieces)
            span_start, span_end = float(seg["start"]), float(seg["end"])
            cursor = 0
            for piece in pieces:
                frac_a = cursor / total_chars
                cursor += len(piece)
                frac_b = cursor / total_chars
                sentences.extend(
                    _emit_text(
                        piece,
                        span_start + (span_end - span_start) * frac_a,
                        span_start + (span_end - span_start) * frac_b,
                    )
                )
    return sentences


def _filler_norms() -> set:
    words = load_settings()["filler_words"]
    norms = set()
    for kind in ("pure", "contextual"):
        for raw in words.get(kind, []):
            norm = re.sub(r"\s+", "", raw).lower()
            if norm:
                norms.add(norm)
    return norms


def _content_tokens(text: str, filler_norms: set) -> list:
    tokens = []
    for tok in jieba.lcut(text):
        if len(tok) < 2 or not _WORD_CHAR_RE.search(tok):
            continue
        low = tok.lower()
        if low in STOPWORDS or low in filler_norms:
            continue
        tokens.append(low)
    return tokens


def _filler_ratio(sent: dict, fillers: list) -> float:
    total = filler.count_tokens(sent["text"])
    if total == 0:
        return 1.0
    hit = 0
    for f in fillers:
        mid = (f["start"] + f["end"]) / 2
        if sent["start"] - 1e-6 <= mid <= sent["end"] + 1e-6:
            hit += max(1, filler.count_tokens(f["word"]))
    return hit / total


def _textrank(token_lists: list) -> list:
    n = len(token_lists)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    counters = [Counter(toks) for toks in token_lists]
    log_lens = [math.log(len(toks) + 1) for toks in token_lists]
    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            common = sum((counters[i] & counters[j]).values())
            if common == 0:
                continue
            s = common / (log_lens[i] + log_lens[j])
            sim[i][j] = sim[j][i] = s
    out_sum = [sum(row) for row in sim]
    scores = [1.0 / n] * n
    for _ in range(MAX_ITER):
        dangling = sum(scores[j] for j in range(n) if out_sum[j] == 0) / n
        new_scores = []
        for i in range(n):
            incoming = sum(
                sim[j][i] / out_sum[j] * scores[j]
                for j in range(n)
                if out_sum[j] > 0 and sim[j][i] > 0
            )
            new_scores.append((1 - DAMPING) / n + DAMPING * (incoming + dangling))
        delta = sum(abs(a - b) for a, b in zip(scores, new_scores))
        scores = new_scores
        if delta < TOL:
            break
    return scores


def _position_weight(index: int, total: int) -> float:
    if total <= 1:
        return POSITION_BOOST
    p = index / (total - 1)
    if p <= HEAD_TAIL_FRACTION or p >= 1 - HEAD_TAIL_FRACTION:
        return POSITION_BOOST
    return 1.0


def _target_count(span: float) -> int:
    extra = max(0, int((span - 120) // 120))
    return min(MAX_POINTS, MIN_POINTS + extra)


def _extract_offline(segments: list, analysis: dict | None) -> list:
    sentences = _split_sentences(segments)
    if not sentences:
        return []
    if analysis is None:
        analysis = filler.analyze(segments)
    fillers = analysis.get("fillers", [])
    filler_norms = _filler_norms()

    candidates = []
    for idx, sent in enumerate(sentences):
        ratio = _filler_ratio(sent, fillers)
        char_len = _char_len(sent["text"])
        if ratio >= PURE_FILLER_RATIO:
            continue
        if ratio > FILLER_RATIO_THRESHOLD and char_len < SHORT_SENTENCE_CHARS:
            continue
        tokens = _content_tokens(sent["text"], filler_norms)
        if not tokens:
            continue
        candidates.append({"sent": sent, "idx": idx, "tokens": tokens})
    if not candidates:
        return []

    tr_scores = _textrank([c["tokens"] for c in candidates])
    keyword_counts = Counter(tok for c in candidates for tok in c["tokens"])
    keywords = {w for w, _ in keyword_counts.most_common(TOP_KEYWORDS)}
    total_sents = len(sentences)

    scored = []
    for c, tr in zip(candidates, tr_scores):
        distinct = set(c["tokens"])
        coverage = len(distinct & keywords) / len(keywords) if keywords else 0.0
        pos = _position_weight(c["idx"], total_sents)
        density = 1 + DENSITY_WEIGHT * min(len(distinct), MAX_DENSITY_TOKENS)
        raw = tr * pos * (1 + COVERAGE_WEIGHT * coverage) * density
        scored.append((raw, c))
    max_raw = max(raw for raw, _ in scored) or 1.0

    span = max(s["end"] for s in sentences) - min(s["start"] for s in sentences)
    limit = _target_count(span)
    scored.sort(key=lambda item: (-item[0], item[1]["sent"]["start"]))

    points = []
    for raw, c in scored[:limit]:
        sent = c["sent"]
        importance = max(1, min(5, math.ceil(raw / max_raw * 5 - 1e-9)))
        points.append(
            {
                "point": sent["text"].strip(_EDGE_PUNCT),
                "importance": importance,
                "start": round(sent["start"], 3),
                "end": round(sent["end"], 3),
            }
        )
    return points


LLM_CHUNK_CHARS = 3000
LLM_TIMEOUT_SECONDS = 60.0
LLM_MAX_POINTS = 15
LLM_TEMPERATURE = 0.2

_LLM_SYSTEM = "你是会议/课程内容分析助手。只输出 JSON 数组本身，不要输出解释、前后缀或 Markdown 代码围栏。"
_LLM_PROMPT = (
    "从下面带时间戳的转写中提取 8~15 条有用重点，忽略语气词、废话与寒暄。"
    '严格输出 JSON 数组，每项形如 {"point": "重点文本", "importance": 1到5的整数, '
    '"timestamp_start": 起始秒数, "timestamp_end": 结束秒数}，'
    "时间戳必须取自转写中标注的区间：\n"
)


def _build_chunks(segments: list) -> list:
    lines = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"[{float(seg['start']):.1f}-{float(seg['end']):.1f}] {text}")
    chunks = []
    buf: list = []
    size = 0
    for line in lines:
        if buf and size + len(line) > LLM_CHUNK_CHARS:
            chunks.append("\n".join(buf))
            buf = []
            size = 0
        buf.append(line)
        size += len(line)
    if buf:
        chunks.append("\n".join(buf))
    return chunks


def _parse_llm_points(content: str) -> list:
    text = content.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("["), text.rfind("]")
        if start < 0 or end <= start:
            raise ValueError("LLM 响应中未找到 JSON 数组")
        data = json.loads(text[start : end + 1])
    if not isinstance(data, list):
        raise ValueError("LLM 响应不是 JSON 数组")
    points = []
    for item in data:
        if not isinstance(item, dict):
            continue
        point = str(item.get("point") or "").strip()
        if not point:
            continue
        try:
            importance = int(item.get("importance", 3))
        except (TypeError, ValueError):
            importance = 3
        importance = max(1, min(5, importance))
        raw_start = item.get("timestamp_start", item.get("start", 0.0))
        raw_end = item.get("timestamp_end", item.get("end", raw_start))
        try:
            start, end = float(raw_start), float(raw_end)
        except (TypeError, ValueError):
            continue
        if end < start:
            end = start
        points.append(
            {
                "point": point,
                "importance": importance,
                "start": round(start, 3),
                "end": round(end, 3),
            }
        )
    return points


def _extract_llm(segments: list, llm_cfg: dict) -> list:
    url = llm_cfg["base_url"].rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if llm_cfg.get("key"):
        headers["Authorization"] = f"Bearer {llm_cfg['key']}"
    points = []
    for chunk in _build_chunks(segments):
        payload = {
            "model": llm_cfg["model"],
            "temperature": LLM_TEMPERATURE,
            "messages": [
                {"role": "system", "content": _LLM_SYSTEM},
                {"role": "user", "content": _LLM_PROMPT + chunk},
            ],
        }
        resp = httpx.post(url, json=payload, headers=headers, timeout=LLM_TIMEOUT_SECONDS)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        points.extend(_parse_llm_points(content))
    merged: dict = {}
    for p in points:
        key = re.sub(r"\s+", "", p["point"])
        if key not in merged or p["importance"] > merged[key]["importance"]:
            merged[key] = p
    ordered = sorted(merged.values(), key=lambda p: (-p["importance"], p["start"]))
    return ordered[:LLM_MAX_POINTS]


def extract_keypoints_detailed(
    segments: list, mode: str = "offline", analysis: dict | None = None
) -> dict:
    warnings: list = []
    if mode == "llm":
        llm_cfg = load_settings().get("llm") or {}
        if llm_cfg.get("enabled") and llm_cfg.get("base_url") and llm_cfg.get("model"):
            try:
                points = _extract_llm(segments, llm_cfg)
                if points or not _build_chunks(segments):
                    return {"keypoints": points, "keypoint_mode": "llm", "warnings": []}
                warnings.append("LLM 失败已回退离线：返回重点为空")
            except Exception as exc:
                warnings.append(f"LLM 失败已回退离线：{exc}")
    elif mode != "offline":
        raise ValueError(f"未知的 keypoint mode：{mode}")
    return {
        "keypoints": _extract_offline(segments, analysis),
        "keypoint_mode": "offline",
        "warnings": warnings,
    }


def extract_keypoints(segments: list, mode: str = "offline", analysis: dict | None = None) -> list:
    return extract_keypoints_detailed(segments, mode=mode, analysis=analysis)["keypoints"]