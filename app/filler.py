import logging
import re
import unicodedata
from dataclasses import dataclass

import jieba

from .settings import load_settings

jieba.setLogLevel(logging.ERROR)

PAUSE_THRESHOLD = 0.5
WINDOW_SECONDS = 30.0
WINDOW_MIN_COUNT = 3
MAX_PHRASE_TOKENS = 4
_EPS = 1e-9

_TOKEN_RE = re.compile(r"[0-9A-Za-z\u4e00-\u9fff]")
_COMMAS = (",", "，")


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip().lower()
    while text and unicodedata.category(text[0])[0] in "PS":
        text = text[1:]
    while text and unicodedata.category(text[-1])[0] in "PS":
        text = text[:-1]
    return text


def _nospace(text: str) -> str:
    return re.sub(r"\s+", "", _norm(text))


@dataclass(frozen=True)
class _Entry:
    word: str
    norm: str
    kind: str


def _build_vocab(filler_words: dict):
    vocab = {}
    singles = []
    for kind in ("pure", "contextual"):
        for raw in filler_words.get(kind, []):
            norm = _nospace(raw)
            if not norm:
                continue
            entry = _Entry(word=raw.strip(), norm=norm, kind=kind)
            vocab[norm] = entry
            if " " not in raw.strip():
                singles.append(entry)
    return vocab, singles


def count_tokens(text: str) -> int:
    return sum(1 for piece in jieba.lcut(text) if _TOKEN_RE.search(piece))


def filter_hallucination_segments(
    segments: list, min_run: int = 3, min_ratio: float = 0.3
) -> list:
    total = sum(s["end"] - s["start"] for s in segments)
    if total <= 0:
        return list(segments)
    kept = []
    i = 0
    while i < len(segments):
        text = segments[i]["text"].strip()
        j = i
        while (
            text
            and j + 1 < len(segments)
            and segments[j + 1]["text"].strip() == text
        ):
            j += 1
        run_len = j - i + 1
        run_dur = sum(s["end"] - s["start"] for s in segments[i : j + 1])
        if text and run_len >= min_run and run_dur / total >= min_ratio:
            kept.append(segments[i])
        else:
            kept.extend(segments[i : j + 1])
        i = j + 1
    return kept


@dataclass
class _Match:
    entry: _Entry
    i: int
    j: int
    seg: int
    start: float
    end: float
    forced: bool = False


def _find_matches(tokens: list, vocab: dict, singles: list) -> list:
    matches = []
    i = 0
    n = len(tokens)
    while i < n:
        if not tokens[i]["norm"]:
            i += 1
            continue
        hit = None
        for k in range(min(MAX_PHRASE_TOKENS, n - i), 0, -1):
            if tokens[i + k - 1]["seg"] != tokens[i]["seg"]:
                continue
            joined = "".join(t["norm"] for t in tokens[i : i + k])
            if joined in vocab:
                hit = (k, vocab[joined])
                break
        if hit:
            k, entry = hit
            matches.append(
                _Match(entry, i, i + k, tokens[i]["seg"], tokens[i]["start"], tokens[i + k - 1]["end"])
            )
            i += k
            continue
        text = tokens[i]["norm"]
        for entry in singles:
            reps, rem = divmod(len(text), len(entry.norm))
            if reps >= 2 and rem == 0 and text == entry.norm * reps:
                matches.append(
                    _Match(entry, i, i + 1, tokens[i]["seg"], tokens[i]["start"], tokens[i]["end"], forced=True)
                )
                break
        i += 1
    return matches


def _next_speech_index(tokens: list) -> list:
    nxt = [None] * (len(tokens) + 1)
    for k in range(len(tokens) - 1, -1, -1):
        nxt[k] = k if tokens[k]["norm"] else nxt[k + 1]
    return nxt


def _pause_after(match: _Match, tokens: list, nxt: list) -> float | None:
    k = nxt[match.j]
    if k is None:
        return None
    return tokens[k]["start"] - match.end


def _comma_after(match: _Match, tokens: list) -> bool:
    last_raw = tokens[match.j - 1]["raw"].rstrip()
    if last_raw.endswith(_COMMAS):
        return True
    if match.j < len(tokens):
        nxt_raw = tokens[match.j]["raw"].lstrip()
        if nxt_raw.startswith(_COMMAS):
            return True
    return False


def _window_flagged(matches: list) -> set:
    flagged = set()
    by_word: dict[str, list[int]] = {}
    for idx, m in enumerate(matches):
        if m.entry.kind == "contextual":
            by_word.setdefault(m.entry.norm, []).append(idx)
    for idxs in by_word.values():
        left = 0
        for right in range(len(idxs)):
            while (
                matches[idxs[right]].start - matches[idxs[left]].start
                > WINDOW_SECONDS + _EPS
            ):
                left += 1
            if right - left + 1 >= WINDOW_MIN_COUNT:
                flagged.update(idxs[left : right + 1])
    return flagged


def _contextual_hit(match: _Match, tokens: list, nxt: list, first_idx: dict) -> bool:
    if match.forced:
        return True
    norm = match.entry.norm
    if match.i > 0 and tokens[match.i - 1]["norm"] == norm:
        return True
    if match.j < len(tokens) and tokens[match.j]["norm"] == norm:
        return True
    gap = _pause_after(match, tokens, nxt)
    if gap is not None and gap >= PAUSE_THRESHOLD - _EPS:
        return True
    if first_idx.get(match.seg) == match.i and (
        _comma_after(match, tokens)
        or (gap is not None and gap >= PAUSE_THRESHOLD - _EPS)
    ):
        return True
    return False


def analyze(segments: list, filler_words: dict | None = None) -> dict:
    if filler_words is None:
        filler_words = load_settings()["filler_words"]
    vocab, singles = _build_vocab(filler_words)

    tokens = []
    first_idx = {}
    for si, seg in enumerate(segments):
        for wd in seg.get("words") or []:
            norm = _norm(wd["word"])
            if norm and si not in first_idx:
                first_idx[si] = len(tokens)
            tokens.append(
                {
                    "raw": wd["word"],
                    "norm": norm,
                    "start": float(wd["start"]),
                    "end": float(wd["end"]),
                    "seg": si,
                }
            )

    matches = _find_matches(tokens, vocab, singles)
    flagged = _window_flagged(matches)
    nxt = _next_speech_index(tokens)
    fillers = []
    for idx, m in enumerate(matches):
        if m.entry.kind == "pure" or m.forced or idx in flagged or _contextual_hit(m, tokens, nxt, first_idx):
            fillers.append(
                {
                    "word": m.entry.word,
                    "start": m.start,
                    "end": m.end,
                    "sentence": segments[m.seg]["text"].strip(),
                }
            )

    text = " ".join(seg["text"] for seg in segments)
    total_tokens = count_tokens(text)
    total_speech = sum(seg["end"] - seg["start"] for seg in segments)
    filler_duration = sum(f["end"] - f["start"] for f in fillers)

    groups: dict[str, dict] = {}
    for f in fillers:
        g = groups.setdefault(
            f["word"], {"word": f["word"], "count": 0, "duration": 0.0, "examples": []}
        )
        g["count"] += 1
        g["duration"] += f["end"] - f["start"]
        if len(g["examples"]) < 3:
            g["examples"].append(
                {"start": f["start"], "end": f["end"], "sentence": f["sentence"]}
            )
    top10 = sorted(
        groups.values(), key=lambda g: (-g["count"], -g["duration"], g["word"])
    )[:10]
    for g in top10:
        g["duration"] = round(g["duration"], 3)

    return {
        "total_tokens": total_tokens,
        "total_speech_duration": round(total_speech, 3),
        "filler_count": len(fillers),
        "filler_count_ratio": round(len(fillers) / total_tokens, 6) if total_tokens else 0.0,
        "filler_duration": round(filler_duration, 3),
        "filler_duration_ratio": round(filler_duration / total_speech, 6) if total_speech > 0 else 0.0,
        "top10": top10,
        "fillers": fillers,
    }