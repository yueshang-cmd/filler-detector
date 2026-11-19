import json
import math
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SETTINGS_FILE = DATA_DIR / "settings.json"

DEFAULT_SETTINGS = {
    "default_engine": "cpu",
    "default_model": "small",
    "language": "zh",
    "watchdog_seconds": 600,
    "llm": {"base_url": "", "key": "", "model": "", "enabled": False},
    "filler_words": {
        "pure": ["嗯", "啊", "呃", "额", "唔", "emmm", "um", "uh", "er"],
        "contextual": [
            "这个", "那个", "就是", "然后", "其实", "反正", "对吧",
            "就是说", "怎么说呢", "basically", "like", "you know",
            "actually", "so", "right",
        ],
    },
}

VALID_ENGINES = {"cpu", "cuda", "rocm"}
VALID_MODELS = {"small", "medium", "large-v3"}
VALID_LANGUAGES = {"zh", "auto"}

_lock = threading.Lock()


def load_settings() -> dict:
    with _lock:
        settings = json.loads(json.dumps(DEFAULT_SETTINGS))
        if SETTINGS_FILE.exists():
            try:
                saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                saved = {}
            if isinstance(saved, dict):
                settings.update(saved)
        llm_saved = settings.get("llm")
        settings["llm"] = (
            {**DEFAULT_SETTINGS["llm"], **llm_saved}
            if isinstance(llm_saved, dict)
            else dict(DEFAULT_SETTINGS["llm"])
        )
        return settings


def save_settings(settings: dict) -> dict:
    with _lock:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(
            json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return settings


def validate_settings(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("settings 必须是 JSON 对象")
    merged = load_settings()
    llm_payload = payload.get("llm")
    merged.update(payload)
    if isinstance(llm_payload, dict):
        merged["llm"] = {**load_settings()["llm"], **llm_payload}
    if merged.get("default_engine") not in VALID_ENGINES:
        raise ValueError(f"default_engine 必须是 {sorted(VALID_ENGINES)} 之一")
    if merged.get("default_model") not in VALID_MODELS:
        raise ValueError(f"default_model 必须是 {sorted(VALID_MODELS)} 之一")
    if merged.get("language") not in VALID_LANGUAGES:
        raise ValueError(f"language 必须是 {sorted(VALID_LANGUAGES)} 之一")
    words = merged.get("filler_words")
    if not isinstance(words, dict):
        raise ValueError("filler_words 必须包含 pure 与 contextual 两个列表")
    for key in ("pure", "contextual"):
        value = words.get(key)
        if not isinstance(value, list) or not all(isinstance(w, str) for w in value):
            raise ValueError(f"filler_words.{key} 必须是字符串列表")
    watchdog = merged.get("watchdog_seconds")
    if (
        isinstance(watchdog, bool)
        or not isinstance(watchdog, (int, float))
        or not math.isfinite(watchdog)
        or watchdog <= 0
    ):
        raise ValueError("watchdog_seconds 必须是大于 0 的有限数字（秒）")
    llm = merged.get("llm")
    if not isinstance(llm, dict):
        raise ValueError("llm 必须是包含 base_url/key/model/enabled 的 JSON 对象")
    for key in ("base_url", "key", "model"):
        if not isinstance(llm.get(key), str):
            raise ValueError(f"llm.{key} 必须是字符串")
    if not isinstance(llm.get("enabled"), bool):
        raise ValueError("llm.enabled 必须是布尔值")
    if llm["enabled"] and (not llm["base_url"].strip() or not llm["model"].strip()):
        raise ValueError("llm.enabled=true 时 base_url 与 model 必须非空")
    return merged