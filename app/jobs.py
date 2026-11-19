import json
import re
import threading
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from . import audio, engines as engines_probe, filler, keypoints, settings as app_settings
from .asr.faster_engine import FasterWhisperEngine
from .asr.rocm_engine import RocmWhisperEngine
from .asr.watchdog import run_with_watchdog
from .settings import DATA_DIR, VALID_ENGINES, VALID_MODELS

UPLOADS_DIR = DATA_DIR / "uploads"

router = APIRouter(prefix="/api")

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

_JOB_META_KEYS = ("job_id", "filename", "engine", "model", "created_at", "upload_path")


def build_engine(engine: str, model: str, language: str):
    if engine == "rocm":
        return RocmWhisperEngine(model=model, language=language)
    return FasterWhisperEngine(device=engine, model=model, language=language)


def _public_view(job: dict) -> dict:
    result = job["result"] or {}
    return {
        "job_id": job["job_id"],
        "filename": job["filename"],
        "engine": job["engine"],
        "model": job["model"],
        "status": job["status"],
        "progress": job["progress"],
        "error": job["error"],
        "created_at": job["created_at"],
        "result": job["result"],
        "engine_fallback": result.get("engine_fallback"),
        "warnings": result.get("warnings") or [],
    }


def _set_status(job_id: str, status: str, progress: int) -> None:
    with _jobs_lock:
        job = _jobs[job_id]
        job["status"] = status
        job["progress"] = progress


def _persist_result(job: dict) -> None:
    payload = {
        "job": {k: job[k] for k in _JOB_META_KEYS},
        "result": job["result"],
    }
    result_path = Path(job["upload_path"]).parent / "result.json"
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def restore_jobs() -> int:
    restored = 0
    if not UPLOADS_DIR.exists():
        return restored
    for result_path in sorted(UPLOADS_DIR.glob("*/result.json")):
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        meta = payload.get("job")
        result = payload.get("result")
        if not isinstance(meta, dict) or not isinstance(result, dict):
            continue
        job_id = meta.get("job_id")
        if not job_id or any(k not in meta for k in _JOB_META_KEYS):
            continue
        with _jobs_lock:
            if job_id in _jobs:
                continue
            _jobs[job_id] = {
                **{k: meta[k] for k in _JOB_META_KEYS},
                "status": "done",
                "progress": 100,
                "error": None,
                "result": result,
            }
            restored += 1
    return restored


def _transcribe_with_fallback(job: dict, wav_path: Path, language: str, cfg: dict, on_progress):
    requested = job["engine"]
    fallback = None
    warnings: list = []
    if requested in ("cuda", "rocm"):
        probe = engines_probe.probe_engines().get(
            requested, {"available": False, "detail": "未知引擎"}
        )
        reason = None
        if probe["available"]:
            try:
                engine = build_engine(requested, job["model"], language)
                timeout = float(cfg.get("watchdog_seconds", 600))

                def target(heartbeat):
                    def cb(segment_end: float) -> None:
                        heartbeat()
                        on_progress(segment_end)

                    return engine.transcribe(wav_path, progress_cb=cb)

                return run_with_watchdog(target, timeout), None, []
            except Exception as exc:
                reason = str(exc)
        else:
            reason = probe["detail"]
        fallback = {"from": requested, "to": "cpu", "reason": reason}
        warnings.append(f"{requested} 引擎失败已回退 CPU：{reason}")
        with _jobs_lock:
            job["progress"] = 0
    engine = build_engine("cpu", job["model"], language)
    return engine.transcribe(wav_path, progress_cb=on_progress), fallback, warnings


def _run_pipeline(job_id: str) -> None:
    job = _jobs[job_id]
    try:
        _set_status(job_id, "extracting", 0)
        wav_path = audio.extract_audio(
            job["upload_path"], Path(job["upload_path"]).parent / "audio.wav"
        )
        duration = audio.probe_duration(wav_path)
        cfg = app_settings.load_settings()
        language = cfg.get("language", "zh")
        _set_status(job_id, "transcribing", 0)

        def on_progress(segment_end: float) -> None:
            pct = int(segment_end / duration * 100) if duration > 0 else 99
            pct = max(0, min(99, pct))
            with _jobs_lock:
                job["progress"] = max(job["progress"], pct)

        transcript, fallback, warnings = _transcribe_with_fallback(
            job, wav_path, language, cfg, on_progress
        )
        _set_status(job_id, "analyzing", 99)
        segments = filler.filter_hallucination_segments(transcript["segments"])
        analysis = filler.analyze(segments)
        report = keypoints.extract_keypoints_detailed(segments, mode="llm", analysis=analysis)
        result = {
            "wav_path": str(wav_path),
            "duration": duration,
            "language": transcript["language"],
            "text": "".join(seg["text"] for seg in segments),
            "segments": segments,
            "analysis": analysis,
            "keypoints": report["keypoints"],
            "keypoint_mode": report["keypoint_mode"],
            "engine_used": fallback["to"] if fallback else job["engine"],
            "engine_fallback": fallback,
            "warnings": warnings + report["warnings"],
        }
        with _jobs_lock:
            job["result"] = result
        _persist_result(job)
        _set_status(job_id, "done", 100)
    except Exception as exc:
        with _jobs_lock:
            job["status"] = "failed"
            job["error"] = str(exc)


def create_job(
    filename: str, engine: str, model: str, upload_path: Path, job_id: str
) -> dict:
    job = {
        "job_id": job_id,
        "filename": filename,
        "engine": engine,
        "model": model,
        "status": "queued",
        "progress": 0,
        "error": None,
        "created_at": time.time(),
        "upload_path": str(upload_path),
        "result": None,
    }
    with _jobs_lock:
        _jobs[job_id] = job
    threading.Thread(target=_run_pipeline, args=(job_id,), daemon=True).start()
    return job


@router.post("/jobs")
async def submit_job(
    file: UploadFile,
    engine: str = Form("cpu"),
    model: str = Form("small"),
):
    if engine not in VALID_ENGINES:
        raise HTTPException(400, f"engine 必须是 {sorted(VALID_ENGINES)} 之一")
    if model not in VALID_MODELS:
        raise HTTPException(400, f"model 必须是 {sorted(VALID_MODELS)} 之一")
    filename = Path(file.filename or "").name
    try:
        audio.check_extension(filename)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    safe_name = re.sub(r'[^\w.\-]', "_", filename) or f"upload{Path(filename).suffix}"
    job_id = uuid.uuid4().hex[:12]
    job_dir = UPLOADS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    upload_path = job_dir / safe_name
    with upload_path.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            out.write(chunk)
    job = create_job(filename, engine, model, upload_path, job_id)
    return {"job_id": job["job_id"], "status": job["status"]}


@router.get("/engines")
def get_engines():
    return engines_probe.probe_engines()


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "任务不存在")
        return _public_view(job)


@router.get("/jobs/{job_id}/result")
def get_job_result(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "任务不存在")
        if job["result"] is None:
            raise HTTPException(409, f"任务尚未产出结果（当前状态：{job['status']}）")
        return job["result"]


@router.get("/jobs")
def list_jobs():
    with _jobs_lock:
        jobs = sorted(_jobs.values(), key=lambda j: j["created_at"], reverse=True)
        return [_public_view(j) for j in jobs]


_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")
_AUDIO_CHUNK = 64 * 1024


def _iter_file(path: Path, start: int, length: int):
    with path.open("rb") as f:
        f.seek(start)
        remaining = length
        while remaining > 0:
            chunk = f.read(min(_AUDIO_CHUNK, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


@router.get("/audio/{job_id}")
def stream_audio(job_id: str, request: Request):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "任务不存在")
    wav_path = Path(job["upload_path"]).parent / "audio.wav"
    if not wav_path.is_file():
        raise HTTPException(404, "音频文件不存在")
    size = wav_path.stat().st_size
    start, end = 0, size - 1
    status = 200
    range_header = request.headers.get("range")
    if range_header:
        match = _RANGE_RE.match(range_header.strip())
        if match and (match.group(1) or match.group(2)):
            if not match.group(1):
                suffix = int(match.group(2))
                if suffix == 0 or size == 0:
                    raise HTTPException(
                        416, "Range 超出范围", headers={"Content-Range": f"bytes */{size}"}
                    )
                start = max(0, size - suffix)
            else:
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else size - 1
                if start >= size or start > end:
                    raise HTTPException(
                        416, "Range 超出范围", headers={"Content-Range": f"bytes */{size}"}
                    )
                end = min(end, size - 1)
            status = 206
    length = end - start + 1
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
    }
    if status == 206:
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    return StreamingResponse(
        _iter_file(wav_path, start, length),
        status_code=status,
        headers=headers,
        media_type="audio/wav",
    )