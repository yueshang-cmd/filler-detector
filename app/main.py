from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from . import audio, jobs, settings

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        exe, source = audio.find_ffmpeg()
        print(f"[filler-detector] ffmpeg ({source}): {exe}")
    except audio.FfmpegNotFoundError:
        print(f"[filler-detector] {audio.FFMPEG_INSTALL_HINT}")
    restored = jobs.restore_jobs()
    if restored:
        print(f"[filler-detector] 已从磁盘恢复 {restored} 个历史任务")
    yield


app = FastAPI(title="filler-detector", lifespan=lifespan)
app.include_router(jobs.router)


@app.get("/api/settings")
def get_settings():
    return settings.load_settings()


@app.put("/api/settings")
def put_settings(payload: dict):
    try:
        merged = settings.validate_settings(payload)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return settings.save_settings(merged)


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")