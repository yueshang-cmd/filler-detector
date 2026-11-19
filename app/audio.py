import re
import shutil
import subprocess
import wave
from pathlib import Path

SUPPORTED_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".mp4", ".mkv", ".mov", ".webm",
}

FFMPEG_INSTALL_HINT = (
    "未找到 ffmpeg。请任选其一：\n"
    "1) pip install imageio-ffmpeg（本项目 requirements.txt 已包含，"
    "确认已在 .venv 中安装依赖）；\n"
    "2) 安装系统 ffmpeg 并加入 PATH：winget install Gyan.FFmpeg，"
    "或从 https://www.gyan.dev/ffmpeg/builds/ 下载后将 bin 目录加入 PATH。"
)


class FfmpegNotFoundError(RuntimeError):
    pass


def find_ffmpeg() -> tuple[str, str]:
    system = shutil.which("ffmpeg")
    if system:
        return system, "system"
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe(), "imageio-ffmpeg"
    except Exception as exc:
        raise FfmpegNotFoundError(FFMPEG_INSTALL_HINT) from exc


def check_extension(path: str | Path) -> None:
    ext = Path(path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"不支持的文件类型 {ext or '(无扩展名)'}，"
            f"支持：{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )


def extract_audio(src: str | Path, dst: str | Path) -> Path:
    src, dst = Path(src), Path(dst)
    check_extension(src)
    ffmpeg, _ = find_ffmpeg()
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-i", str(src),
        "-vn", "-ar", "16000", "-ac", "1", str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0 or not dst.exists():
        tail = (proc.stderr or "").strip().splitlines()[-8:]
        raise RuntimeError("ffmpeg 转码失败：\n" + "\n".join(tail))
    return dst


def probe_duration(path: str | Path) -> float:
    path = Path(path)
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as wf:
                return wf.getnframes() / wf.getframerate()
        except (wave.Error, EOFError):
            pass
    ffmpeg, _ = find_ffmpeg()
    # imageio-ffmpeg 不附带 ffprobe，只能解析 `ffmpeg -i` 的 stderr
    proc = subprocess.run(
        [ffmpeg, "-i", str(path)], capture_output=True, text=True, errors="replace"
    )
    match = re.search(
        r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr or ""
    )
    if not match:
        raise RuntimeError(f"无法探测时长：{path.name}")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)