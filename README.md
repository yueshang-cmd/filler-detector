# 会议/课程废话检测器

本地网页应用：上传会议/课程录音或视频 → 本地转写 → 输出废话浓度（次数/时长双口径）、废话 Top 10（可点击跳转）、有用重点列表、带时间戳完整转写，并支持导出 Markdown/JSON 报告。全程本地运行，录音不出本机（除非主动配置 LLM API 提炼重点）。

实施规格见 [PLAN.md](PLAN.md)。

## 功能概览

- **三引擎 ASR**（faster-whisper / transformers，词级时间戳 + VAD）：CPU（默认，int8）、CUDA GPU（float16）、ROCm GPU（可选依赖）；自动探测，GPU 失败或看门狗超时自动回退 CPU 并在结果中提示
- **废话判定**：内置纯语气词 + 语境词词表（设置面板可增删），四条启发式规则（连续重复 / 词后 ≥0.5s 停顿 / 句首承接 / 30s 窗口口头禅）
- **重点提炼**：默认离线 TextRank；设置面板可配置 OpenAI 兼容 LLM API（base_url / key / model），失败自动回退离线
- **界面**：FastAPI 托管单页（Vue 3，本地 vendor，无构建链），浏览器访问 http://127.0.0.1:8020

## 快速开始（Windows）

1. 安装 [Python 3.11+](https://www.python.org/downloads/)（本机实测 3.12.10），确保 `python` 在 PATH。
2. 双击 `run.cmd`：首次自动创建 `.venv` → 安装依赖 → 启动服务 → 打开浏览器。
3. 手动方式：

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8020
```

支持格式：mp3 / wav / m4a / flac / ogg / mp4 / mkv / mov / webm。

## 安装与下载渠道

| 依赖 | 渠道 | 说明 |
|---|---|---|
| Python | https://www.python.org/downloads/ | 3.11+，Windows x64 |
| 核心 Python 包 | PyPI（`pip install -r requirements.txt`） | 版本已精确锁定：`faster-whisper==1.2.1` + `av==14.2.0`（av≥15 与 faster-whisper 不兼容，真实转写会 TypeError）+ `ctranslate2==4.9.0` 等 |
| PyPI 镜像（可选） | `-i https://pypi.tuna.tsinghua.edu.cn/simple` | pip 官方源慢/不通时使用 |
| ffmpeg | **无需单独安装**：`imageio-ffmpeg` 随 requirements 自动携带；若系统已有 ffmpeg 则优先使用 | 手动安装渠道：https://www.gyan.dev/ffmpeg/builds/ 或 https://github.com/BtbN/FFmpeg-Builds/releases |
| 转写模型 | HuggingFace Hub `Systran/faster-whisper-small`（首次运行自动下载并缓存到 `~/.cache/huggingface`） | 直连失败时设环境变量 `HF_ENDPOINT=https://hf-mirror.com` 重试；medium / large-v3 同理 |
| ROCm GPU 引擎（可选） | AMD 官方 wheel 源 https://stable.repo.amd.com/rocm/whl-next （301 → /rocm/pytorch/whl-next/） | 2026-11-20 验证可用：`torch-2.15.0+rocm10.2.0-cp312-cp312-win_amd64.whl` + `amd_torch_device_gfx1201`（RX 9070 XT）；精确安装行见 [requirements-rocm.txt](requirements-rocm.txt) 头部注释。不安装不影响 CPU/CUDA |
| LLM 重点提炼（可选） | 任意 OpenAI 兼容 API（自填 base_url / key / model） | 仅配置后启用；失败自动回退离线算法 |

## 测试

```bash
.venv\Scripts\python.exe -m pytest            # 默认套件（FakeEngine，快）
.venv\Scripts\python.exe -m pytest -m real    # 真实音频端到端冒烟（需模型缓存与 CPU 时间）
```

测试音频由 edge-tts 合成（`requirements-dev.txt`），底稿见 `tests/assets/chinese_speech_text.txt`。

## 目录结构

```
app/            FastAPI 入口、任务队列、ASR 引擎、废话分析、重点提炼、设置
static/         单页前端（Vue 3 本地 vendor，无构建链）
data/           上传文件、结果 JSON、settings.json（已 gitignore，含本机密钥不入库）
tests/          pytest 用例与真实音频资产
run.cmd         双击启动脚本
requirements.txt / requirements-rocm.txt / requirements-dev.txt
```

## 注意事项

- 默认引擎为 CPU（本机 AMD GPU 有慢性挂起史）；GPU 引擎带 10 分钟无进度看门狗，超时自动回退 CPU。
- 数据与配置全部存于本机 `data/`；`settings.json` 中的 LLM key 为明文本地存储，仓库已排除该目录。
- 长音频（>1h）建议 CPU + small 模型分段处理，medium/large-v3 会显著变慢。
