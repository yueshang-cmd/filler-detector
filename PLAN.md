# 会议/课程废话检测器 — 实施规格（PLAN）

> 本文档是唯一执行依据。编程会话请严格按里程碑顺序实施，每个里程碑有明确验收标准。
> 项目根目录：`D:\QO\filler-detector`

## 1. 产品目标

本地网页应用：用户上传会议/课程录音或视频 → 程序转写音频 → 输出：

1. **废话浓度占比**（两个口径都要给出）：
   - 次数占比 = 废话词出现次数 / 总词数
   - 时长占比 = 废话词占用时长 / 总语音时长（利用 ASR 词级时间戳）
2. **废话排行榜 Top 10**：按出现次数降序，每项含次数、占用时长、例句时间戳（可点击跳转到音频对应位置）。
3. **有用重点列表**：按重要性降序，每条附来源时间戳。
4. 完整转写文本（带时间戳），可导出分析报告（Markdown/JSON）。

全程本地运行，录音不出本机（除非用户主动配置 LLM API 提炼重点）。

## 2. 已确认的选型（用户拍板，不得更改）

| 项 | 决定 |
|---|---|
| ASR | 本地运行，三引擎：**CPU（默认）+ CUDA GPU + ROCm GPU**，设置中可切换，自动探测可用设备 |
| 重点提炼 | 默认离线算法（TextRank 类），设置中可选配 OpenAI 兼容 LLM API（用户自填 base_url / key / model），配置后自动改用 LLM |
| 界面 | 本地网页应用：Python 后端 + 浏览器页面，双击脚本启动 |

## 3. 技术栈

- Python 3.11+，FastAPI + uvicorn
- **CPU / CUDA 引擎**：`faster-whisper`（CTranslate2）。CPU 用 `compute_type=int8`，CUDA 用 `float16`，`word_timestamps=True`，`vad_filter=True`（Silero VAD，抑制静音幻觉）
- **ROCm 引擎**：`faster-whisper 不支持 ROCm`，此引擎用 PyTorch 路线：`transformers` 的 Whisper pipeline（`return_timestamps="word"`）跑在 ROCm 版 torch 上
  - 本机为 RX 9070 XT（gfx1201），Windows。ROCm torch 安装源参考：`https://stable.repo.amd.com/rocm/whl-next`（gfx1201 专用 wheel）。**实施时必须先验证该源当前可用的包名与版本，再写 requirements**；ROCm 依赖放入可选文件 `requirements-rocm.txt`，不装不影响 CPU/CUDA 运行
  - 用户 GPU 有慢性挂起史：ROCm/CUDA 引擎必须带看门狗超时（默认 10 分钟无进度即中止），失败自动回退 CPU 并在结果中提示
- 视频抽音频：ffmpeg（16kHz 单声道 wav）。启动时检测 ffmpeg，缺失则给出明确安装指引；支持 mp3/wav/m4a/flac/ogg/mp4/mkv/mov/webm
- 中文分词：jieba；离线重点：TextRank（可用 textrank4zh 或自实现，句子粒度）
- 前端：单页应用，FastAPI 托管静态文件。Vue 3（CDN 引入即可，不引入构建链）+ 原生 CSS。不引入 npm/打包工具
- 任务状态：内存任务表 + 后台线程即可（单机单用户），无需数据库；结果落盘为 JSON 便于重启后查看历史

## 4. 核心算法设计

### 4.1 废话判定

内置词表（可在设置中增删），分两类：

- **纯语气词**（无条件计为废话）：嗯、啊、呃、额、唔、emmm、um、uh、er
- **语境词**（需启发式判定，避免误伤正常用法）：这个、那个、就是、然后、其实、反正、对吧、就是说、怎么说呢、 basically、like、you know、actually、so、right

语境词判定规则（满足任一即计为废话）：
1. 连续重复（"这个这个"、"就是就是"）
2. 词后紧跟 ≥0.5s 停顿（用词级时间戳判断）
3. 位于句首且后接逗号/停顿
4. 同一语境词在 30 秒窗口内出现 ≥3 次（口头禅模式）

被判定为正常用法的语境词不计入。每条废话记录：词、start、end、所在句子。

### 4.2 统计口径

- 总词数：jieba 分词后的词元总数（英文按空格）
- 废话浓度（次数）= 废话词次数 / 总词数
- 废话浓度（时长）= Σ(废话词 end-start) / Σ(所有语音段时长)
- Top 10 按次数降序；同词合并统计

### 4.3 重点提炼

- **离线模式**：转写按句切分 → 过滤含高浓度废话的短句 → TextRank 句间相似度排序 → 叠加位置权重（开头/结尾段落加权）与关键词覆盖度 → 取 Top 8~15 句，附时间戳
- **LLM 模式**：将带时间戳的转写分块（每块 ~3000 字）发给 OpenAI 兼容接口，prompt 要求返回 JSON：`[{point, importance(1-5), timestamp_start, timestamp_end}]`，合并去重后按 importance 降序。LLM 调用失败自动回退离线模式并提示

## 5. API 设计

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | /api/jobs | multipart 上传文件，返回 job_id；表单字段 engine=cpu/cuda/rocm、model=small/medium/large-v3 |
| GET | /api/jobs/{id} | 状态与进度：queued→extracting→transcribing(百分比)→analyzing→done/failed |
| GET | /api/jobs/{id}/result | 完整结果 JSON（浓度、top10、重点、转写） |
| GET | /api/jobs | 历史任务列表 |
| GET | /api/audio/{id} | 供前端播放/跳转的音频流（支持 Range） |
| GET/PUT | /api/settings | 废话语料、LLM 配置、默认引擎/模型 |
| GET | /api/engines | 探测结果：cpu 恒可用；cuda 检测 ctranslate2 CUDA 可用性；rocm 检测 torch.version.hip |

进度上报：转写阶段按已处理音频时长/总时长估算百分比，前端 2s 轮询即可（不用 SSE/WebSocket，降低复杂度）。

## 6. 前端页面

单页三区块：
1. **上传区**：拖拽/点击上传，引擎与模型下拉（显示探测结果，不可用置灰），开始按钮
2. **进度区**：阶段文字 + 进度条；失败显示原因与"回退 CPU"提示
3. **结果区**：
   - 顶部两个浓度仪表盘（次数占比/时长占比）+ 音频播放器
   - 废话 Top 10 表格（词、次数、时长、例句时间戳按钮→播放器 seek）
   - 重点列表（重要性星级/分值、时间戳按钮）
   - 折叠的完整转写（废话词高亮）
   - 导出按钮（Markdown / JSON）

## 7. 目录结构

```
filler-detector/
├── PLAN.md                  # 本文档
├── run.cmd                  # 双击启动：建 venv(首次)→装依赖→起服务→开浏览器
├── requirements.txt         # 核心依赖（CPU/CUDA 可用）
├── requirements-rocm.txt    # 可选 ROCm torch
├── app/
│   ├── main.py              # FastAPI 入口 + 静态托管
│   ├── jobs.py              # 任务队列/状态/后台线程
│   ├── asr/
│   │   ├── base.py          # 引擎抽象：transcribe(path)->词级时间戳结果
│   │   ├── faster_engine.py # CPU/CUDA
│   │   └── rocm_engine.py   # transformers+ROCm torch（含看门狗）
│   ├── audio.py             # ffmpeg 抽音频/时长探测
│   ├── filler.py            # 词表+启发式判定+统计
│   ├── keypoints.py         # TextRank 离线 + LLM 可选
│   └── settings.py          # 配置持久化(settings.json)
├── static/                  # index.html + app.js + style.css (Vue3 CDN)
├── data/                    # 上传文件、结果 JSON、settings.json（gitignore）
└── tests/                   # pytest：filler 判定、统计口径、API 冒烟
```

## 8. 里程碑与验收（编程会话按此顺序执行）

- **M1 骨架**：目录、run.cmd、FastAPI 起服务、上传→ffmpeg 转 wav→返回时长。验收：上传 mp4 能拿到 16k wav 与正确时长。
- **M2 CPU 转写**：faster-whisper small int8 + VAD + 词级时间戳，任务进度可查。验收：一段 1 分钟中文录音转写基本准确，进度条走动。
- **M3 废话分析**：词表+启发式+双口径浓度+Top10。验收：构造含明显"嗯/这个这个/然后…"的测试音频，判定与计数正确；pytest 覆盖 4.1 全部规则。
- **M4 重点+前端**：离线 TextRank、完整单页 UI、播放器跳转、导出。验收：浏览器走通全流程（上传→进度→三类结果→点击时间戳跳转→导出）。
- **M5 GPU 引擎+LLM**：CUDA 引擎、ROCm 引擎（先验证 whl 源）、看门狗与 CPU 回退、LLM 配置与调用。验收：引擎探测接口正确；无 GPU 环境全部功能不受影响；LLM 失败能回退。

每个里程碑完成后运行测试再进入下一个。M5 的 ROCm 部分若安装源验证失败，如实报告并跳过，不得阻塞交付。

## 9. 智能体团队分工（编程会话组建）

| 角色 | 智能体 | 职责 |
|---|---|---|
| 技术总监 | fullstack-dev-group:钟帷幄 | 里程碑验收裁决、选型争议拍板 |
| Python 后端（主力） | fullstack-dev-group:彭森宇 | M1–M3、M5 全部后端 |
| 前端 | fullstack-dev-group:韦悠然 | M4 单页 UI（Vue3 CDN，遵守不引入构建链约束） |
| 测试 | fullstack-dev-group:查无漏 | 各里程碑用例与回归（重点：废话误判边界、GPU 回退） |

建议由主会话直接驱动 + 按里程碑委派上表角色，避免全员并行空转消耗 token。

## 10. 风险与对策

| 风险 | 对策 |
|---|---|
| ROCm torch wheel 源变动/安装失败 | 可选依赖隔离；M5 先验证再实施；失败不阻塞 |
| GPU 挂起（本机慢性问题） | 默认 CPU；GPU 引擎看门狗超时→自动回退 CPU |
| Whisper 静音段幻觉重复词 | vad_filter=True；结果侧过滤孤立重复段 |
| 语境词误判（"然后"正常承接） | 4.1 启发式 + 词表可配置 + 转写页高亮供人工核对 |
| 长音频（>1h）内存/耗时 | faster-whisper 流式分段；CPU 默认 small 模型，medium/large 可选并提示耗时 |
| 中文转写准确率 | 默认 small 起步，设置可切 medium；prompt 指定 language=zh（可设 auto） |