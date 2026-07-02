# voice-note-ai

`voice-note-ai` 是一个面向中文访谈、面试录音的语音分析流水线。项目会把音频转成带时间戳的文本，做说话人分离，将转录内容整理成结构化对话，并调用大语言模型提取面试官提出的问题，最终生成 JSON 和 Markdown 报告。

## 功能特性

- 使用 Whisper 进行语音转录，支持自动选择 GPU/CPU，并根据显存自动降级模型。
- 使用 `pyannote.audio` 进行说话人分离，默认按 2 位说话人处理。
- 将 Whisper 片段与说话人时间段对齐，生成结构化对话。
- 支持说话人重命名，例如 `SPEAKER_00 -> 面试官`、`SPEAKER_01 -> 候选人`。
- 调用 LLM 从面试官发言中提取完整问题列表，并生成面试总结。
- 支持单文件处理、批量处理、分步骤调用。
- 支持文件级缓存和断点续跑，避免重复跑耗时步骤。

## 处理流程

```text
音频文件
  -> Step 1: Whisper 转录
  -> Step 2: pyannote 说话人分离
  -> Step 3: 转录与说话人结果合并
  -> Step 4: LLM 提取面试问题
  -> JSON / Markdown 报告
```

## 项目结构

```text
voice-note-ai/
├── prompts/
│   └── extract_questions.txt          # LLM 提问抽取 Prompt 模板
├── src/
│   ├── pipeline/
│   │   ├── runner.py                  # 完整流水线入口
│   │   ├── step1_transcribe.py        # Whisper 转录
│   │   ├── step2_diarize.py           # 说话人分离
│   │   ├── step3_postprocess.py       # 对齐并生成结构化对话
│   │   └── step4_extract_questions.py # LLM 提取问题与报告生成
│   └── utils/
│       ├── audio_utils.py             # 音频检查、格式转换、切片
│       ├── cache.py                   # 步骤缓存
│       ├── device.py                  # GPU/CPU 与显存检测
│       ├── file_utils.py              # 文件读写与目录管理
│       ├── llm_client.py              # Groq/Gemini 客户端封装
│       ├── logger.py                  # 统一日志
│       └── text_utils.py              # 文本清洗、时间戳、分块
├── requirements.txt
├── .env.example
└── README.md
```

运行后会自动创建以下数据目录：

```text
data/
├── raw/          # 原始音频文件
├── transcripts/  # 转录与说话人分离 JSON
├── processed/    # 结构化对话 JSON
├── outputs/      # 问题 JSON 与 Markdown 报告
└── cache/        # 断点续跑缓存

logs/
└── run.log       # 运行日志
```

## 环境要求

建议使用 Python 3.10 或 3.11，并优先在有 NVIDIA GPU 的环境运行。CPU 也可以运行，但 Whisper 和说话人分离会明显更慢。

系统依赖：

- `ffmpeg` / `ffprobe`：用于获取音频时长和音频格式转换。
- CUDA 与可用 GPU：可选，但强烈推荐。
- Hugging Face 账号和 Token：用于加载 `pyannote/speaker-diarization-3.1`。
- Groq 或 Gemini API Key：用于 LLM 提取问题。

安装 `ffmpeg` 示例：

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt-get update
sudo apt-get install -y ffmpeg
```

如果在 HPC 环境中运行，通常可以使用：

```bash
module load ffmpeg
```

## 安装

进入项目目录：

```bash
cd voice-note-ai
```

创建虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
```

安装依赖：

```bash
pip install -r requirements.txt
```

当前代码还会直接 import 一些未写入 `requirements.txt` 的运行依赖，建议额外安装：

```bash
pip install openai-whisper pyannote.audio torch librosa groq google-generativeai
```

如需安装指定 CUDA 版本的 PyTorch，请以 PyTorch 官网给出的命令为准。

## 配置环境变量

复制环境变量模板：

```bash
cp .env.example .env
```

按需填写：

```dotenv
# Hugging Face Token，用于 pyannote 说话人分离
HF_TOKEN=your_huggingface_token

# 当前 runner.py 会检查 GEMINI_API_KEY 是否存在
GEMINI_API_KEY=your_llm_api_key

# 如果你改为显式使用 Groq，也可以保留 Groq Key
GROQ_API_KEY=your_groq_api_key
```

注意事项：

- 使用 `pyannote/speaker-diarization-3.1` 前，需要在 Hugging Face 上登录并接受该模型相关使用条款。
- 当前 `src/pipeline/runner.py` 会检查 `GEMINI_API_KEY` 和 `HF_TOKEN`。
- 当前 `LLMClient` 默认 `provider="groq"`，而 `runner.py` 传入的是 `GEMINI_API_KEY`。如果你要直接跑完整流水线，请确保传入的 Key 与实际 provider 匹配，或在 `runner.py` 中初始化 `LLMClient` 时显式设置 `provider="gemini"` / `provider="groq"`。

## 快速开始

把音频放到 `data/raw/`：

```text
data/raw/interview_001.wav
```

运行单个音频：

```python
from src.pipeline.runner import run

result = run(
    audio_path="data/raw/interview_001.wav",
    speaker_mapping={
        "SPEAKER_00": "面试官",
        "SPEAKER_01": "候选人",
    },
    model_size="large-v3",
    language="zh",
    interviewer_label="面试官",
)

print(result["result"]["total_questions"])
```

输出报告默认生成在：

```text
data/outputs/interview_001_questions.json
data/outputs/interview_001_report.md
```

## 批量处理

处理 `data/raw/` 下所有支持的音频文件：

```python
from src.pipeline.runner import run_batch

results = run_batch(
    audio_dir="data/raw",
    speaker_mapping={
        "SPEAKER_00": "面试官",
        "SPEAKER_01": "候选人",
    },
    language="zh",
)

print(f"共处理 {len(results)} 个文件")
```

批量模式会逐个处理音频。单个文件失败时，错误会记录到结果列表中，不影响后续文件继续运行。

## 分步骤使用

如果需要调试某一步，可以分别调用各模块。

### 1. 语音转录

```python
from src.pipeline.step1_transcribe import transcribe

transcript = transcribe(
    audio_path="data/raw/interview_001.wav",
    model_size="large-v3",
    language="zh",
)
```

输出：

```text
data/transcripts/interview_001_transcript.json
```

### 2. 说话人分离

```python
import os
from dotenv import load_dotenv
from src.pipeline.step2_diarize import diarize

load_dotenv()

diarization = diarize(
    audio_path="data/raw/interview_001.wav",
    hf_token=os.getenv("HF_TOKEN"),
    num_speakers=2,
)
```

输出：

```text
data/transcripts/interview_001_diarization.json
```

### 3. 生成结构化对话

```python
from src.pipeline.step3_postprocess import postprocess, preview_dialog, get_speaker_stats

dialog = postprocess(
    transcript=transcript,
    diarization=diarization,
    speaker_mapping={
        "SPEAKER_00": "面试官",
        "SPEAKER_01": "候选人",
    },
)

preview_dialog(dialog, n=8)
print(get_speaker_stats(dialog))
```

输出：

```text
data/processed/interview_001_dialog.json
```

### 4. 提取面试问题

```python
from src.pipeline.step4_extract_questions import extract_questions
from src.utils.llm_client import LLMClient

client = LLMClient(
    api_key="your_api_key",
    provider="gemini",
    model="gemini-2.0-flash",
)

result = extract_questions(
    dialog=dialog,
    llm_client=client,
    audio_path="data/raw/interview_001.wav",
    interviewer_label="面试官",
)

print(result["questions"])
```

输出：

```text
data/outputs/interview_001_questions.json
data/outputs/interview_001_report.md
```

## 支持的音频格式

批量处理默认查找：

- `.wav`
- `.mp3`
- `.m4a`
- `.flac`

工具函数中也定义了对 `.ogg`、`.aac` 的支持。非 WAV 音频在 Whisper 转录前会自动转换为 16kHz 单声道 WAV：

```text
原文件: interview_001.m4a
转换后: interview_001_16k.wav
```

## 缓存与强制重跑

项目有两层跳过机制：

- 各步骤会检查对应输出文件是否已存在。
- 完整流水线会使用 `data/cache/` 中基于音频 MD5 的步骤缓存。

默认情况下，已完成步骤会被跳过。强制重跑：

```python
from src.pipeline.runner import run

result = run(
    audio_path="data/raw/interview_001.wav",
    force=True,
)
```

禁用缓存：

```python
result = run(
    audio_path="data/raw/interview_001.wav",
    use_cache=False,
)
```

手动清理缓存：

```python
from src.utils.cache import StepCache

cache = StepCache("data/cache")
cache.clear()              # 清理全部缓存
cache.clear("transcribe")  # 只清理转录缓存
```

## 输出格式

### 转录结果

`data/transcripts/*_transcript.json`：

```json
{
  "audio_file": "data/raw/interview_001.wav",
  "language": "zh",
  "model": "large-v3",
  "duration_seconds": 1234.56,
  "full_text": "...",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 4.2,
      "text": "请先介绍一下你自己"
    }
  ]
}
```

### 结构化对话

`data/processed/*_dialog.json`：

```json
[
  {
    "speaker": "面试官",
    "start": 0.0,
    "end": 4.2,
    "text": "请先介绍一下你自己"
  },
  {
    "speaker": "候选人",
    "start": 4.8,
    "end": 30.5,
    "text": "好的，我叫..."
  }
]
```

### 问题抽取结果

`data/outputs/*_questions.json`：

```json
{
  "audio_stem": "interview_001",
  "total_questions": 3,
  "questions": [
    {
      "index": 1,
      "text": "请先介绍一下你自己",
      "timestamp": "[00:00:01]"
    }
  ],
  "summary": "本次面试主要考察候选人的项目经验、技术深度和问题解决能力。"
}
```

Markdown 报告：

```text
data/outputs/interview_001_report.md
```

## Prompt 定制

LLM 提问抽取模板位于：

```text
prompts/extract_questions.txt
```

模板中可以使用变量：

- `{dialog_text}`：输入给 LLM 的对话文本或面试官发言文本。
- `{interviewer_label}`：面试官标签。

修改该文件后，下一次运行 Step 4 会使用新的 Prompt。

## 常见问题

### 1. `ffprobe` 或 `ffmpeg` 找不到

说明系统没有安装 `ffmpeg`，或命令不在 `PATH` 中。安装后重新打开终端，运行：

```bash
ffmpeg -version
ffprobe -version
```

确认能正常输出版本信息。

### 2. `pyannote.audio` 加载模型失败

常见原因：

- 没有设置 `HF_TOKEN`。
- Hugging Face Token 没有读取模型的权限。
- 没有在 Hugging Face 页面接受 `pyannote/speaker-diarization-3.1` 的使用条款。
- 网络环境无法访问 Hugging Face。

### 3. GPU 显存不足

`src/utils/device.py` 会根据显存自动降级 Whisper 模型。也可以手动指定较小模型：

```python
run(
    audio_path="data/raw/interview_001.wav",
    model_size="medium",
)
```

可选模型包括：

- `large-v3`
- `large-v2`
- `medium`
- `small`
- `base`
- `tiny`

### 4. LLM API 配额不足

Step 4 捕获到配额类错误时，会尝试切换到演示模式生成示例输出。演示模式主要用于展示流程，不建议作为正式结果使用。

### 5. 说话人映射反了

先预览结构化对话：

```python
from src.pipeline.step3_postprocess import preview_dialog

preview_dialog(dialog, n=10)
```

如果发现 `SPEAKER_00` 和 `SPEAKER_01` 对调，交换 `speaker_mapping` 后设置 `force=True` 重新生成：

```python
run(
    audio_path="data/raw/interview_001.wav",
    speaker_mapping={
        "SPEAKER_00": "候选人",
        "SPEAKER_01": "面试官",
    },
    interviewer_label="面试官",
    force=True,
)
```

## 开发建议

- 优先使用 `run()` 验证端到端流程。
- 若结果不符合预期，按 Step 1 到 Step 4 分步排查。
- 修改 Prompt 后，只需要强制重跑 Step 4 或删除对应 `data/outputs/*_questions.json`。
- 对长音频可先用 `src.utils.audio_utils.split_audio()` 切片。
- 运行日志会写入 `logs/run.log`，排查问题时优先查看该文件。

## 当前限制

- 项目目前没有命令行 CLI，主要通过 Python / Notebook 调用。
- 说话人数量在完整流水线中默认由 `speaker_mapping` 长度决定；未传入时默认按 2 人处理并自动命名。
- `run_pipeline()` 接收 `gemini_api_key` 和 `hf_token` 参数，但当前实现仍通过环境变量读取。
- `requirements.txt` 与代码实际依赖不完全一致，首次部署建议参考本文的额外依赖安装命令。
