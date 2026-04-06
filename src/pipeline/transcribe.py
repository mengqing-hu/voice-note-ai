"""
pipeline/transcribe.py
模块1：Whisper 语音转录
- 输入: 音频文件路径
- 输出: 带时间戳的转录结果，保存到 data/transcripts/
"""

import whisper
from pathlib import Path

from src.utils.logger import get_logger, Timer
from src.utils.file_utils import file_exists, save_json, load_json, build_output_path, ensure_dirs
from src.utils.device import get_device_str, validate_whisper_model
from src.utils.audio_utils import check_audio_file, convert_to_wav
from src.utils.text_utils import clean_segments

logger = get_logger(__name__)

OUTPUT_DIR = "data/transcripts"


def transcribe(
    audio_path: str,
    model_size: str = "large-v3",
    language: str = "zh",
    output_dir: str = OUTPUT_DIR,
    force: bool = False,
) -> dict:
    """
    使用 Whisper 对音频文件进行语音转录。

    Args:
        audio_path:  音频文件路径
        model_size:  Whisper 模型大小，会自动根据显存验证并降级
        language:    音频语言，中文填 "zh"
        output_dir:  转录结果保存目录
        force:       True = 忽略已有结果强制重新转录

    Returns:
        {
            "audio_file":        str,
            "language":          str,
            "duration_seconds":  float,
            "full_text":         str,
            "segments": [
                {"id": int, "start": float, "end": float, "text": str},
                ...
            ]
        }

    Usage:
        from src.pipeline.transcribe import transcribe
        transcript = transcribe("data/raw/interview_001.wav")
    """
    audio_path = str(audio_path)
    output_path = build_output_path(audio_path, "transcript", output_dir, ".json")
    ensure_dirs(output_dir)

    # ── 检查是否已有结果 ──────────────────────────────────────
    if not force and file_exists(output_path):
        logger.info(f"已存在转录结果，跳过（force=False）: {output_path}")
        return load_json(output_path)

    # ── 验证音频文件 ──────────────────────────────────────────
    info = check_audio_file(audio_path)
    if not info["exists"]:
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    logger.info(f"音频文件: {Path(audio_path).name}  时长: {info['duration_str']}")

    # ── 非 WAV 格式自动转换 ───────────────────────────────────
    if info["suffix"] not in (".wav",):
        logger.info(f"非 WAV 格式（{info['suffix']}），转换为 16kHz WAV...")
        audio_path = convert_to_wav(audio_path)

    # ── 验证模型与设备 ────────────────────────────────────────
    actual_model = validate_whisper_model(model_size)
    device = get_device_str()

    # ── 转录 ──────────────────────────────────────────────────
    with Timer(f"Whisper {actual_model} 转录", logger):
        logger.info(f"加载模型: whisper-{actual_model}  设备: {device}")
        model = whisper.load_model(actual_model, device=device)

        logger.info("开始转录...")
        result = model.transcribe(
            audio_path,
            language=language,
            word_timestamps=True,
            verbose=False,
            fp16=(device == "cuda"),
        )

    # ── 整理输出 ──────────────────────────────────────────────
    raw_segments = result.get("segments", [])
    cleaned_segments = clean_segments([
        {
            "id":    seg["id"],
            "start": round(seg["start"], 2),
            "end":   round(seg["end"], 2),
            "text":  seg["text"].strip(),
        }
        for seg in raw_segments
    ])

    duration = cleaned_segments[-1]["end"] if cleaned_segments else 0.0

    output = {
        "audio_file":       audio_path,
        "language":         language,
        "model":            actual_model,
        "duration_seconds": duration,
        "full_text":        result["text"].strip(),
        "segments":         cleaned_segments,
    }

    save_json(output, output_path)
    logger.info(
        f"转录完成 | 片段数: {len(cleaned_segments)} | "
        f"时长: {duration:.0f}s | 已保存: {output_path}"
    )
    return output