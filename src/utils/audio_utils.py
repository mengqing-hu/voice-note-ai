"""
utils/audio_utils.py
音频处理工具：时长获取、格式转换、文件切片、GPU 信息
依赖 ffmpeg（HPC 上通常已安装，或 module load ffmpeg）
"""

import subprocess
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

AUDIO_SUFFIXES = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac")


# ── 音频信息 ──────────────────────────────────────────────────

def get_duration(audio_path: str) -> float:
    """
    获取音频时长（秒），依赖 ffprobe。
    ffprobe 是 ffmpeg 的一部分，HPC 上通常可用。

    Returns:
        时长（秒），获取失败返回 -1.0
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        return float(result.stdout.strip())
    except Exception as e:
        logger.warning(f"获取音频时长失败: {e}")
        return -1.0


def format_duration(seconds: float) -> str:
    """
    将秒数格式化为 HH:MM:SS 字符串。

    Usage:
        format_duration(3723)  →  "01:02:03"
    """
    if seconds < 0:
        return "未知"
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def check_audio_file(audio_path: str) -> dict:
    """
    检查音频文件并返回基本信息。

    Returns:
        {"path": str, "exists": bool, "duration": float, "duration_str": str, "suffix": str}
    """
    p = Path(audio_path)
    duration = get_duration(audio_path) if p.exists() else -1.0
    return {
        "path": str(p),
        "exists": p.exists(),
        "duration": duration,
        "duration_str": format_duration(duration),
        "suffix": p.suffix.lower(),
    }


# ── 格式转换 ──────────────────────────────────────────────────

def convert_to_wav(input_path: str, output_path: str = None, sample_rate: int = 16000) -> str:
    """
    将任意音频转为 16kHz 单声道 WAV（Whisper 推荐输入格式）。
    依赖 ffmpeg。

    Args:
        input_path:  原始音频路径
        output_path: 输出路径，默认在同目录生成 {stem}_16k.wav
        sample_rate: 采样率，Whisper 要求 16000

    Returns:
        转换后的 WAV 文件路径

    Usage:
        wav_path = convert_to_wav("interview.m4a")
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = str(input_path.parent / f"{input_path.stem}_16k.wav")

    logger.info(f"转换音频格式: {input_path.name} → {Path(output_path).name}")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(input_path),
            "-ar", str(sample_rate),
            "-ac", "1",               # 单声道
            "-c:a", "pcm_s16le",      # 16-bit PCM
            output_path,
        ],
        check=True,
        capture_output=True,
    )
    logger.info(f"转换完成: {output_path}")
    return output_path


# ── 音频切片 ──────────────────────────────────────────────────

def split_audio(audio_path: str, chunk_minutes: int = 10, output_dir: str = None) -> list:
    """
    将长音频切分为固定时长的片段。
    用于：音频超长导致 Whisper 显存不足时的降级处理。

    Args:
        audio_path:    音频文件路径
        chunk_minutes: 每段时长（分钟），默认 10 分钟
        output_dir:    切片输出目录，默认在原文件同目录下新建子文件夹

    Returns:
        切片文件路径列表，按顺序排列

    Usage:
        chunks = split_audio("data/raw/interview_001.wav", chunk_minutes=10)
        for chunk in chunks:
            result = transcribe_audio(chunk)
    """
    audio_path = Path(audio_path)
    if output_dir is None:
        output_dir = audio_path.parent / f"{audio_path.stem}_chunks"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    chunk_seconds = chunk_minutes * 60
    output_pattern = str(Path(output_dir) / f"{audio_path.stem}_chunk_%03d.wav")

    logger.info(f"切分音频: {audio_path.name}，每段 {chunk_minutes} 分钟")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(audio_path),
            "-f", "segment",
            "-segment_time", str(chunk_seconds),
            "-c", "copy",
            output_pattern,
        ],
        check=True,
        capture_output=True,
    )

    chunks = sorted(Path(output_dir).glob(f"{audio_path.stem}_chunk_*.wav"))
    logger.info(f"切分完成，共 {len(chunks)} 段")
    return [str(c) for c in chunks]


# GPU 相关功能已迁移至 src/utils/device.py
# 请使用: from src.utils.device import get_device, get_recommended_whisper_model