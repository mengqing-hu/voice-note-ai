"""
utils/file_utils.py
文件与路径工具：目录创建、JSON 读写、文件检查、路径解析
所有业务模块的文件操作都通过这里，不直接用 os/json
"""

import json
import os
import hashlib
from pathlib import Path
from typing import Any


# ── 目录 ────────────────────────────────────────────────────

def ensure_dirs(*dirs: str) -> None:
    """
    批量创建目录，已存在则跳过。

    Usage:
        ensure_dirs("data/raw", "data/transcripts", "data/outputs")
    """
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def ensure_project_dirs() -> None:
    """一键创建项目所需的全部数据目录"""
    ensure_dirs(
        "data/raw",
        "data/transcripts",
        "data/processed",
        "data/outputs",
        "data/cache",
        "logs",
    )


# ── 文件检查 ─────────────────────────────────────────────────

def file_exists(path: str, min_bytes: int = 10) -> bool:
    """
    检查文件是否存在且不为空。
    min_bytes: 低于此字节数视为无效文件（默认 10 字节）

    Usage:
        if file_exists("data/transcripts/interview_001_transcript.json"):
            # 跳过，直接加载
    """
    p = Path(path)
    return p.exists() and p.stat().st_size >= min_bytes


def get_stem(path: str) -> str:
    """
    获取文件名（不含扩展名），用于生成输出文件名。

    Usage:
        get_stem("data/raw/interview_001.wav")  →  "interview_001"
    """
    return Path(path).stem


def get_md5(path: str) -> str:
    """
    计算文件 MD5，用于缓存 key 和去重。

    Usage:
        md5 = get_md5("data/raw/interview_001.wav")
    """
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def list_files(directory: str, suffixes: tuple = ()) -> list:
    """
    列出目录下指定后缀的文件，递归搜索，结果排序。

    Usage:
        audio_files = list_files("data/raw", suffixes=(".wav", ".mp3", ".m4a"))
        json_files  = list_files("data/transcripts", suffixes=(".json",))
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    if suffixes:
        return sorted([
            str(p) for p in directory.rglob("*")
            if p.suffix.lower() in suffixes
        ])
    return sorted([str(p) for p in directory.rglob("*") if p.is_file()])


# ── JSON 读写 ─────────────────────────────────────────────────

def save_json(data: Any, path: str, indent: int = 2) -> str:
    """
    保存数据为 JSON 文件，自动创建父目录。

    Returns:
        保存的文件路径（方便链式调用后打印）

    Usage:
        save_json(transcript, "data/transcripts/interview_001_transcript.json")
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    return path


def load_json(path: str) -> Any:
    """
    读取 JSON 文件。

    Usage:
        transcript = load_json("data/transcripts/interview_001_transcript.json")
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── 文本文件读写 ──────────────────────────────────────────────

def save_text(text: str, path: str) -> str:
    """
    保存纯文本文件，自动创建父目录。

    Usage:
        save_text(report_md, "data/outputs/interview_001_report.md")
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def load_text(path: str) -> str:
    """
    读取纯文本文件（如 Prompt 模板）。

    Usage:
        prompt_template = load_text("prompts/extract_questions.txt")
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ── 路径构建 ──────────────────────────────────────────────────

def build_output_path(audio_path: str, stage: str, output_dir: str, ext: str = ".json") -> str:
    """
    根据音频文件路径和处理阶段，自动生成输出文件路径。

    Args:
        audio_path: 原始音频路径，如 "data/raw/interview_001.wav"
        stage:      处理阶段标识，如 "transcript" / "diarization" / "dialog" / "questions"
        output_dir: 输出目录
        ext:        文件后缀

    Returns:
        如 "data/transcripts/interview_001_transcript.json"

    Usage:
        path = build_output_path("data/raw/interview_001.wav", "transcript", "data/transcripts")
    """
    stem = get_stem(audio_path)
    filename = f"{stem}_{stage}{ext}"
    return str(Path(output_dir) / filename)