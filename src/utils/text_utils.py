"""
utils/text_utils.py
文本处理工具：转录清洗、token 估算、对话格式化、超长文本分块
"""

import re
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── 转录文本清洗 ──────────────────────────────────────────────

# Whisper 中文常见幻觉词（无实际内容、重复出现的片段）
_HALLUCINATION_PATTERNS = [
    r"字幕由.{0,20}提供",
    r"翻译.{0,10}字幕",
    r"感谢(您的)?观看",
    r"请订阅.{0,15}频道",
    r"点击.{0,10}订阅",
    r"如果(你|您)喜欢.{0,20}",
    r"[\(（].*?音乐.*?[\)）]",
    r"[\(（].*?掌声.*?[\)）]",
]


def clean_text(text: str) -> str:
    """
    清洗 Whisper 转录文本：
    - 去除首尾空白
    - 合并多余空格和换行
    - 过滤常见幻觉词

    Usage:
        cleaned = clean_text(segment["text"])
    """
    if not text:
        return ""

    text = text.strip()
    text = re.sub(r"\s+", " ", text)

    for pattern in _HALLUCINATION_PATTERNS:
        text = re.sub(pattern, "", text)

    return text.strip()


def clean_segments(segments: list) -> list:
    """
    批量清洗 Whisper segments 列表，过滤空段。

    Usage:
        segments = clean_segments(transcript["segments"])
    """
    cleaned = []
    for seg in segments:
        text = clean_text(seg.get("text", ""))
        if text:
            cleaned.append({**seg, "text": text})
    return cleaned


# ── Token 估算 ────────────────────────────────────────────────

def estimate_tokens(text: str, lang: str = "zh") -> int:
    """
    粗略估算文本 token 数，用于判断是否超出模型上下文限制。
    中文：1字 ≈ 1.5 token
    英文：1词 ≈ 1.3 token

    Usage:
        tokens = estimate_tokens(dialog_text)
        if tokens > 900_000:
            logger.warning("接近 Gemini 上下文限制，建议分块")
    """
    if lang == "zh":
        return int(len(text) * 1.5)
    else:
        return int(len(text.split()) * 1.3)


def check_context_limit(text: str, model: str = "gemini-2.0-flash", lang: str = "zh") -> dict:
    """
    检查文本是否超出模型上下文限制，返回详细报告。

    模型上下文限制（token）：
        gemini-2.0-flash:  1,000,000
        gemini-1.5-pro:    2,000,000
        gpt-4o:            128,000
        llama-3.3-70b:     128,000

    Returns:
        {"tokens": int, "limit": int, "within_limit": bool, "usage_pct": float}
    """
    limits = {
        "gemini-2.0-flash": 1_000_000,
        "gemini-1.5-pro":   2_000_000,
        "gpt-4o":             128_000,
        "llama-3.3-70b":      128_000,
    }
    limit = limits.get(model, 128_000)
    tokens = estimate_tokens(text, lang)
    return {
        "tokens": tokens,
        "limit": limit,
        "within_limit": tokens <= limit,
        "usage_pct": round(tokens / limit * 100, 1),
    }


# ── 对话格式化 ────────────────────────────────────────────────

def seconds_to_timestamp(seconds: float) -> str:
    """
    将秒数转为 [HH:MM:SS] 格式。

    Usage:
        seconds_to_timestamp(3723)  →  "[01:02:03]"
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"[{h:02d}:{m:02d}:{s:02d}]"


def dialog_to_text(dialog: list, with_timestamp: bool = True) -> str:
    """
    将结构化对话列表转为纯文本，喂给 LLM 前调用。

    Args:
        dialog:         postprocess.py 输出的对话列表
        with_timestamp: 是否包含时间戳（默认包含，方便 LLM 定位问题位置）

    Returns:
        格式化文本，每行一个说话轮次

    Usage:
        text = dialog_to_text(dialog)
        # → "[00:00:05] 面试官: 请介绍一下你自己..."
        # → "[00:01:23] 候选人: 您好，我叫..."
    """
    lines = []
    for item in dialog:
        if with_timestamp:
            ts = seconds_to_timestamp(item.get("start", 0))
            lines.append(f"{ts} {item['speaker']}: {item['text']}")
        else:
            lines.append(f"{item['speaker']}: {item['text']}")
    return "\n".join(lines)


# ── 超长文本分块 ──────────────────────────────────────────────

def chunk_dialog(dialog: list, max_tokens: int = 80_000, lang: str = "zh") -> list:
    """
    将对话列表按 token 上限切分为多个块。
    按说话轮次切割，不会截断单条发言。
    用于：音频超长或模型上下文限制较小时的降级处理。

    Args:
        dialog:     结构化对话列表
        max_tokens: 每块最大 token 数，默认 80,000（留余量）
        lang:       语言

    Returns:
        二维列表 [[dialog_item, ...], [...]]

    Usage:
        chunks = chunk_dialog(dialog, max_tokens=80_000)
        results = [extract_questions(dialog_to_text(c), ...) for c in chunks]
    """
    if not dialog:
        return []

    chunks, current, current_tokens = [], [], 0

    for item in dialog:
        item_tokens = estimate_tokens(item["text"], lang)
        if current and current_tokens + item_tokens > max_tokens:
            chunks.append(current)
            current, current_tokens = [], 0
        current.append(item)
        current_tokens += item_tokens

    if current:
        chunks.append(current)

    if len(chunks) > 1:
        logger.info(f"对话已分为 {len(chunks)} 块（每块上限 {max_tokens} tokens）")

    return chunks


def truncate_text(text: str, max_chars: int = 50_000) -> str:
    """
    截断文本到指定字符数，末尾加提示。
    用于调试或日志输出时避免打印过长内容。

    Usage:
        logger.info(f"对话预览:\n{truncate_text(dialog_text, 500)}")
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [已截断，原文 {len(text)} 字]"