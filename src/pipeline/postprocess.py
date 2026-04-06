"""
pipeline/postprocess.py
模块3：转录与说话人合并，生成结构化对话
- 输入: Whisper transcript dict + diarization list
- 输出: 结构化对话 JSON [{speaker, start, end, text}, ...]
"""

from src.utils.logger import get_logger
from src.utils.file_utils import file_exists, save_json, load_json, build_output_path, ensure_dirs
from src.utils.text_utils import dialog_to_text

logger = get_logger(__name__)

OUTPUT_DIR = "data/processed"


def postprocess(
    transcript: dict,
    diarization: list,
    speaker_mapping: dict = None,
    output_dir: str = OUTPUT_DIR,
    force: bool = False,
) -> list:
    """
    合并 Whisper 转录与说话人分离结果，生成结构化对话列表。

    策略：对每个 Whisper segment，找时间重叠最多的说话人标签。
    合并连续同一说话人的相邻片段（间隔 < 1.5s），减少碎片。

    Args:
        transcript:      transcribe() 的返回值
        diarization:     diarize() 的返回值
        speaker_mapping: 说话人重命名，如 {"SPEAKER_00": "面试官", "SPEAKER_01": "候选人"}
                         None = 保留原始标签
        output_dir:      结果保存目录
        force:           True = 强制重新处理

    Returns:
        结构化对话列表:
        [
            {"speaker": "面试官", "start": 0.5, "end": 8.2, "text": "请先介绍一下你自己"},
            {"speaker": "候选人", "start": 9.1, "end": 45.3, "text": "好的，我叫..."},
            ...
        ]

    Usage:
        from src.pipeline.postprocess import postprocess
        dialog = postprocess(transcript, diarization,
                             speaker_mapping={"SPEAKER_00": "面试官", "SPEAKER_01": "候选人"})
    """
    audio_path = transcript.get("audio_file", "unknown")
    output_path = build_output_path(audio_path, "dialog", output_dir, ".json")
    ensure_dirs(output_dir)

    # ── 检查是否已有结果 ──────────────────────────────────────
    if not force and file_exists(output_path):
        logger.info(f"已存在结构化对话，跳过（force=False）: {output_path}")
        return load_json(output_path)

    # ── 步骤1：每个 segment 匹配说话人 ───────────────────────
    dialog = []
    for seg in transcript.get("segments", []):
        speaker = _find_speaker(seg["start"], seg["end"], diarization)
        dialog.append({
            "speaker": speaker,
            "start":   seg["start"],
            "end":     seg["end"],
            "text":    seg["text"],
        })

    # ── 步骤2：合并连续同一说话人的片段 ──────────────────────
    merged = _merge_consecutive(dialog, gap_threshold=1.5)

    # ── 步骤3：重命名说话人 ───────────────────────────────────
    if speaker_mapping:
        merged = _rename_speakers(merged, speaker_mapping)
        logger.info(f"说话人重命名: {speaker_mapping}")

    save_json(merged, output_path)
    logger.info(
        f"对话结构化完成 | 轮次: {len(merged)} | 已保存: {output_path}"
    )
    return merged


# ── 内部工具函数 ──────────────────────────────────────────────

def _find_speaker(seg_start: float, seg_end: float, diarization: list) -> str:
    """找与 segment 时间重叠最多的说话人"""
    best_speaker = "UNKNOWN"
    best_overlap = 0.0
    for d in diarization:
        overlap = min(seg_end, d["end"]) - max(seg_start, d["start"])
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = d["speaker"]
    return best_speaker


def _merge_consecutive(dialog: list, gap_threshold: float = 1.5) -> list:
    """合并连续同一说话人且间隔小于阈值的片段"""
    if not dialog:
        return []
    merged = [dict(dialog[0])]
    for item in dialog[1:]:
        last = merged[-1]
        gap = item["start"] - last["end"]
        if last["speaker"] == item["speaker"] and gap <= gap_threshold:
            last["end"]   = item["end"]
            last["text"] += item["text"]
        else:
            merged.append(dict(item))
    return merged


def _rename_speakers(dialog: list, mapping: dict) -> list:
    """将说话人标签按 mapping 重命名"""
    for item in dialog:
        item["speaker"] = mapping.get(item["speaker"], item["speaker"])
    return dialog


def rename_speakers(dialog: list, mapping: dict) -> list:
    """
    将说话人标签按 mapping 重命名（公开接口）。

    Args:
        dialog:  结构化对话列表
        mapping: 说话人重命名字典，如 {"SPEAKER_00": "面试官", "SPEAKER_01": "候选人"}

    Returns:
        重命名后的对话列表

    Usage:
        from src.pipeline.postprocess import rename_speakers
        dialog = rename_speakers(dialog, mapping={"SPEAKER_00": "面试官", "SPEAKER_01": "候选人"})
    """
    return _rename_speakers(dialog, mapping)


def merge_transcript_diarization(
    transcript: dict,
    diarization: list,
    audio_stem: str = "",
    output_dir: str = OUTPUT_DIR,
    force: bool = False,
) -> list:
    """
    合并 Whisper 转录与说话人分离结果，生成结构化对话列表（不重命名说话人）。

    这是 postprocess() 的便利封装，不执行说话人重命名。

    Args:
        transcript:  transcribe() 的返回值
        diarization: diarize() 的返回值
        audio_stem:  音频文件名干（用于日志，可选）
        output_dir:  结果保存目录
        force:       True = 强制重新处理

    Returns:
        结构化对话列表（保留原始说话人标签）

    Usage:
        from src.pipeline.postprocess import merge_transcript_diarization
        dialog = merge_transcript_diarization(transcript, diarization)
    """
    return postprocess(transcript, diarization, speaker_mapping=None, output_dir=output_dir, force=force)


# ── 工具函数：对话预览 ────────────────────────────────────────

def preview_dialog(dialog: list, n: int = 5) -> None:
    """
    在 notebook 中快速预览前 n 轮对话，辅助判断说话人分配是否正确。

    Usage:
        preview_dialog(dialog, n=8)
    """
    for item in dialog[:n]:
        ts = f"[{item['start']:.0f}s]"
        text_preview = item["text"][:60] + ("..." if len(item["text"]) > 60 else "")
        print(f"{ts:>8}  {item['speaker']:<6}  {text_preview}")


def get_speaker_stats(dialog: list) -> dict:
    """
    统计每位说话人的发言时长和轮次，辅助验证说话人分配。

    Returns:
        {"面试官": {"turns": 12, "duration": 180.5}, ...}

    Usage:
        stats = get_speaker_stats(dialog)
        for speaker, s in stats.items():
            print(f"{speaker}: {s['turns']} 轮, {s['duration']:.0f}s")
    """
    stats = {}
    for item in dialog:
        sp = item["speaker"]
        if sp not in stats:
            stats[sp] = {"turns": 0, "duration": 0.0}
        stats[sp]["turns"]    += 1
        stats[sp]["duration"] += item["end"] - item["start"]
    return stats