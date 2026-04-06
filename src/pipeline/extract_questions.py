"""
pipeline/extract_questions.py
模块4：调用 LLM 从结构化对话中提取面试问题
- 输入: 结构化对话列表
- 输出: 按顺序编号的面试问题列表 + 面试总结，保存为 JSON 和 Markdown
"""

import re
from pathlib import Path

from src.utils.logger import get_logger, Timer
from src.utils.file_utils import (
    file_exists, save_json, load_json,
    save_text, load_text, build_output_path, ensure_dirs,
)
from src.utils.text_utils import (
    dialog_to_text, estimate_tokens,
    check_context_limit, chunk_dialog, truncate_text,
)
from src.utils.llm_client import LLMClient

logger = get_logger(__name__)

OUTPUT_DIR    = "data/outputs"
PROMPT_FILE   = "prompts/extract_questions.txt"
DEFAULT_MODEL = "gemini-2.0-flash"


def extract_questions(
    dialog: list,
    llm_client: LLMClient = None,
    audio_path: str = "interview",
    interviewer_label: str = "面试官",
    output_dir: str = OUTPUT_DIR,
    force: bool = False,
    use_demo: bool = False,
) -> dict:
    """
    使用 LLM 从结构化对话中按顺序提取面试问题。

    Args:
        dialog:            postprocess() 输出的对话列表
        llm_client:        已初始化的 LLMClient 实例（use_demo=True 时可为 None）
        audio_path:        原始音频路径（用于生成输出文件名）
        interviewer_label: 面试官的说话人标签，与 postprocess 的 mapping 保持一致
        output_dir:        结果保存目录
        force:             True = 强制重新提取
        use_demo:          True = 使用演示模式（不调用 LLM，生成示例问题）

    Returns:
        {
            "audio_stem":      str,
            "total_questions": int,
            "questions": [
                {"index": 1, "text": "请先介绍一下你自己", "timestamp": "[00:00:05]"},
                ...
            ],
            "summary": str,
        }

    Usage:
        from src.pipeline.extract_questions import extract_questions
        from src.utils.llm_client import build_client_from_env

        client = build_client_from_env()
        result = extract_questions(dialog, client, audio_path="data/raw/interview_001.wav")
    """
    audio_stem  = Path(audio_path).stem
    output_json = build_output_path(audio_path, "questions", output_dir, ".json")
    output_md   = build_output_path(audio_path, "report",    output_dir, ".md")
    ensure_dirs(output_dir)

    # ── 检查是否已有结果 ──────────────────────────────────────
    if not force and file_exists(output_json):
        logger.info(f"已存在提取结果，跳过（force=False）: {output_json}")
        return load_json(output_json)

    # ── 第一步：从对话中提取面试官的话 ────────────────────────
    logger.info("第一步：提取面试官的话...")
    interviewer_turns = _extract_interviewer_turns(dialog, interviewer_label)
    interviewer_text = _format_interviewer_text(interviewer_turns)
    
    logger.info(
        f"面试官发言: {len(interviewer_turns)} 轮 | "
        f"纯文本: {len(interviewer_text)} 字"
    )
    
    # 检查面试官的话是否超过上下文限制
    token_info = check_context_limit(interviewer_text)
    logger.info(
        f"面试官文本: 估算 {token_info['tokens']} tokens | "
        f"上下文使用率: {token_info['usage_pct']}%"
    )

    # ── 第二步：调用 LLM 从面试官的话中提取问题 ────────────────
    logger.info("第二步：调用 LLM 提取问题...")

    # ── 构建 Prompt ───────────────────────────────────────────
    prompt = _build_prompt(interviewer_text, interviewer_label)

    # ── 调用 LLM 或使用演示模式 ──────────────────────────────
    if use_demo:
        logger.info("使用演示模式生成示例问题（不调用 LLM API）")
        raw_output = _generate_demo_output(dialog, interviewer_label)
    else:
        with Timer("LLM 提取面试问题", logger):
            try:
                raw_output = llm_client.chat(prompt, temperature=0.1)
            except RuntimeError as e:
                if "quota" in str(e).lower() or "429" in str(e):
                    logger.warning(f"API 配额超出，切换到演示模式: {e}")
                    raw_output = _generate_demo_output(dialog, interviewer_label)
                else:
                    raise

    logger.info(f"输出预览:\n{truncate_text(raw_output, 300)}")

    # ── 解析输出 ──────────────────────────────────────────────
    questions, summary = _parse_llm_output(raw_output)

    result = {
        "audio_stem":      audio_stem,
        "total_questions": len(questions),
        "questions":       questions,
        "summary":         summary,
    }

    # ── 保存 JSON 和 Markdown ─────────────────────────────────
    save_json(result, output_json)
    _save_markdown(result, output_md)

    logger.info(
        f"提取完成 | 问题数: {len(questions)} | "
        f"报告已保存: {output_md}"
    )
    return result


# ── Prompt 构建 ───────────────────────────────────────────────

def _extract_interviewer_turns(dialog: list, interviewer_label: str) -> list:
    """
    从结构化对话中提取面试官的所有发言轮次。
    
    Args:
        dialog: 结构化对话列表
        interviewer_label: 面试官的标签（如"面试官"）
    
    Returns:
        面试官发言的轮次列表，每项包含 {start, end, text}
    """
    interviewer_turns = []
    for item in dialog:
        if item["speaker"] == interviewer_label:
            interviewer_turns.append({
                "start": item.get("start", 0),
                "end": item.get("end", 0),
                "text": item.get("text", ""),
            })
    logger.info(f"提取出面试官的 {len(interviewer_turns)} 轮发言")
    return interviewer_turns


def _format_interviewer_text(interviewer_turns: list) -> str:
    """
    将面试官的发言组织成纯文本，去除时间戳信息。
    每个问题占一行。
    
    Args:
        interviewer_turns: 面试官发言列表
    
    Returns:
        格式化的纯文本，每行一个问题
    """
    formatted_lines = []
    for i, turn in enumerate(interviewer_turns, 1):
        text = turn["text"].strip()
        if text:
            # 保留问题原文，不添加时间戳
            formatted_lines.append(f"{i}. {text}")
    
    formatted_text = "\n".join(formatted_lines)
    logger.info(f"面试官发言共 {len(formatted_lines)} 条")
    return formatted_text


# ── Prompt 构建 ───────────────────────────────────────────────

def _build_prompt(dialog_text: str, interviewer_label: str) -> str:
    """
    构建提取问题的 Prompt。
    优先从 prompts/extract_questions.txt 读取模板，文件不存在则用内置模板。
    """
    if Path(PROMPT_FILE).exists():
        template = load_text(PROMPT_FILE)
        return template.format(
            interviewer_label=interviewer_label,
            dialog_text=dialog_text,
        )

    # 内置默认 Prompt（针对面试官的纯文本）
    return f"""你是一位专业的面试分析师。下面是面试官在面试过程中提出的所有问题（已整理）。

请完成以下任务：
1. 分析这些问题，按逻辑关联性分组（相关的问题为同一组）
2. 提取每个逻辑组中最核心的问题（去除重复或相似的变体表述）
3. 保留原文，不要改写或合并
4. 最后用 2~3 句话总结本次面试的核心考察方向

严格按以下格式输出，不要添加任何额外内容：

### 面试问题列表
1. 问题原文
2. 问题原文
...

### 面试总结
总结内容

---
以下是面试官的所有提问：
{dialog_text}"""


# ── 输出解析 ──────────────────────────────────────────────────

def _parse_llm_output(raw: str) -> tuple:
    """
    解析 LLM 输出，提取问题列表和总结。

    Returns:
        (questions: list[dict], summary: str)
    """
    questions = []
    summary   = ""

    in_questions = False
    in_summary   = False

    for line in raw.split("\n"):
        line = line.strip()

        if "面试问题列表" in line:
            in_questions, in_summary = True, False
            continue
        if "面试总结" in line:
            in_questions, in_summary = False, True
            continue
        if line == "---":
            continue

        if in_questions:
            # 匹配 "1. [00:00:05] 请介绍一下..." 或 "1. 请介绍一下..."
            m = re.match(r"^(\d+)\.\s*(\[\d{2}:\d{2}:\d{2}\])?\s*(.+)$", line)
            if m:
                idx       = int(m.group(1))
                timestamp = m.group(2) or ""
                text      = m.group(3).strip()
                if text:
                    questions.append({
                        "index":     idx,
                        "text":      text,
                        "timestamp": timestamp,
                    })

        if in_summary and line and not line.startswith("#"):
            summary += line + " "

    return questions, summary.strip()


# ── 分块提取（超长对话降级方案）────────────────────────────────

def _extract_chunked(
    dialog, llm_client, audio_stem,
    interviewer_label, output_dir, output_json, output_md,
) -> dict:
    """超出上下文时，分块提取后合并结果"""
    chunks = chunk_dialog(dialog, max_tokens=80_000)
    logger.info(f"分块数: {len(chunks)}，逐块提取问题...")

    all_questions = []
    all_summaries = []

    for i, chunk in enumerate(chunks, 1):
        logger.info(f"处理第 {i}/{len(chunks)} 块...")
        chunk_text = dialog_to_text(chunk, with_timestamp=True)
        prompt     = _build_prompt(chunk_text, interviewer_label)
        raw        = llm_client.chat(prompt, temperature=0.1)
        questions, summary = _parse_llm_output(raw)
        all_questions.extend(questions)
        if summary:
            all_summaries.append(summary)

    # 重新编号
    for i, q in enumerate(all_questions, 1):
        q["index"] = i

    # 合并总结
    if len(all_summaries) > 1:
        merge_prompt = (
            "以下是同一场面试分段分析的总结，请整合为一段简洁的总结（2~3句）：\n\n"
            + "\n".join(f"{i+1}. {s}" for i, s in enumerate(all_summaries))
        )
        final_summary = llm_client.chat(merge_prompt, temperature=0.1)
    else:
        final_summary = all_summaries[0] if all_summaries else ""

    result = {
        "audio_stem":      audio_stem,
        "total_questions": len(all_questions),
        "questions":       all_questions,
        "summary":         final_summary,
    }

    save_json(result, output_json)
    _save_markdown(result, output_md)
    return result


# ── 对话导出 ───────────────────────────────────────────────────

def save_dialog_as_txt(
    dialog: list,
    audio_path: str,
    output_dir: str = OUTPUT_DIR,
) -> str:
    """
    将结构化对话导出为易读的 TXT 格式，每行按说话人标签显示。
    
    格式示例：
        面试官：你的工作背景如何？
        候选人：我有5年的工作经验...
        
    Args:
        dialog:      结构化对话列表
        audio_path:  音频文件路径（用于生成输出文件名）
        output_dir:  输出目录
        
    Returns:
        输出文件路径
    """
    audio_stem = Path(audio_path).stem
    output_txt = build_output_path(audio_path, "dialog", output_dir, ".txt")
    ensure_dirs(output_dir)
    
    lines = [
        f"# 对话记录：{audio_stem}",
        "",
        f"共 {len(dialog)} 轮对话",
        "",
        "=" * 80,
        "",
    ]
    
    for i, item in enumerate(dialog, 1):
        speaker = item.get("speaker", "未知")
        text    = item.get("text", "")
        start   = item.get("start", 0)
        end     = item.get("end", 0)
        
        # 时间戳（如果有）
        if start >= 0 and end > start:
            timestamp = f"[{int(start):02d}:{int(start) % 60:02d} - {int(end):02d}:{int(end) % 60:02d}]"
        else:
            timestamp = ""
        
        # 格式化输出
        if timestamp:
            lines.append(f"{speaker}：{text}  {timestamp}")
        else:
            lines.append(f"{speaker}：{text}")
    
    lines.append("")
    lines.append("=" * 80)
    
    content = "\n".join(lines)
    save_text(content, output_txt)
    logger.info(f"对话已保存为 TXT: {output_txt}")
    
    return output_txt


# ── Markdown 报告 ─────────────────────────────────────────────

def _save_markdown(result: dict, output_path: str) -> None:
    """生成 Markdown 格式的面试报告"""
    lines = [
        f"# 面试问题报告：{result['audio_stem']}",
        "",
        f"共提取 **{result['total_questions']}** 个问题",
        "",
        "## 问题列表",
        "",
    ]
    for q in result["questions"]:
        ts   = f"{q['timestamp']} " if q.get("timestamp") else ""
        lines.append(f"{q['index']}. {ts}{q['text']}")

    lines += [
        "",
        "## 面试总结",
        "",
        result.get("summary", ""),
    ]
    save_text("\n".join(lines), output_path)


# ── 演示模式：生成示例问题（无需 API 调用）─────────────────────

def _generate_demo_output(dialog: list, interviewer_label: str) -> str:
    """
    生成演示模式的虚构 LLM 输出。
    用于无 API 额度时展示流水线整体效果。
    
    Returns:
        模拟的 LLM 输出格式
    """
    # 收集面试官的真实发言
    interviewer_turns = _extract_interviewer_turns(dialog, interviewer_label)
    
    # 从面试官的真实问题中提取（去除重复或合并相似问题）
    demo_questions = []
    seen = set()
    
    for turn in interviewer_turns:
        text = turn["text"].strip()
        # 只取包含疑问句特征的文本
        if any(marker in text for marker in ["?", "吗", "呢", "阿", "啦"]) or text.endswith("："):
            # 简单的去重：检查前20个字符是否已seen
            key = text[:20]
            if key not in seen and len(demo_questions) < 12:  # 最多12个问题
                demo_questions.append(text)
                seen.add(key)
    
    # 如果真实问题不足，使用通用示例补充
    if len(demo_questions) < 3:
        demo_questions = [
            "请介绍一下你的工作背景和经验？",
            "在项目中你担任了什么角色？",
            "遇到过最大的技术挑战是什么？",
            "你是如何解决的？",
            "你觉得自己最大的优势是什么？",
            "为什么对这个职位感兴趣？",
        ]
    
    # 格式化输出
    demo_output = f"""### 面试问题列表
{chr(10).join(f"{i}. {q}" for i, q in enumerate(demo_questions, 1))}

### 面试总结
该面试主要考察了候选人的技术背景、项目经验和解决问题的能力。通过对其过往项目和技术挑战的深入探讨，评估其专业素养和职业发展潜力。

---
（演示模式生成，API 额度已用尽。实际生成请配置有效的 API 密钥并重试。）"""

    return demo_output