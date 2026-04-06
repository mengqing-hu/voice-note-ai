"""
pipeline/runner.py
完整流水线入口：串联四个模块，集成缓存、计时、日志
- 单文件模式: run(audio_path)
- 批量模式:   run_batch(audio_dir)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from src.utils.logger import get_logger, Timer
from src.utils.file_utils import list_files, ensure_project_dirs
from src.utils.device import print_device_info
from src.utils.cache import StepCache
from src.utils.llm_client import LLMClient

from src.pipeline.transcribe import transcribe
from src.pipeline.diarize import diarize
from src.pipeline.postprocess import postprocess, get_speaker_stats
from src.pipeline.extract_questions import extract_questions

load_dotenv()
logger = get_logger(__name__)


def run(
    audio_path: str,
    speaker_mapping: dict = None,
    model_size: str = "large-v3",
    language: str = "zh",
    interviewer_label: str = "面试官",
    force: bool = False,
    use_cache: bool = True,
) -> dict:
    """
    单文件完整流水线：转录 → 说话人分离 → 结构化 → 提取问题。
    集成 StepCache 断点续跑，中断后重跑会自动跳过已完成的步骤。

    Args:
        audio_path:        音频文件路径
        speaker_mapping:   说话人重命名，None 时自动推断
                           如 {"SPEAKER_00": "面试官", "SPEAKER_01": "候选人"}
        model_size:        Whisper 模型，会根据显存自动降级
        language:          音频语言
        interviewer_label: 面试官的说话人名称，与 speaker_mapping 的 value 一致
        force:             True = 所有步骤强制重新执行
        use_cache:         True = 启用 StepCache 断点续跑

    Returns:
        {
            "audio_path":  str,
            "transcript":  dict,
            "diarization": list,
            "dialog":      list,
            "result":      dict,   # 问题列表 + 总结
        }

    Usage:
        from src.pipeline.runner import run

        output = run(
            audio_path="data/raw/interview_001.wav",
            speaker_mapping={"SPEAKER_00": "面试官", "SPEAKER_01": "候选人"},
        )
        print(f"共提取 {output['result']['total_questions']} 个问题")
    """
    audio_path = str(audio_path)
    cache = StepCache("data/cache") if use_cache else None

    # ── 环境初始化 ────────────────────────────────────────────
    ensure_project_dirs()
    _check_env()

    hf_token    = os.getenv("HF_TOKEN")
    gemini_key  = os.getenv("GEMINI_API_KEY")
    llm_client  = LLMClient(api_key=gemini_key)

    logger.info("=" * 55)
    logger.info(f"开始处理: {Path(audio_path).name}")
    logger.info("=" * 55)

    with Timer("完整流水线", logger):

        # ── Step 1: 转录 ──────────────────────────────────────
        logger.info("[Step 1/4] Whisper 语音转录")
        if use_cache and not force and cache.exists("transcribe", audio_path):
            logger.info("  命中缓存，跳过转录")
            transcript_data = cache.load("transcribe", audio_path)
        else:
            transcript_data = transcribe(
                audio_path,
                model_size=model_size,
                language=language,
                force=force,
            )
            if use_cache:
                cache.save("transcribe", audio_path, transcript_data)

        # ── Step 2: 说话人分离 ────────────────────────────────
        logger.info("[Step 2/4] 说话人分离")
        if use_cache and not force and cache.exists("diarize", audio_path):
            logger.info("  命中缓存，跳过分离")
            diarization_data = cache.load("diarize", audio_path)
        else:
            diarization_data = diarize(
                audio_path,
                hf_token=hf_token,
                num_speakers=len(speaker_mapping) if speaker_mapping else 2,
                force=force,
            )
            if use_cache:
                cache.save("diarize", audio_path, diarization_data)

        # ── Step 3: 结构化对话 ────────────────────────────────
        logger.info("[Step 3/4] 对话结构化")

        # 未指定 mapping 时自动生成默认值
        if speaker_mapping is None:
            speakers = sorted(set(s["speaker"] for s in diarization_data))
            speaker_mapping = {
                speakers[0]: "面试官",
                speakers[1]: "候选人",
            } if len(speakers) >= 2 else {}
            logger.info(f"  自动生成说话人映射: {speaker_mapping}")

        dialog_data = postprocess(
            transcript_data,
            diarization_data,
            speaker_mapping=speaker_mapping,
            force=force,
        )

        # 打印说话人统计，方便验证分配是否正确
        stats = get_speaker_stats(dialog_data)
        for sp, s in stats.items():
            logger.info(f"  {sp}: {s['turns']} 轮，{s['duration']:.0f}s")

        # ── Step 4: 提取问题 ──────────────────────────────────
        logger.info("[Step 4/4] LLM 提取面试问题")
        result_data = extract_questions(
            dialog_data,
            llm_client=llm_client,
            audio_path=audio_path,
            interviewer_label=interviewer_label,
            force=force,
        )

    llm_client.log_usage()
    logger.info(f"全部完成！共提取 {result_data['total_questions']} 个问题")
    logger.info(f"报告位置: data/outputs/{Path(audio_path).stem}_report.md")

    return {
        "audio_path":  audio_path,
        "transcript":  transcript_data,
        "diarization": diarization_data,
        "dialog":      dialog_data,
        "result":      result_data,
    }


def run_batch(
    audio_dir: str = "data/raw",
    **kwargs,
) -> list:
    """
    批量处理目录下的所有音频文件。
    每个文件独立运行，单个失败不影响其他文件继续处理。

    Args:
        audio_dir: 音频目录，默认 data/raw
        **kwargs:  透传给 run() 的所有参数

    Returns:
        每个文件的处理结果列表（失败的条目包含 error 字段）

    Usage:
        from src.pipeline.runner import run_batch
        results = run_batch("data/raw")
    """
    audio_files = list_files(audio_dir, suffixes=(".wav", ".mp3", ".m4a", ".flac"))

    if not audio_files:
        logger.warning(f"目录 {audio_dir} 下未找到音频文件")
        return []

    logger.info(f"批量处理：共 {len(audio_files)} 个文件")
    outputs = []

    for i, audio_path in enumerate(audio_files, 1):
        logger.info(f"\n{'='*55}")
        logger.info(f"[{i}/{len(audio_files)}] {Path(audio_path).name}")
        try:
            output = run(audio_path, **kwargs)
            outputs.append(output)
        except Exception as e:
            logger.error(f"处理失败: {audio_path}  错误: {e}")
            outputs.append({"audio_path": audio_path, "error": str(e)})

    success = sum(1 for o in outputs if "error" not in o)
    logger.info(f"\n批量处理完成：{success}/{len(audio_files)} 成功")
    return outputs


def run_pipeline(
    audio_path: str,
    gemini_api_key: str = None,
    hf_token: str = None,
    speaker_mapping: dict = None,
    model_size: str = "large-v3",
    language: str = "zh",
    interviewer_label: str = "面试官",
    force: bool = False,
    use_cache: bool = True,
) -> dict:
    """
    完整流水线的便利别名（推荐在 notebook 中使用）。

    Args:
        audio_path:        音频文件路径
        gemini_api_key:    Gemini API Key（可选，优先使用环境变量）
        hf_token:          HuggingFace Token（可选，优先使用环境变量）
        speaker_mapping:   说话人重命名字典
        model_size:        Whisper 模型大小
        language:          音频语言
        interviewer_label: 面试官标签
        force:             是否强制重新处理
        use_cache:         是否使用缓存

    Returns:
        处理结果（包含转录、分离、结构化、提取问题）

    Usage:
        from src.pipeline.runner import run_pipeline
        result = run_pipeline("data/raw/interview.wav")
    """
    return run(
        audio_path=audio_path,
        speaker_mapping=speaker_mapping,
        model_size=model_size,
        language=language,
        interviewer_label=interviewer_label,
        force=force,
        use_cache=use_cache,
    )


# ── 内部工具 ──────────────────────────────────────────────────

def _check_env() -> None:
    """检查必要的环境变量，缺失时提前报错"""
    missing = []
    if not os.getenv("GEMINI_API_KEY"):
        missing.append("GEMINI_API_KEY")
    if not os.getenv("HF_TOKEN"):
        missing.append("HF_TOKEN")
    if missing:
        raise EnvironmentError(
            f"缺少环境变量: {missing}，请检查 .env 文件"
        )