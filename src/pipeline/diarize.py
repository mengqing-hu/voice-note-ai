"""
pipeline/diarize.py
模块2：说话人分离（Speaker Diarization）
- 输入: 音频文件路径
- 输出: 每段时间对应的说话人标签，保存到 data/transcripts/
依赖: pyannote.audio 3.x（需要 HuggingFace Token）
"""

from pathlib import Path
import torch
import librosa

from src.utils.logger import get_logger, Timer
from src.utils.file_utils import file_exists, save_json, load_json, build_output_path, ensure_dirs
from src.utils.device import get_device
from src.utils.audio_utils import check_audio_file

logger = get_logger(__name__)

OUTPUT_DIR = "data/transcripts"


def extract_annotation(diarization):
    """
    兼容 pyannote 2.x 和 3.x
    """
    if hasattr(diarization, "speaker_diarization"):
        return diarization.speaker_diarization  # pyannote 3.x
    return diarization  # pyannote 2.x


def diarize(
    audio_path: str,
    hf_token: str,
    num_speakers: int = 2,
    output_dir: str = OUTPUT_DIR,
    force: bool = False,
) -> list:

    audio_path = str(audio_path)
    output_path = build_output_path(audio_path, "diarization", output_dir, ".json")
    ensure_dirs(output_dir)

    # ── 缓存检查 ─────────────────────────────────────────────
    if not force and file_exists(output_path):
        logger.info(f"已存在说话人分离结果，跳过（force=False）: {output_path}")
        return load_json(output_path)

    # ── 音频检查 ─────────────────────────────────────────────
    info = check_audio_file(audio_path)
    if not info["exists"]:
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    logger.info(f"音频文件: {Path(audio_path).name}  时长: {info['duration_str']}")

    # ── 加载 pipeline ────────────────────────────────────────
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        raise ImportError("未安装 pyannote.audio，请运行: pip install pyannote.audio")

    device = get_device()

    with Timer("说话人分离", logger):
        logger.info("加载 pyannote/speaker-diarization-3.1 ...")

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
        )
        pipeline = pipeline.to(device)

        logger.info(f"开始说话人分离，指定说话人数: {num_speakers}")

        # ── 预加载音频（绕过 torchcodec）──────────────────────
        logger.info("预加载音频文件...")

        try:
            waveform, sample_rate = librosa.load(audio_path, sr=None, mono=False)

            # 转 torch tensor
            if not isinstance(waveform, torch.Tensor):
                waveform = torch.from_numpy(waveform).float()

            # 保证 shape = (channel, time)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)

            audio_dict = {
                "waveform": waveform,
                "sample_rate": sample_rate,
            }

            diarization = pipeline(audio_dict, num_speakers=num_speakers)

        except Exception as e:
            logger.warning(f"预加载失败，回退到文件路径方式: {e}")
            diarization = pipeline(audio_path, num_speakers=num_speakers)

    # ── 提取 annotation（关键修复）────────────────────────────
    annotation = extract_annotation(diarization)

    # ── 转换为标准 segments ─────────────────────────────────
    segments = []

    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append({
            "start": round(turn.start, 2),
            "end": round(turn.end, 2),
            "speaker": speaker,
        })

    speakers = sorted(set(s["speaker"] for s in segments))

    # ── 保存结果 ────────────────────────────────────────────
    save_json(segments, output_path)

    logger.info(
        f"说话人分离完成 | 说话人: {speakers} | "
        f"片段数: {len(segments)} | 已保存: {output_path}"
    )

    return segments


def diarize_audio(
    audio_path: str,
    hf_token: str,
    num_speakers: int = 2,
    output_dir: str = OUTPUT_DIR,
    force: bool = False,
) -> list:
    """
    Notebook 友好调用入口
    """
    return diarize(audio_path, hf_token, num_speakers, output_dir, force)