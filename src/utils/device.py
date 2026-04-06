"""
utils/device.py
设备检测与管理：自动选择 GPU/CPU，显存检查，模型推荐，环境报告
所有需要指定 device 的模块（transcribe、diarize）统一从这里获取
"""

import torch
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── 核心：获取设备 ────────────────────────────────────────────

def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    获取当前可用的最优设备。
    优先使用 GPU，不可用时自动降级到 CPU。

    Args:
        prefer_gpu: 是否优先使用 GPU，默认 True

    Returns:
        torch.device，直接传给模型的 .to(device)

    Usage:
        from src.utils.device import get_device
        device = get_device()
        model = model.to(device)
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        logger.info(f"使用 GPU: {name}（显存 {vram:.1f} GB）")
    else:
        device = torch.device("cpu")
        if prefer_gpu:
            logger.warning("未检测到可用 GPU，已降级到 CPU（转录速度较慢）")
        else:
            logger.info("已指定使用 CPU")
    return device


def get_device_str(prefer_gpu: bool = True) -> str:
    """
    返回设备字符串 "cuda" 或 "cpu"。
    部分库（如 Whisper）接受字符串而非 torch.device。

    Usage:
        model = whisper.load_model("large-v3", device=get_device_str())
    """
    return "cuda" if (prefer_gpu and torch.cuda.is_available()) else "cpu"


# ── 显存信息 ──────────────────────────────────────────────────

def get_vram_gb(gpu_index: int = 0) -> float:
    """
    获取指定 GPU 的总显存（GB）。
    GPU 不可用时返回 0.0。
    """
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(gpu_index).total_memory / 1024 ** 3


def get_free_vram_gb(gpu_index: int = 0) -> float:
    """
    获取指定 GPU 当前空闲显存（GB）。
    用于判断是否有足够显存加载模型。
    """
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    free_bytes, _ = torch.cuda.mem_get_info(gpu_index)
    return free_bytes / 1024 ** 3


def check_vram_sufficient(required_gb: float, gpu_index: int = 0) -> bool:
    """
    检查当前空闲显存是否满足需求。

    Args:
        required_gb: 需要的显存（GB）
        gpu_index:   GPU 编号

    Returns:
        True = 显存充足，False = 不足或无 GPU

    Usage:
        # large-v3 约需 10GB 显存
        if not check_vram_sufficient(10.0):
            logger.warning("显存不足，切换到 medium 模型")
    """
    free = get_free_vram_gb(gpu_index)
    sufficient = free >= required_gb
    if not sufficient:
        logger.warning(
            f"显存不足：需要 {required_gb:.1f} GB，当前空闲 {free:.1f} GB"
        )
    return sufficient


# ── Whisper 模型推荐 ──────────────────────────────────────────

# Whisper 各模型所需显存（GB，保守估计）
_WHISPER_VRAM_REQUIREMENTS = {
    "large-v3": 10.0,
    "large-v2": 10.0,
    "medium":    5.0,
    "small":     2.0,
    "base":      1.0,
    "tiny":      0.5,
}


def get_recommended_whisper_model() -> str:
    """
    根据可用显存自动推荐最优 Whisper 模型。

    显存映射：
        >= 40 GB (A100/H100)  →  large-v3  （最高精度）
        >= 10 GB              →  large-v3
        >= 5  GB              →  medium
        >= 2  GB              →  small
        CPU / 显存不足        →  base

    Returns:
        模型名称字符串，直接传给 whisper.load_model()

    Usage:
        model_size = get_recommended_whisper_model()
        model = whisper.load_model(model_size, device=get_device_str())
    """
    if not torch.cuda.is_available():
        logger.warning("无 GPU，推荐使用 base 模型（CPU 下 large 速度极慢）")
        return "base"

    vram = get_vram_gb()
    logger.info(f"总显存: {vram:.1f} GB，正在推荐 Whisper 模型...")

    if vram >= 10.0:
        model = "large-v3"
    elif vram >= 5.0:
        model = "medium"
    elif vram >= 2.0:
        model = "small"
    else:
        model = "base"

    logger.info(f"推荐模型: {model}（需要 {_WHISPER_VRAM_REQUIREMENTS[model]} GB 显存）")
    return model


def validate_whisper_model(model_size: str) -> str:
    """
    验证指定的 Whisper 模型能否在当前设备运行。
    如果显存不足，自动降级到合适的模型。

    Args:
        model_size: 期望使用的模型，如 "large-v3"

    Returns:
        实际可用的模型名称（可能被降级）

    Usage:
        # 用户指定 large-v3，但显存只有 4GB，自动降级到 medium
        actual_model = validate_whisper_model("large-v3")
    """
    if not torch.cuda.is_available():
        logger.warning(f"无 GPU，{model_size} 在 CPU 上速度极慢，已降级到 base")
        return "base"

    required = _WHISPER_VRAM_REQUIREMENTS.get(model_size, 10.0)
    free = get_free_vram_gb()

    if free >= required:
        logger.info(f"显存充足（{free:.1f} GB），使用 {model_size}")
        return model_size

    # 降级：找当前显存能跑的最大模型
    for model in ["large-v3", "medium", "small", "base", "tiny"]:
        if free >= _WHISPER_VRAM_REQUIREMENTS[model]:
            logger.warning(
                f"显存不足（空闲 {free:.1f} GB < 需要 {required:.1f} GB），"
                f"已从 {model_size} 降级到 {model}"
            )
            return model

    return "tiny"


# ── 环境报告 ──────────────────────────────────────────────────

def print_device_info() -> None:
    """
    打印完整的设备环境报告，项目启动时调用一次。
    包含：PyTorch 版本、CUDA 版本、GPU 列表、推荐模型。
    """
    logger.info("=" * 50)
    logger.info("设备环境报告")
    logger.info(f"PyTorch 版本: {torch.__version__}")
    logger.info(f"CUDA 可用:    {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA 版本:    {torch.version.cuda}")
        logger.info(f"GPU 数量:     {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total = props.total_memory / 1024 ** 3
            free  = get_free_vram_gb(i)
            logger.info(f"  GPU {i}: {props.name} | 总显存 {total:.1f} GB | 空闲 {free:.1f} GB")
        logger.info(f"推荐 Whisper: {get_recommended_whisper_model()}")
    else:
        logger.warning("  未检测到 GPU，所有模型将在 CPU 上运行")

    logger.info("=" * 50)


def get_device_summary() -> dict:
    """
    返回设备信息字典，用于日志存档或调试。

    Returns:
        {
            "cuda_available": bool,
            "device_str": str,
            "gpu_count": int,
            "gpu_name": str,
            "total_vram_gb": float,
            "free_vram_gb": float,
            "recommended_whisper": str,
        }
    """
    cuda = torch.cuda.is_available()
    return {
        "cuda_available":       cuda,
        "device_str":           "cuda" if cuda else "cpu",
        "gpu_count":            torch.cuda.device_count() if cuda else 0,
        "gpu_name":             torch.cuda.get_device_name(0) if cuda else "N/A",
        "total_vram_gb":        round(get_vram_gb(), 1),
        "free_vram_gb":         round(get_free_vram_gb(), 1),
        "recommended_whisper":  get_recommended_whisper_model(),
    }