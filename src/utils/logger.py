"""
utils/logger.py
统一日志模块 + 计时器
所有模块通过 get_logger(__name__) 获取 logger，格式统一，同时输出控制台和文件
"""

import logging
import time
from pathlib import Path
from datetime import datetime


def get_logger(name: str = "voice-note-ai", log_file: str = "logs/run.log") -> logging.Logger:
    """
    获取统一格式的 logger。

    Args:
        name:     logger 名称，推荐传 __name__
        log_file: 日志文件路径，None 则只输出控制台

    Usage:
        from src.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("开始转录...")
        logger.warning("显存不足，切换模型")
        logger.error("文件不存在")
    """
    logger = logging.getLogger(name)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)-5s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台 handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 文件 handler（自动创建目录）
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


class Timer:
    """
    上下文管理器计时器，自动打印耗时。

    Usage:
        with Timer("Whisper 转录", logger):
            result = transcribe_audio(...)
        # 输出: [Timer] Whisper 转录 耗时 3分12秒

        # 不传 logger 则直接 print
        with Timer("测试"):
            time.sleep(1)
    """

    def __init__(self, name: str = "", logger=None):
        self.name = name
        self.logger = logger
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        m, s = divmod(int(self.elapsed), 60)
        msg = f"[Timer] {self.name} 耗时 {m}分{s}秒" if self.name else f"[Timer] 耗时 {m}分{s}秒"
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    @staticmethod
    def now() -> str:
        """返回当前时间字符串，用于文件命名等场景"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")