"""
utils/cache.py
断点续跑缓存：基于音频文件 MD5 判断某步骤是否已处理，避免重复跑 Whisper
"""

from pathlib import Path
from src.utils.file_utils import get_md5, save_json, load_json, file_exists


class StepCache:
    """
    基于文件 MD5 的步骤缓存。
    同一个音频文件 + 同一个步骤名 → 唯一缓存文件。
    音频内容变了（MD5 变化）→ 自动视为新文件，重新处理。

    Usage:
        cache = StepCache("data/cache")

        # 在 runner.py 里这样用：
        if cache.exists("transcribe", audio_path):
            transcript = cache.load("transcribe", audio_path)
            logger.info("命中缓存，跳过转录")
        else:
            transcript = transcribe_audio(audio_path)
            cache.save("transcribe", audio_path, transcript)
    """

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, step: str, source_path: str) -> Path:
        """生成缓存文件路径：data/cache/{step}_{md5}.json"""
        if Path(source_path).exists():
            key = get_md5(source_path)
        else:
            # source_path 不是文件时（如直接传字符串 key），用字符串 hash
            import hashlib
            key = hashlib.md5(source_path.encode()).hexdigest()
        return self.cache_dir / f"{step}_{key}.json"

    def exists(self, step: str, source_path: str) -> bool:
        """检查该步骤的缓存是否存在"""
        return file_exists(str(self._cache_path(step, source_path)))

    def save(self, step: str, source_path: str, data: any) -> None:
        """保存步骤结果到缓存"""
        save_json(data, str(self._cache_path(step, source_path)))

    def load(self, step: str, source_path: str) -> any:
        """从缓存加载步骤结果"""
        return load_json(str(self._cache_path(step, source_path)))

    def clear(self, step: str = None) -> int:
        """
        清除缓存文件。
        step=None 清除全部；指定 step 只清除该步骤的缓存。

        Returns:
            删除的文件数量
        """
        pattern = f"{step}_*.json" if step else "*.json"
        deleted = 0
        for f in self.cache_dir.glob(pattern):
            f.unlink()
            deleted += 1
        return deleted

    def list_cached(self) -> list:
        """列出所有已缓存的步骤文件名（用于调试）"""
        return sorted([f.name for f in self.cache_dir.glob("*.json")])