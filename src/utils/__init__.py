"""
utils/__init__.py
统一导出，方便在业务模块中简洁导入

Usage:
    from src.utils import get_logger, Timer
    from src.utils import save_json, load_json, file_exists, build_output_path
    from src.utils import LLMClient, build_client_from_env
    from src.utils import get_duration, get_recommended_whisper_model
    from src.utils import clean_text, dialog_to_text, estimate_tokens
    from src.utils import StepCache
"""

from src.utils.logger import get_logger, Timer
from src.utils.file_utils import (
    ensure_dirs,
    ensure_project_dirs,
    file_exists,
    get_stem,
    get_md5,
    list_files,
    save_json,
    load_json,
    save_text,
    load_text,
    build_output_path,
)
from src.utils.cache import StepCache
from src.utils.audio_utils import (
    get_duration,
    format_duration,
    check_audio_file,
    convert_to_wav,
    split_audio,
)
from src.utils.device import (
    get_device,
    get_device_str,
    get_vram_gb,
    get_free_vram_gb,
    check_vram_sufficient,
    get_recommended_whisper_model,
    validate_whisper_model,
    print_device_info,
    get_device_summary,
)
from src.utils.text_utils import (
    clean_text,
    clean_segments,
    estimate_tokens,
    check_context_limit,
    seconds_to_timestamp,
    dialog_to_text,
    chunk_dialog,
    truncate_text,
)
from src.utils.llm_client import LLMClient, build_client_from_env

__all__ = [
    "get_logger", "Timer",
    "ensure_dirs", "ensure_project_dirs", "file_exists", "get_stem", "get_md5",
    "list_files", "save_json", "load_json", "save_text", "load_text", "build_output_path",
    "StepCache",
    "get_duration", "format_duration", "check_audio_file", "convert_to_wav", "split_audio",
    "get_device", "get_device_str", "get_vram_gb", "get_free_vram_gb",
    "check_vram_sufficient", "get_recommended_whisper_model", "validate_whisper_model",
    "print_device_info", "get_device_summary",
    "clean_text", "clean_segments", "estimate_tokens", "check_context_limit",
    "seconds_to_timestamp", "dialog_to_text", "chunk_dialog", "truncate_text",
    "LLMClient", "build_client_from_env",
]