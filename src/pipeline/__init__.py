"""
pipeline/__init__.py
快捷导入：from src.pipeline import run, run_batch
"""

from src.pipeline.runner import run, run_batch
from src.pipeline.transcribe import transcribe
from src.pipeline.diarize import diarize
from src.pipeline.postprocess import postprocess, preview_dialog, get_speaker_stats
from src.pipeline.extract_questions import extract_questions

__all__ = [
    "run", "run_batch",
    "transcribe", "diarize", "postprocess", "extract_questions",
    "preview_dialog", "get_speaker_stats",
]