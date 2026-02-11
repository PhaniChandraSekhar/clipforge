"""Pipeline configuration using pydantic-settings."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class PipelineConfig(BaseSettings):
    """Configuration for the ClipForge pipeline."""

    input_video: Path
    output_dir: Path = Path(".")
    whisper_model: str = "base"
    ollama_model: str = "llama3.1:8b"
    min_clip_duration: int = 30
    max_clip_duration: int = 600
    language: Optional[str] = None
    skip_review: bool = False
    resume: bool = False
    device: str = "auto"
    verbose: bool = False
