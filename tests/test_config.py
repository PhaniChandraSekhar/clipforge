"""Tests for clipforge.config."""
from pathlib import Path

from clipforge.config import PipelineConfig


def test_default_config(tmp_video):
    config = PipelineConfig(input_video=tmp_video)
    assert config.whisper_model == "base"
    assert config.ollama_model == "llama3.1:8b"
    assert config.min_clip_duration == 30
    assert config.max_clip_duration == 600
    assert config.skip_review is False
    assert config.resume is False
    assert config.device == "auto"
    assert config.verbose is False


def test_custom_config(tmp_video, tmp_path):
    config = PipelineConfig(
        input_video=tmp_video,
        output_dir=tmp_path / "custom_output",
        whisper_model="large",
        ollama_model="mistral:7b",
        min_clip_duration=15,
        max_clip_duration=120,
        language="en",
        skip_review=True,
        device="cuda",
        verbose=True,
    )
    assert config.whisper_model == "large"
    assert config.language == "en"
    assert config.skip_review is True
    assert config.verbose is True
